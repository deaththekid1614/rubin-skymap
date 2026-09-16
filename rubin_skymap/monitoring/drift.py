"""Feature drift monitoring using the Kolmogorov-Smirnov statistic.

Compares the distribution of features in a reference training dataset against
a rolling window of live predictions.  Exposes
:func:`feature_drift` as a pure function and :func:`compute_drift_for_stats`
as the integration point for the ``/api/stats`` endpoint.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import ks_2samp

from rubin_skymap.features.lightcurve import FEATURE_NAMES

_log = logging.getLogger(__name__)

_REFERENCE_PARQUET = Path("data/processed/features_train.parquet")
_LIVE_BUFFER_SIZE = 200


def feature_drift(
    train_df: pd.DataFrame,
    live_df: pd.DataFrame,
    cols: list[str],
) -> dict[str, float]:
    """Compute per-feature KS statistics between training and live distributions.

    Parameters
    ----------
    train_df:
        Reference feature DataFrame (training distribution).
    live_df:
        Live feature DataFrame (rolling window of recent predictions).
    cols:
        Column names to compare.  Missing columns are silently skipped.

    Returns
    -------
    dict[str, float]
        Mapping of feature name → KS statistic (0 = identical, 1 = maximally
        different).  Returns ``{}`` if either DataFrame is empty.
    """
    if train_df.empty or live_df.empty:
        return {}

    result: dict[str, float] = {}
    for col in cols:
        if col not in train_df.columns or col not in live_df.columns:
            continue
        train_vals = train_df[col].dropna().to_numpy()
        live_vals = live_df[col].dropna().to_numpy()
        if len(train_vals) < 2 or len(live_vals) < 2:
            continue
        ks_stat, _ = ks_2samp(train_vals, live_vals)
        result[col] = float(ks_stat)

    return result


def compute_drift_for_stats(db_url: str) -> dict[str, Any]:
    """Compute drift scores for the ``/api/stats`` endpoint.

    Reads the last :data:`_LIVE_BUFFER_SIZE` rows from the database, parses
    their ``features_json`` columns into a DataFrame, and compares against the
    cached training feature parquet (if it exists).

    Parameters
    ----------
    db_url:
        SQLAlchemy database URL (used to read recent predictions).

    Returns
    -------
    dict
        Either a mapping of feature → KS statistic, or a dict with a
        ``"note"`` key explaining why drift cannot be computed.
    """
    if not _REFERENCE_PARQUET.exists():
        return {"note": "no reference data"}

    try:
        train_df = pd.read_parquet(_REFERENCE_PARQUET)
    except Exception as exc:
        _log.warning("Could not load reference parquet: %s", exc)
        return {"note": "reference data unreadable"}

    # Read recent rows from DB.
    try:
        from rubin_skymap.db.session import latest_alerts

        rows = latest_alerts(db_url, limit=_LIVE_BUFFER_SIZE)
    except Exception as exc:
        _log.warning("Could not read live alerts for drift: %s", exc)
        return {"note": "db unavailable"}

    if not rows:
        return {"note": "no live data yet"}

    live_records: list[dict[str, float]] = []
    for row in rows:
        try:
            feats = json.loads(row.get("features_json", "{}"))
            if isinstance(feats, dict):
                live_records.append({k: feats.get(k, float("nan")) for k in FEATURE_NAMES})
        except (json.JSONDecodeError, TypeError):
            continue

    if not live_records:
        return {"note": "no parseable live features"}

    live_df = pd.DataFrame(live_records, columns=FEATURE_NAMES)
    return feature_drift(train_df, live_df, FEATURE_NAMES)
