"""Tests for the model training and prediction pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from rubin_skymap.features.lightcurve import FEATURE_NAMES, extract_features
from rubin_skymap.ingest.mock_source import _TEMPLATES, MockSource, _generate_lc
from rubin_skymap.models.train import INT_TO_LABEL, LABEL_TO_INT, train_model

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tiny_parquet(tmp_path: Path, n_objects: int = 120) -> Path:
    """Build a tiny synthetic parquet file for fast training tests.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    n_objects:
        Total number of synthetic objects to generate.

    Returns
    -------
    Path
        Path to the written parquet file.
    """
    import uuid

    classes = list(_TEMPLATES.keys())
    rng = np.random.default_rng(0)
    records = []
    per_class = n_objects // len(classes)

    for cls_name in classes:
        tmpl = _TEMPLATES[cls_name]
        for _ in range(per_class):
            jd, mag, magerr, flt = _generate_lc(cls_name, tmpl, rng)
            oid = f"TEST-{cls_name.replace(' ', '')}-{uuid.uuid4().hex[:6]}"
            for j, m, me, f in zip(jd, mag, magerr, flt):
                records.append({
                    "object_id": oid,
                    "class": cls_name,
                    "ra": float(rng.uniform(0, 24)),
                    "dec": float(rng.uniform(-90, 90)),
                    "jd": j,
                    "mag": m,
                    "magerr": me,
                    "filter": f,
                })

    df = pd.DataFrame(records)
    out = tmp_path / "tiny_train.parquet"
    df.to_parquet(str(out), index=False)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_train_model_produces_files(tmp_path: Path) -> None:
    """train_model should create model and label map files."""
    parquet_path = _make_tiny_parquet(tmp_path)
    model_path = str(tmp_path / "lgbm_test.txt")
    label_map_path = str(tmp_path / "label_map_test.json")

    params = {
        "num_leaves": 8,
        "learning_rate": 0.1,
        "n_estimators": 20,
        "random_state": 42,
    }

    metrics = train_model(
        data_parquet=str(parquet_path),
        model_path=model_path,
        label_map_path=label_map_path,
        params=params,
    )

    assert Path(model_path).exists(), "Model file not created"
    assert Path(label_map_path).exists(), "Label map not created"
    assert "macro_f1" in metrics
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_predictor_returns_valid_output(tmp_path: Path) -> None:
    """Predictor.predict should return a known label and probability in [0,1]."""
    parquet_path = _make_tiny_parquet(tmp_path, n_objects=120)
    model_path = str(tmp_path / "lgbm_pred_test.txt")
    label_map_path = str(tmp_path / "label_map_pred_test.json")

    params = {"num_leaves": 8, "learning_rate": 0.1, "n_estimators": 20, "random_state": 0}
    train_model(str(parquet_path), model_path, label_map_path, params)

    from rubin_skymap.models.predict import Predictor

    predictor = Predictor(model_path=model_path, label_map_path=label_map_path)

    # Generate 5 test alerts and predict.
    source = MockSource(n_per_batch=5, seed=99)
    alerts = list(source)
    assert len(alerts) == 5

    known_classes = set(LABEL_TO_INT.keys())

    for alert in alerts:
        feats = extract_features(alert)
        label, prob = predictor.predict(feats)
        assert label in known_classes, f"Unknown label: {label}"
        assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"

        proba_dict = predictor.predict_proba(feats)
        assert set(proba_dict.keys()) == set(INT_TO_LABEL.values())
        total = sum(proba_dict.values())
        assert abs(total - 1.0) < 0.01, f"Probabilities don't sum to 1: {total}"


def test_label_map_roundtrip() -> None:
    """LABEL_TO_INT and INT_TO_LABEL should be consistent inverses."""
    for label, idx in LABEL_TO_INT.items():
        assert INT_TO_LABEL[idx] == label, f"Roundtrip failed for '{label}'"
    for idx, label in INT_TO_LABEL.items():
        assert LABEL_TO_INT[label] == idx, f"Roundtrip failed for idx {idx}"


def test_predict_with_all_nan_features(tmp_path: Path) -> None:
    """Predictor must not raise on all-zero features (NaN filled to 0)."""
    parquet_path = _make_tiny_parquet(tmp_path, n_objects=120)
    model_path = str(tmp_path / "lgbm_nan_test.txt")
    label_map_path = str(tmp_path / "label_map_nan_test.json")
    params = {"num_leaves": 8, "learning_rate": 0.1, "n_estimators": 20, "random_state": 0}
    train_model(str(parquet_path), model_path, label_map_path, params)

    from rubin_skymap.models.predict import Predictor

    predictor = Predictor(model_path=model_path, label_map_path=label_map_path)

    nan_feats = {k: float("nan") for k in FEATURE_NAMES}
    label, prob = predictor.predict(nan_feats)
    assert isinstance(label, str)
    assert 0.0 <= prob <= 1.0
