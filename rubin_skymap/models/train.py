"""LightGBM multi-class training pipeline for rubin-skymap.

Usage
-----
Called by ``scripts/train_model.py`` or imported directly::

    from rubin_skymap.models.train import train_model
    metrics = train_model(data_parquet, model_path, label_map_path, params)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

from rubin_skymap.features.lightcurve import FEATURE_NAMES, extract_features
from rubin_skymap.ingest.schemas import Alert

_log = logging.getLogger(__name__)

# Canonical label mapping: class name → integer target.
LABEL_TO_INT: dict[str, int] = {
    "SN Ia": 0,
    "SN II": 1,
    "SN Ibc": 2,
    "SLSN": 3,
    "Kilonova": 4,
    "AGN": 5,
    "RRL": 6,
    "Other": 7,
}
INT_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_TO_INT.items()}


def _build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix X and label vector y from the raw parquet frame.

    Parameters
    ----------
    df:
        DataFrame with columns ``[object_id, class, jd, mag, magerr, filter]``
        where each row represents one observation (long format).

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        ``(X, y)`` where X has columns from ``FEATURE_NAMES`` and y contains
        integer class labels.
    """
    rows: list[dict[str, float]] = []
    labels: list[int] = []

    for oid, group in df.groupby("object_id", sort=False):
        cls_name: str = str(group["class"].iloc[0])
        try:
            alert = Alert(
                object_id=str(oid),
                ra=0.0,
                dec=0.0,
                jd=group["jd"].tolist(),
                mag=group["mag"].tolist(),
                magerr=group["magerr"].tolist(),
                filter=group["filter"].tolist(),
            )
        except Exception as exc:
            _log.debug("Skipping object %s: %s", oid, exc)
            continue

        feats = extract_features(alert)
        rows.append(feats)
        int_label = LABEL_TO_INT.get(cls_name, LABEL_TO_INT["Other"])
        labels.append(int_label)

    X = pd.DataFrame(rows, columns=FEATURE_NAMES)
    y = pd.Series(labels, name="target", dtype=int)
    return X, y


def train_model(
    data_parquet: str,
    model_path: str,
    label_map_path: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Train a LightGBM multi-class classifier on the processed parquet file.

    Parameters
    ----------
    data_parquet:
        Path to the parquet file with columns
        ``[object_id, class, jd, mag, magerr, filter]``.
    model_path:
        Destination path for the saved LightGBM text model (e.g.
        ``"models/lgbm_v1.txt"``).
    label_map_path:
        Destination path for the JSON label map (e.g.
        ``"models/label_map.json"``).
    params:
        LightGBM hyperparameter dict (keys: ``num_leaves``,
        ``learning_rate``, ``n_estimators``, ``random_state``).

    Returns
    -------
    dict[str, Any]
        Dictionary with keys ``macro_f1``, ``per_class_f1``, and
        ``confusion_matrix``.
    """
    _log.info("Loading parquet: %s", data_parquet)
    df = pd.read_parquet(data_parquet)

    _log.info("Building feature matrix …")
    X, y = _build_feature_matrix(df)
    _log.info("Feature matrix: %d rows, %d features.", len(X), len(X.columns))

    # Drop rows with >50 % NaN features.
    nan_frac = X.isnull().mean(axis=1)
    keep_mask = nan_frac <= 0.5
    X = X.loc[keep_mask].reset_index(drop=True)
    y = y.loc[keep_mask].reset_index(drop=True)
    _log.info("After NaN filter: %d rows.", len(X))

    # Fill remaining NaNs with column medians.
    col_medians = X.median()
    X = X.fillna(col_medians)

    # Stratified split.
    random_state = int(params.get("random_state", 42))
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(X, y))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Compute class weights.
    class_counts = y_train.value_counts()
    total = float(len(y_train))
    n_classes = len(LABEL_TO_INT)
    weight_dict: dict[int, float] = {}
    for cls_int in range(n_classes):
        cnt = float(class_counts.get(cls_int, 1))
        weight_dict[cls_int] = total / (n_classes * cnt)
    sample_weights = np.array([weight_dict.get(int(lbl), 1.0) for lbl in y_train])

    # Build LightGBM datasets.
    lgb_train = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    lgb_valid = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

    lgb_params: dict[str, Any] = {
        "objective": "multiclass",
        "num_class": n_classes,
        "num_leaves": int(params.get("num_leaves", 31)),
        "learning_rate": float(params.get("learning_rate", 0.05)),
        "verbose": -1,
        "seed": random_state,
        "metric": "multi_logloss",
    }

    n_estimators = int(params.get("n_estimators", 300))

    _log.info("Training LightGBM (n_estimators=%d) …", n_estimators)

    callbacks = [lgb.log_evaluation(period=50), lgb.early_stopping(50, verbose=False)]
    booster = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=n_estimators,
        valid_sets=[lgb_valid],
        callbacks=callbacks,
    )

    # Evaluate.
    y_pred_proba = booster.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    # Only include classes that actually appear in y_test to avoid sklearn mismatch.
    present_labels = sorted(y_test.unique().tolist())
    present_names = [INT_TO_LABEL[i] for i in present_labels]

    per_class_report = classification_report(
        y_test,
        y_pred,
        labels=present_labels,
        target_names=present_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=present_labels).tolist()

    per_class_f1: dict[str, float] = {
        INT_TO_LABEL[i]: float(
            per_class_report.get(INT_TO_LABEL[i], {}).get("f1-score", 0.0)
        )
        for i in range(n_classes)
    }

    # Save model.
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(model_path)
    _log.info("Saved model → %s", model_path)

    # Save label map.
    label_map_data = {
        "int_to_label": {str(k): v for k, v in INT_TO_LABEL.items()},
        "label_to_int": LABEL_TO_INT,
    }
    Path(label_map_path).parent.mkdir(parents=True, exist_ok=True)
    with open(label_map_path, "w", encoding="utf-8") as fh:
        json.dump(label_map_data, fh, indent=2)
    _log.info("Saved label map → %s", label_map_path)

    # Save training features for drift monitoring.
    feat_parquet = "data/processed/features_train.parquet"
    Path(feat_parquet).parent.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(feat_parquet, index=False)
    _log.info("Saved training features → %s", feat_parquet)

    metrics: dict[str, Any] = {
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
    }
    _log.info("Training complete. macro_f1=%.4f", macro_f1)
    return metrics
