"""LightGBM inference module.

Loads the trained booster once and exposes :meth:`Predictor.predict` and
:meth:`Predictor.predict_proba` for single-alert inference.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from rubin_skymap.features.lightcurve import FEATURE_NAMES

_log = logging.getLogger(__name__)

# Module-level cache so multiple callers share the same booster instance.
_predictor_instance: "Predictor | None" = None


class Predictor:
    """Wraps a trained LightGBM booster for single-alert inference.

    Parameters
    ----------
    model_path:
        Path to the LightGBM text model file (``*.txt``).
    label_map_path:
        Path to the JSON label map produced by :func:`~rubin_skymap.models.train.train_model`.
    """

    def __init__(self, model_path: str, label_map_path: str) -> None:
        _log.info("Loading model from %s", model_path)
        self.booster: lgb.Booster = lgb.Booster(model_file=model_path)

        with open(label_map_path, "r", encoding="utf-8") as fh:
            lmap = json.load(fh)

        # int_to_label: keys are stored as strings in JSON.
        self.int_to_label: dict[int, str] = {
            int(k): str(v) for k, v in lmap["int_to_label"].items()
        }
        self.label_to_int: dict[str, int] = {
            str(k): int(v) for k, v in lmap["label_to_int"].items()
        }
        self.n_classes: int = len(self.int_to_label)
        _log.info("Predictor ready. Classes: %s", list(self.int_to_label.values()))

    def _features_to_array(self, features: dict[str, Any]) -> pd.DataFrame:
        """Convert a feature dict to a single-row DataFrame in the correct column order.

        Parameters
        ----------
        features:
            Mapping of feature name → value (may contain ``nan``).

        Returns
        -------
        pd.DataFrame
            Single row with columns matching ``FEATURE_NAMES``.
        """
        row = {k: features.get(k, float("nan")) for k in FEATURE_NAMES}
        df = pd.DataFrame([row], columns=FEATURE_NAMES)
        # Replace NaN with column medians (0 for a single row — use 0 as fallback).
        df = df.fillna(0.0)
        return df

    def predict_proba(self, features: dict[str, Any]) -> dict[str, float]:
        """Return a probability distribution over all classes.

        Parameters
        ----------
        features:
            Feature dict as returned by :func:`~rubin_skymap.features.lightcurve.extract_features`.

        Returns
        -------
        dict[str, float]
            Mapping of class name → probability (sum ≈ 1.0).
        """
        X = self._features_to_array(features)
        proba: np.ndarray = self.booster.predict(X)  # shape (1, n_classes)
        proba_row: np.ndarray = proba[0]
        return {
            self.int_to_label[i]: float(proba_row[i])
            for i in range(self.n_classes)
        }

    def predict(self, features: dict[str, Any]) -> tuple[str, float]:
        """Return the most probable class and its probability.

        Parameters
        ----------
        features:
            Feature dict as returned by :func:`~rubin_skymap.features.lightcurve.extract_features`.

        Returns
        -------
        tuple[str, float]
            ``(label, probability)`` where *probability* is in ``[0, 1]``.
        """
        proba_dict = self.predict_proba(features)
        best_label = max(proba_dict, key=lambda k: proba_dict[k])
        return best_label, proba_dict[best_label]


def get_predictor(cfg: dict[str, Any]) -> Predictor:
    """Return the module-level singleton Predictor, creating it if needed.

    Parameters
    ----------
    cfg:
        Loaded configuration dict (as returned by
        :func:`~rubin_skymap.config.load_config`).

    Returns
    -------
    Predictor
    """
    global _predictor_instance
    if _predictor_instance is None:
        from rubin_skymap.config import cfg_get

        model_path = cfg_get(cfg, "model.model_path", "models/lgbm_v1.txt")
        label_map_path = cfg_get(cfg, "model.label_map_path", "models/label_map.json")
        _predictor_instance = Predictor(model_path=model_path, label_map_path=label_map_path)
    return _predictor_instance
