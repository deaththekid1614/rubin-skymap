"""SHAP-based explainability for the LightGBM classifier.

Wraps ``shap.TreeExplainer`` and returns a human-readable list of the top-k
feature contributions for a single prediction.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import shap

from rubin_skymap.features.lightcurve import FEATURE_NAMES
from rubin_skymap.models.predict import Predictor

_log = logging.getLogger(__name__)


class Explainer:
    """Computes SHAP feature contributions for individual predictions.

    Parameters
    ----------
    predictor:
        A trained :class:`~rubin_skymap.models.predict.Predictor` instance.
    """

    def __init__(self, predictor: Predictor) -> None:
        _log.info("Building SHAP TreeExplainer …")
        self._predictor = predictor
        self._explainer: shap.TreeExplainer = shap.TreeExplainer(predictor.booster)

    def explain(
        self,
        features: dict[str, Any],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return top-k SHAP contributions for the predicted class.

        Parameters
        ----------
        features:
            Feature dict as returned by
            :func:`~rubin_skymap.features.lightcurve.extract_features`.
        top_k:
            Number of top features to return.

        Returns
        -------
        list[dict]
            List of dicts with keys ``feature``, ``value``, ``contribution``
            sorted by absolute contribution (descending).
        """
        import pandas as pd

        row = {k: features.get(k, float("nan")) for k in FEATURE_NAMES}
        X = pd.DataFrame([row], columns=FEATURE_NAMES).fillna(0.0)

        # shap_values: list of arrays shape (1, n_features) per class, OR
        # a single array of shape (1, n_features, n_classes) depending on SHAP version.
        shap_values: Any = self._explainer.shap_values(X)

        # Determine predicted class index.
        _, _ = self._predictor.predict(features)
        proba_dict = self._predictor.predict_proba(features)
        best_label = max(proba_dict, key=lambda k: proba_dict[k])
        class_idx = self._predictor.label_to_int.get(best_label, 0)

        # Extract the contribution array for the predicted class.
        if isinstance(shap_values, list):
            # List of arrays: one per class, each shape (1, n_features).
            cls_idx_safe = min(class_idx, len(shap_values) - 1)
            contribs: np.ndarray = np.asarray(shap_values[cls_idx_safe][0])
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3:
                # Shape (1, n_features, n_classes)
                cls_idx_safe = min(class_idx, shap_values.shape[2] - 1)
                contribs = shap_values[0, :, cls_idx_safe]
            elif shap_values.ndim == 2:
                # Shape (1, n_features) — binary or single class.
                contribs = shap_values[0]
            else:
                _log.warning("Unexpected SHAP array shape: %s", shap_values.shape)
                contribs = np.zeros(len(FEATURE_NAMES))
        else:
            _log.warning("Unexpected SHAP values type: %s", type(shap_values))
            contribs = np.zeros(len(FEATURE_NAMES))

        # Build result list.
        results: list[dict[str, Any]] = []
        for feat_name, feat_val, contrib in zip(
            FEATURE_NAMES, list(X.iloc[0]), contribs
        ):
            results.append(
                {
                    "feature": feat_name,
                    "value": float(feat_val) if math.isfinite(float(feat_val)) else 0.0,
                    "contribution": float(contrib) if math.isfinite(float(contrib)) else 0.0,
                }
            )

        # Sort by absolute contribution descending, take top_k.
        results.sort(key=lambda d: abs(d["contribution"]), reverse=True)
        return results[:top_k]
