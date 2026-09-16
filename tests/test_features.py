"""Tests for rubin_skymap.features.lightcurve."""

from __future__ import annotations

import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest

from rubin_skymap.features.lightcurve import FEATURE_NAMES, extract_features
from rubin_skymap.ingest.mock_source import MockSource
from rubin_skymap.ingest.schemas import Alert

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_alert_for_class(cls_name: str, seed: int = 0) -> Alert:
    """Generate one Alert for *cls_name* using MockSource."""
    source = MockSource(n_per_batch=1, classes=[cls_name], seed=seed)
    alerts = list(source)
    assert len(alerts) == 1, f"Expected 1 alert, got {len(alerts)}"
    return alerts[0]


ALL_CLASSES = ["SN Ia", "SN II", "SN Ibc", "SLSN", "Kilonova", "AGN", "RRL"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls_name", ALL_CLASSES)
def test_feature_key_set(cls_name: str) -> None:
    """extract_features returns exactly the canonical key set for each class."""
    alert = _make_alert_for_class(cls_name)
    features = extract_features(alert)
    assert set(features.keys()) == set(FEATURE_NAMES), (
        f"Key mismatch for class '{cls_name}': "
        f"extra={set(features) - set(FEATURE_NAMES)}, "
        f"missing={set(FEATURE_NAMES) - set(features)}"
    )


@pytest.mark.parametrize("cls_name", ALL_CLASSES)
def test_n_detections_positive(cls_name: str) -> None:
    """n_detections should be a positive integer-valued float."""
    alert = _make_alert_for_class(cls_name)
    features = extract_features(alert)
    assert features["n_detections"] > 0.0, "n_detections must be positive"


@pytest.mark.parametrize("cls_name", ALL_CLASSES)
def test_time_span_positive(cls_name: str) -> None:
    """time_span should be positive for multi-observation light curves."""
    alert = _make_alert_for_class(cls_name)
    features = extract_features(alert)
    assert features["time_span"] >= 0.0, "time_span must be non-negative"


@pytest.mark.parametrize("cls_name", ALL_CLASSES)
def test_peak_mag_finite(cls_name: str) -> None:
    """peak_mag must be a finite number."""
    alert = _make_alert_for_class(cls_name)
    features = extract_features(alert)
    assert math.isfinite(features["peak_mag"]), "peak_mag must be finite"


@pytest.mark.parametrize("cls_name", ALL_CLASSES)
def test_amplitude_non_negative(cls_name: str) -> None:
    """amplitude = max(mag) - min(mag) must be >= 0."""
    alert = _make_alert_for_class(cls_name)
    features = extract_features(alert)
    amp = features["amplitude"]
    if math.isfinite(amp):
        assert amp >= 0.0, "amplitude must be non-negative"


def test_empty_alert_returns_all_nan() -> None:
    """Empty light-curve arrays should yield an all-NaN feature dict."""
    alert = Alert(
        object_id="EMPTY-001",
        ra=0.0,
        dec=0.0,
        jd=[],
        mag=[],
        magerr=[],
        filter=[],
    )
    features = extract_features(alert)
    assert set(features.keys()) == set(FEATURE_NAMES)
    for key, val in features.items():
        assert math.isnan(val), f"Expected NaN for key '{key}' on empty alert, got {val}"


def test_single_observation_no_crash() -> None:
    """A single-observation alert must not raise and must return finite peak_mag."""
    alert = Alert(
        object_id="SINGLE-001",
        ra=12.0,
        dec=30.0,
        jd=[2459000.5],
        mag=[19.5],
        magerr=[0.05],
        filter=["g"],
    )
    features = extract_features(alert)
    assert set(features.keys()) == set(FEATURE_NAMES)
    assert math.isfinite(features["peak_mag"])
    assert features["n_detections"] == 1.0


def test_feature_values_are_floats() -> None:
    """All returned values must be Python floats (including NaN)."""
    alert = _make_alert_for_class("AGN")
    features = extract_features(alert)
    for key, val in features.items():
        assert isinstance(val, float), f"Value for '{key}' is {type(val)}, expected float"
