"""Tests for the ingest layer (mock source and fink source)."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest

from rubin_skymap.ingest.mock_source import MockSource
from rubin_skymap.ingest.schemas import Alert

# ---------------------------------------------------------------------------
# MockSource tests
# ---------------------------------------------------------------------------


def test_mock_source_yields_valid_alerts() -> None:
    """MockSource should yield Alert objects with consistent array lengths."""
    source = MockSource(n_per_batch=5, seed=42)
    alerts = list(source)
    assert len(alerts) == 5, f"Expected 5 alerts, got {len(alerts)}"

    for alert in alerts:
        assert isinstance(alert, Alert)
        n = len(alert.jd)
        assert n > 0
        assert len(alert.mag) == n
        assert len(alert.magerr) == n
        assert len(alert.filter) == n


def test_mock_source_unique_object_ids() -> None:
    """MockSource should generate distinct object IDs."""
    source = MockSource(n_per_batch=20, seed=7)
    alerts = list(source)
    ids = [a.object_id for a in alerts]
    assert len(set(ids)) == len(ids), "MockSource produced duplicate object IDs"


def test_mock_source_ra_dec_ranges() -> None:
    """RA must be in [0, 24) and Dec in [-90, 90]."""
    source = MockSource(n_per_batch=30, seed=1)
    for alert in source:
        assert 0.0 <= alert.ra < 24.0, f"RA out of range: {alert.ra}"
        assert -90.0 <= alert.dec <= 90.0, f"Dec out of range: {alert.dec}"


def test_mock_source_filter_values() -> None:
    """All filter values must be in the allowed LSST band set."""
    allowed = {"u", "g", "r", "i", "z", "y"}
    source = MockSource(n_per_batch=10, seed=3)
    for alert in source:
        for band in alert.filter:
            assert band in allowed, f"Unknown band: {band}"


def test_mock_source_ingested_at_recent() -> None:
    """ingested_at should be a recent Unix timestamp."""
    before = time.time()
    source = MockSource(n_per_batch=3, seed=0)
    alerts = list(source)
    after = time.time()
    for alert in alerts:
        assert before - 1 <= alert.ingested_at <= after + 1


def test_mock_source_covers_all_classes() -> None:
    """When cycling through all 7 classes, each class should appear at least once."""
    classes = ["SN Ia", "SN II", "SN Ibc", "SLSN", "Kilonova", "AGN", "RRL"]
    source = MockSource(n_per_batch=len(classes) * 3, classes=classes, seed=5)
    alerts = list(source)
    # Class appears in the object_id prefix: "MOCK-SNIa-...", etc.
    seen_classes = set()
    for alert in alerts:
        for cls in classes:
            if cls.replace(" ", "") in alert.object_id:
                seen_classes.add(cls)
    assert seen_classes == set(classes), f"Missing classes: {set(classes) - seen_classes}"


# ---------------------------------------------------------------------------
# FinkSource tests (no network — requests.post is monkeypatched)
# ---------------------------------------------------------------------------


def _make_latests_response(object_ids: list[str]) -> list[dict]:
    """Return a canned 'latests' API response."""
    return [{"i:objectId": oid} for oid in object_ids]


def _make_objects_response(oid: str, n_rows: int = 10) -> list[dict]:
    """Return a canned 'objects' API response with synthetic LC rows."""
    rows = []
    for i in range(n_rows):
        rows.append({
            "i:objectId": oid,
            "i:jd": 2459000.5 + i,
            "i:magpsf": 19.0 + 0.1 * i,
            "i:sigmapsf": 0.05,
            "i:fid": (i % 3) + 1,
            "d:ra": 180.0,
            "d:dec": 30.0,
        })
    return rows


def test_fink_source_yields_alerts_with_mocked_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """FinkSource should yield at least one Alert when the API is mocked."""
    from rubin_skymap.ingest import fink_source as fs_module

    oids = ["ZTF21aaa0001", "ZTF21aaa0002"]

    call_count = [0]

    def mock_post(url: str, **kwargs) -> MagicMock:
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        call_count[0] += 1
        if "latests" in url:
            resp.json.return_value = _make_latests_response(oids)
        elif "objects" in url:
            body = kwargs.get("json", {})
            oid = body.get("objectId", oids[0])
            resp.json.return_value = _make_objects_response(oid)
        else:
            resp.json.return_value = []
        return resp

    monkeypatch.setattr(fs_module.requests, "post", mock_post)

    from rubin_skymap.ingest.fink_source import FinkSource

    source = FinkSource(
        api_base="https://mock.fink.test",
        timeout=5.0,
        retries=1,
        backoff_base=0.01,
        classes=["SN Ia"],
    )

    alerts = list(source)  # One __iter__ call yields all objects for one class.
    assert len(alerts) >= 1
    for alert in alerts:
        assert isinstance(alert, Alert)
        assert len(alert.jd) > 0


def test_fink_source_handles_http_error_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    """FinkSource must yield nothing (not raise) when the API returns an error."""
    import requests as req_lib

    from rubin_skymap.ingest import fink_source as fs_module

    def mock_post_fail(url: str, **kwargs):  # noqa: ANN001
        raise req_lib.ConnectionError("Simulated network failure")

    monkeypatch.setattr(fs_module.requests, "post", mock_post_fail)

    from rubin_skymap.ingest.fink_source import FinkSource

    source = FinkSource(
        api_base="https://mock.fink.test",
        timeout=1.0,
        retries=1,
        backoff_base=0.01,
        classes=["AGN"],
    )

    # Should not raise; should yield nothing.
    alerts = list(source)
    assert alerts == [], f"Expected empty list, got {len(alerts)} alerts"
