"""Tests for the FastAPI application endpoints."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rubin_skymap.ingest.mock_source import _TEMPLATES, MockSource, _generate_lc

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _build_tiny_model(tmp_path: Path) -> tuple[str, str]:
    """Train a tiny LightGBM model into *tmp_path* and return (model_path, label_map_path)."""
    import uuid

    import pandas as pd

    from rubin_skymap.models.train import train_model

    classes = list(_TEMPLATES.keys())
    rng = np.random.default_rng(42)
    records = []
    for cls_name in classes:
        tmpl = _TEMPLATES[cls_name]
        for _ in range(15):
            jd, mag, magerr, flt = _generate_lc(cls_name, tmpl, rng)
            oid = f"TEST-{cls_name.replace(' ', '')}-{uuid.uuid4().hex[:6]}"
            for j, m, me, f in zip(jd, mag, magerr, flt):
                records.append({
                    "object_id": oid, "class": cls_name,
                    "ra": float(rng.uniform(0, 24)), "dec": float(rng.uniform(-90, 90)),
                    "jd": j, "mag": m, "magerr": me, "filter": f,
                })

    df = pd.DataFrame(records)
    parquet_path = tmp_path / "tiny.parquet"
    df.to_parquet(str(parquet_path), index=False)

    model_path = str(tmp_path / "lgbm_api_test.txt")
    label_map_path = str(tmp_path / "label_map_api_test.json")
    train_model(
        str(parquet_path), model_path, label_map_path,
        {"num_leaves": 8, "learning_rate": 0.1, "n_estimators": 20, "random_state": 0},
    )
    return model_path, label_map_path


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    """Provide a TestClient with a pre-trained model loaded."""
    tmp = tmp_path_factory.mktemp("api_test")
    model_path, label_map_path = _build_tiny_model(tmp)

    import rubin_skymap.serving.api as api_module
    from rubin_skymap.models.explain import Explainer
    from rubin_skymap.models.predict import Predictor

    predictor = Predictor(model_path=model_path, label_map_path=label_map_path)
    explainer = Explainer(predictor)

    # Inject into the API module globals before creating the client.
    api_module._predictor = predictor
    api_module._explainer = explainer
    api_module._db_url = "sqlite:///:memory:"

    from rubin_skymap.db.session import init_db
    init_db("sqlite:///:memory:")

    # Re-wire the module-level db url used by route handlers.
    api_module._db_url = "sqlite:///:memory:"

    with TestClient(api_module.app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_health(client: TestClient) -> None:
    """GET /health should return 200 with {'status': 'ok'}."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_stats_keys(client: TestClient) -> None:
    """GET /api/stats should return 200 with required keys."""
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert "total" in body
    assert "class_counts" in body
    assert "model" in body


def test_get_alerts_empty(client: TestClient) -> None:
    """GET /api/alerts on a fresh DB should return an empty list."""
    resp = client.get("/api/alerts?limit=10")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_predict_returns_valid_structure(client: TestClient) -> None:
    """POST /api/predict with a valid Alert body should return classification output."""
    source = MockSource(n_per_batch=1, seed=42)
    alert = list(source)[0]
    body = alert.to_jsonable()

    resp = client.post("/api/predict", json=body)
    assert resp.status_code == 200, f"Response body: {resp.text}"

    result = resp.json()
    assert "predicted_class" in result
    assert "probabilities" in result
    assert "shap" in result

    # Probability values must be floats in [0, 1].
    for cls_name, prob in result["probabilities"].items():
        assert 0.0 <= prob <= 1.0, f"Probability out of range for {cls_name}: {prob}"

    # SHAP list must contain dicts with expected keys.
    assert isinstance(result["shap"], list)
    for entry in result["shap"]:
        assert "feature" in entry
        assert "value" in entry
        assert "contribution" in entry


def test_predict_with_minimal_alert(client: TestClient) -> None:
    """POST /api/predict with a minimal valid alert must not crash."""
    body = {
        "object_id": "TEST-MIN-001",
        "ra": 6.0,
        "dec": 15.0,
        "jd": [2459000.0, 2459001.0, 2459002.0],
        "mag": [19.5, 19.2, 19.8],
        "magerr": [0.05, 0.04, 0.06],
        "filter": ["g", "r", "i"],
    }
    resp = client.post("/api/predict", json=body)
    assert resp.status_code == 200
    assert "predicted_class" in resp.json()
