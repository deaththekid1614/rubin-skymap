"""FastAPI application for rubin-skymap.

Exposes:
    GET  /health
    GET  /api/alerts
    GET  /api/stats
    POST /api/predict
    WS   /ws/alerts
    GET  /            (frontend)
    /static/*         (frontend assets)
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from rubin_skymap.config import cfg_get, load_config
from rubin_skymap.db.session import class_counts, init_db, insert_alert, latest_alerts
from rubin_skymap.features.lightcurve import extract_features
from rubin_skymap.ingest.schemas import Alert
from rubin_skymap.logging_setup import setup_logging
from rubin_skymap.models.explain import Explainer
from rubin_skymap.models.predict import Predictor
from rubin_skymap.serving.bus import bus

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons (populated during lifespan).
# ---------------------------------------------------------------------------
_cfg: dict[str, Any] = {}
_predictor: Predictor | None = None
_explainer: Explainer | None = None
_db_url: str = "sqlite:///./rubin_skymap.db"

# Resolve frontend directory relative to repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FRONTEND_DIR = _REPO_ROOT / "frontend"


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Initialise shared resources on startup and clean up on shutdown."""
    global _cfg, _predictor, _explainer, _db_url

    _cfg = load_config()
    setup_logging(cfg_get(_cfg, "app.log_level", "INFO"))

    _db_url = cfg_get(_cfg, "db.url", "sqlite:///./rubin_skymap.db")
    init_db(_db_url)

    model_path = cfg_get(_cfg, "model.model_path", "models/lgbm_v1.txt")
    label_map_path = cfg_get(_cfg, "model.label_map_path", "models/label_map.json")

    if Path(model_path).exists() and Path(label_map_path).exists():
        _predictor = Predictor(model_path=model_path, label_map_path=label_map_path)
        _explainer = Explainer(_predictor)
        _log.info("Model loaded and ready.")
    else:
        _log.warning(
            "Model files not found (%s, %s). /api/predict will return 503 "
            "until the model is trained.",
            model_path,
            label_map_path,
        )

    yield  # Application is running.

    _log.info("Shutting down rubin-skymap API.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def _create_app() -> FastAPI:
    """Construct and configure the FastAPI application.

    Returns
    -------
    FastAPI
    """
    cfg = load_config()
    cors_origins: list[str] = cfg_get(cfg, "server.cors_origins", ["*"])

    application = FastAPI(
        title="rubin-skymap",
        version="0.1.0",
        description="Real-time astronomical transient classification broker.",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return application


app: FastAPI = _create_app()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_class=JSONResponse, tags=["meta"])
async def health() -> dict[str, str]:
    """Health check endpoint.

    Returns
    -------
    dict
        ``{"status": "ok"}``
    """
    return {"status": "ok"}


@app.get("/api/alerts", response_class=JSONResponse, tags=["data"])
async def get_alerts(limit: int = Query(200, ge=1, le=1000)) -> list[dict[str, Any]]:
    """Return the most recent predictions from the database.

    Parameters
    ----------
    limit:
        Maximum number of records to return (default 200, max 1000).

    Returns
    -------
    list[dict]
        Alert records ordered newest-first.
    """
    return latest_alerts(_db_url, limit=limit)


@app.get("/api/stats", response_class=JSONResponse, tags=["data"])
async def get_stats() -> dict[str, Any]:
    """Return aggregate statistics about processed alerts.

    Returns
    -------
    dict
        Keys: ``total``, ``class_counts``, ``model``, ``drift``.
    """
    counts = class_counts(_db_url)
    total = sum(counts.values())

    drift_info: dict[str, Any] = {}
    try:
        from rubin_skymap.monitoring.drift import compute_drift_for_stats

        drift_info = compute_drift_for_stats(_db_url)
    except Exception as exc:
        _log.debug("Drift computation skipped: %s", exc)
        drift_info = {"note": "no reference data"}

    return {
        "total": total,
        "class_counts": counts,
        "model": "lgbm_v1",
        "drift": drift_info,
    }


@app.post("/api/predict", response_class=JSONResponse, tags=["inference"])
async def predict_alert(alert: Alert) -> dict[str, Any]:
    """Run the full inference pipeline on a single alert.

    Parameters
    ----------
    alert:
        A valid ``Alert`` object (JSON body).

    Returns
    -------
    dict
        Keys: ``predicted_class``, ``probabilities``, ``shap``.

    Raises
    ------
    HTTPException(503):
        If the model has not been trained yet.
    """
    if _predictor is None or _explainer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python scripts/train_model.py` first.",
        )

    features = extract_features(alert)
    label, prob = _predictor.predict(features)
    probabilities = _predictor.predict_proba(features)
    shap_explanation = _explainer.explain(features, top_k=5)

    # Persist to DB (fire-and-forget style; errors are logged, not raised).
    try:
        insert_alert(
            _db_url,
            {
                "object_id": alert.object_id,
                "ra": alert.ra,
                "dec": alert.dec,
                "predicted_class": label,
                "predicted_prob": prob,
                "features_json": json.dumps(features),
                "shap_json": json.dumps(shap_explanation),
            },
        )
    except Exception as exc:
        _log.warning("Failed to persist prediction for %s: %s", alert.object_id, exc)

    return {
        "predicted_class": label,
        "probabilities": probabilities,
        "shap": shap_explanation,
    }


@app.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket) -> None:
    """WebSocket endpoint that streams new alert predictions in real time.

    Clients receive each new prediction as a JSON string immediately after it
    is published by the consumer runner.
    """
    await websocket.accept()
    _log.info("WebSocket client connected: %s", websocket.client)
    try:
        async with bus.subscribe() as queue:
            while True:
                msg = await queue.get()
                await websocket.send_json(msg)
    except WebSocketDisconnect:
        _log.info("WebSocket client disconnected: %s", websocket.client)
    except Exception as exc:
        _log.warning("WebSocket error: %s", exc)


# ---------------------------------------------------------------------------
# Static frontend — mounted AFTER all API routes so /api and /ws win.
# ---------------------------------------------------------------------------

if _FRONTEND_DIR.exists():
    # Serve index.html at root.
    @app.get("/", include_in_schema=False)
    async def serve_index() -> FileResponse:
        """Serve the frontend index page."""
        return FileResponse(str(_FRONTEND_DIR / "index.html"))

    app.mount(
        "/static",
        StaticFiles(directory=str(_FRONTEND_DIR)),
        name="static",
    )
else:
    _log.warning("Frontend directory not found at %s", _FRONTEND_DIR)

    @app.get("/", include_in_schema=False)
    async def serve_index_missing() -> JSONResponse:
        """Placeholder when frontend directory is absent."""
        return JSONResponse({"message": "Frontend not built. See frontend/ directory."})
