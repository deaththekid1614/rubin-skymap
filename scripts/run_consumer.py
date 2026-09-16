"""scripts/run_consumer.py — Continuous alert consumer and prediction pipeline.

Reads alerts from the configured source (mock or Fink), runs the full
inference pipeline, persists results to SQLite, and broadcasts to all
WebSocket clients via the in-process bus.

Usage
-----
    python scripts/run_consumer.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rubin_skymap.config import cfg_get, load_config
from rubin_skymap.db.session import init_db, insert_alert
from rubin_skymap.features.lightcurve import extract_features
from rubin_skymap.logging_setup import setup_logging
from rubin_skymap.models.explain import Explainer
from rubin_skymap.models.predict import Predictor
from rubin_skymap.serving.bus import bus

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful shutdown flag
# ---------------------------------------------------------------------------
_SHUTDOWN = False


def _handle_signal(signum: int, frame: Any) -> None:  # noqa: ANN001
    """Set the global shutdown flag on SIGINT/SIGTERM."""
    global _SHUTDOWN
    _log.info("Shutdown signal received (%s).", signum)
    _SHUTDOWN = True


# ---------------------------------------------------------------------------
# Main async loop
# ---------------------------------------------------------------------------


async def _run_loop(cfg: dict) -> None:
    """Core consumer loop.

    Parameters
    ----------
    cfg:
        Loaded configuration dictionary.
    """
    global _SHUTDOWN

    db_url: str = cfg_get(cfg, "db.url", "sqlite:///./rubin_skymap.db")
    poll_interval: float = float(cfg_get(cfg, "ingest.poll_interval_sec", 2.0))
    batch_size: int = int(cfg_get(cfg, "ingest.batch_size", 10))
    dedupe: bool = bool(cfg_get(cfg, "ingest.dedupe", True))
    mode: str = str(cfg_get(cfg, "ingest.mode", "mock"))

    model_path: str = cfg_get(cfg, "model.model_path", "models/lgbm_v1.txt")
    label_map_path: str = cfg_get(cfg, "model.label_map_path", "models/label_map.json")

    if not Path(model_path).exists():
        _log.error(
            "Model not found at '%s'. Run `python scripts/train_model.py` first.",
            model_path,
        )
        return

    _log.info("Loading predictor …")
    predictor = Predictor(model_path=model_path, label_map_path=label_map_path)
    explainer = Explainer(predictor)

    # Build the alert source.
    if mode == "alerce":
        from rubin_skymap.ingest.alerce_source import AlerceSource

        alerce_cfg = cfg_get(cfg, "ingest.alerce", {})
        source = AlerceSource(
            api_base=str(alerce_cfg.get("api_base", "https://api.alerce.online/ztf/v1")),
            timeout=float(alerce_cfg.get("timeout_sec", 20.0)),
            retries=int(alerce_cfg.get("retries", 3)),
            backoff_base=float(alerce_cfg.get("backoff_base_sec", 2.0)),
            batch_size=batch_size,
            min_detections=int(alerce_cfg.get("min_detections", 5)),
            min_probability=float(alerce_cfg.get("min_probability", 0.7)),
        )
        _log.info("Using AlerceSource (real ZTF data).")

    elif mode == "fink":
        from rubin_skymap.ingest.fink_source import FinkSource, test_connection

        fink_cfg = cfg_get(cfg, "ingest.fink", {})
        api_base = str(fink_cfg.get("api_base", "https://api.fink-portal.org"))
        reachable = test_connection(api_base, timeout=8.0)
        if reachable:
            _log.info("✓ Fink API reachable. Starting live stream.")
        else:
            _log.warning(
                "⚠  Fink API NOT reachable. Check internet/firewall. "
                "Set  ingest.mode: alerce  in configs/dev.yaml as a working alternative."
            )
        source = FinkSource(
            api_base=api_base,
            timeout=float(fink_cfg.get("timeout_sec", 15.0)),
            retries=int(fink_cfg.get("retries", 3)),
            backoff_base=float(fink_cfg.get("backoff_base_sec", 1.5)),
            classes=list(fink_cfg.get(
                "classes", ["SN Ia", "SN II", "SN Ibc", "SLSN", "Kilonova", "AGN", "RRL"]
            )),
            batch_size=batch_size,
        )
        _log.info("Using FinkSource (mode=fink).")

    else:
        from rubin_skymap.ingest.mock_source import MockSource

        source = MockSource(
            n_per_batch=batch_size,
            classes=["SN Ia", "SN II", "SN Ibc", "SLSN", "Kilonova", "AGN", "RRL"],
            seed=int(time.time()) % (2**31),
        )
        _log.info("Using MockSource (mode=mock).")

    seen_ids: set[str] = set()
    processed_total = 0

    _log.info("Consumer started. Poll interval: %.1f s. Ctrl+C to stop.", poll_interval)

    while not _SHUTDOWN:
        try:
            for alert in source:
                if _SHUTDOWN:
                    break

                if dedupe and alert.object_id in seen_ids:
                    _log.debug("Skipping duplicate: %s", alert.object_id)
                    continue

                # Feature extraction.
                features: dict = extract_features(alert)

                # Prediction.
                label, prob = predictor.predict(features)

                # SHAP explanation.
                shap_result: list = explainer.explain(features, top_k=5)

                # Build persistence payload.
                now_utc = datetime.now(timezone.utc)
                payload: dict = {
                    "object_id": alert.object_id,
                    "ra": alert.ra,
                    "dec": alert.dec,
                    "predicted_class": label,
                    "predicted_prob": prob,
                    "features_json": json.dumps(features),
                    "shap_json": json.dumps(shap_result),
                    "created_at": now_utc,
                }

                # Persist.
                insert_alert(db_url, payload)
                seen_ids.add(alert.object_id)
                processed_total += 1

                # Broadcast.
                broadcast_msg: dict = {
                    "object_id": alert.object_id,
                    "ra": alert.ra,
                    "dec": alert.dec,
                    "predicted_class": label,
                    "predicted_prob": round(prob, 4),
                    "shap": shap_result,
                    "features": features,
                    "created_at": now_utc.isoformat(),
                }
                await bus.publish(broadcast_msg)

                _log.info(
                    "[%d] %s → %s (%.2f%%)",
                    processed_total,
                    alert.object_id,
                    label,
                    prob * 100,
                )

        except Exception as exc:
            _log.warning("Consumer loop error: %s", exc, exc_info=True)

        # Sleep between polls using small increments to respond to shutdown quickly.
        elapsed = 0.0
        step = 0.25
        while elapsed < poll_interval and not _SHUTDOWN:
            await asyncio.sleep(step)
            elapsed += step

    _log.info("Consumer stopped. Total processed: %d", processed_total)


async def main() -> None:
    """Async entry point."""
    cfg = load_config()
    setup_logging(cfg_get(cfg, "app.log_level", "INFO"))

    db_url = cfg_get(cfg, "db.url", "sqlite:///./rubin_skymap.db")
    init_db(db_url)

    # Register signal handlers.
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    await _run_loop(cfg)


if __name__ == "__main__":
    asyncio.run(main())
