#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# rubin-skymap — one-command launcher
#
# Usage:
#   ./run.sh              # start API + consumer (opens browser automatically)
#   ./run.sh --api-only   # start API server only (dashboard at localhost:8000)
#   ./run.sh --stop       # kill any running rubin-skymap processes
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PYTHON=/home/death-kid/.pyenv/versions/3.11.9/bin/python3.11
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PORT=8000

cd "$REPO_ROOT"

# ── helpers ──────────────────────────────────────────────────────────────────

green() { printf "\033[0;32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[0;33m%s\033[0m\n" "$*"; }
red()  { printf "\033[0;31m%s\033[0m\n" "$*"; }
bold() { printf "\033[1m%s\033[0m\n" "$*"; }

banner() {
  echo ""
  bold "  ╔══════════════════════════════════════╗"
  bold "  ║        rubin-skymap  v0.1.0           ║"
  bold "  ║   Real-time Transient Classifier      ║"
  bold "  ╚══════════════════════════════════════╝"
  echo ""
}

# ── --stop ───────────────────────────────────────────────────────────────────

if [[ "${1:-}" == "--stop" ]]; then
  yellow "Stopping rubin-skymap processes…"
  pkill -f "uvicorn rubin_skymap.serving.api" 2>/dev/null && green "  API server stopped." || echo "  (API not running)"
  pkill -f "scripts/run_consumer.py"          2>/dev/null && green "  Consumer stopped."   || echo "  (Consumer not running)"
  exit 0
fi

# ── pre-flight checks ────────────────────────────────────────────────────────

banner

echo "Checking prerequisites…"

if ! "$PYTHON" -c "import rubin_skymap" 2>/dev/null; then
  red "ERROR: rubin_skymap package not importable with $PYTHON"
  red "Make sure you are in the repo root and packages are installed:"
  red "  pip install -r requirements.txt"
  exit 1
fi

if [[ ! -f "models/lgbm_v1.txt" || ! -f "models/label_map.json" ]]; then
  yellow "Model not found. Running training pipeline first…"
  echo ""

  if [[ ! -f "data/raw/plasticc_synthetic.parquet" ]]; then
    yellow "  Step 1/2 — Generating synthetic training data…"
    "$PYTHON" scripts/download_plasticc.py --synthetic
    echo ""
  else
    green "  Synthetic data already exists. Skipping download."
  fi

  yellow "  Step 2/2 — Training LightGBM classifier…"
  "$PYTHON" scripts/train_model.py
  echo ""
  green "  Model trained and saved to models/lgbm_v1.txt"
fi

green "✓ Model ready"
echo ""

# ── start API server ─────────────────────────────────────────────────────────

bold "Starting API server on http://localhost:$PORT …"

"$PYTHON" -m uvicorn rubin_skymap.serving.api:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --log-level warning &

API_PID=$!
echo "  API PID: $API_PID"

# Wait for the server to be ready
echo -n "  Waiting for server"
for i in $(seq 1 20); do
  sleep 0.5
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo ""
    green "✓ API server ready"
    break
  fi
  echo -n "."
  if [[ $i -eq 20 ]]; then
    echo ""
    red "ERROR: API server did not start within 10 s."
    kill "$API_PID" 2>/dev/null || true
    exit 1
  fi
done

echo ""

# ── start consumer (unless --api-only) ───────────────────────────────────────

if [[ "${1:-}" != "--api-only" ]]; then
  bold "Starting alert consumer…"
  "$PYTHON" scripts/run_consumer.py &
  CONSUMER_PID=$!
  echo "  Consumer PID: $CONSUMER_PID"
  green "✓ Consumer running"
  echo ""
fi

# ── open browser ──────────────────────────────────────────────────────────────

bold "Dashboard → http://localhost:$PORT"
echo ""
green "Press Ctrl+C to stop everything."
echo ""

# Try to open the browser (best-effort)
xdg-open "http://localhost:$PORT" 2>/dev/null \
  || open "http://localhost:$PORT" 2>/dev/null \
  || true

# ── wait + cleanup ────────────────────────────────────────────────────────────

cleanup() {
  echo ""
  yellow "Shutting down…"
  kill "$API_PID"      2>/dev/null || true
  kill "${CONSUMER_PID:-}" 2>/dev/null || true
  green "Done."
  exit 0
}

trap cleanup SIGINT SIGTERM

# Block until Ctrl+C
wait
