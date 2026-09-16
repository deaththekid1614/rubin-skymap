# rubin-skymap

**A real-time transient classification broker — live ZTF alerts classified by machine learning and streamed onto an interactive equatorial sky map.**

Rubin-skymap pulls live astronomical alerts from the ZTF (Zwicky Transient Facility) survey, automatically extracts light-curve features, classifies each event into one of eight transient categories using a LightGBM model, explains the prediction with SHAP, and pushes everything to a browser dashboard over WebSocket — all within a few seconds of the original detection.

---

## What it does, in plain language

Every few seconds, a real telescope (ZTF) detects something in the sky that changed brightness. It could be a supernova, a variable star, a black-hole flare, or something completely unknown. This project:

1. **Fetches those alerts** from the ALeRCE or Fink broker APIs (or generates synthetic ones offline).
2. **Extracts 15 features** from the light curve — how fast it rose, how bright it got, how symmetric the shape is, etc.
3. **Classifies it** with a trained LightGBM model into one of 8 categories (SN Ia, SN II, SN Ibc, SLSN, Kilonova, AGN, RRL, Other).
4. **Explains the prediction** using SHAP — showing which features pushed the model toward or away from that class.
5. **Stores everything** in a SQLite database and broadcasts it over WebSocket.
6. **Displays it on a live sky map** — a dark equatorial canvas with ~9,000 real Hipparcos stars, 88 IAU constellation lines, the Milky Way band, and coloured dots for each classified alert.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                            │
│                                                                  │
│  ZTF Alert Source          Consumer Process                      │
│  ┌──────────────┐         ┌─────────────────────────────────┐   │
│  │ ALeRCE API   │──poll──▶│ 1. Fetch alert batch            │   │
│  │ Fink API     │         │ 2. Extract 15 light-curve       │   │
│  │ Mock/Offline │         │    features (NumPy / pandas)    │   │
│  └──────────────┘         │ 3. LightGBM classify → 8 classes│   │
│                           │ 4. SHAP explain top-5 features  │   │
│                           │ 5. Save to SQLite DB            │   │
│                           │ 6. Push to asyncio broadcast bus│   │
│                           └──────────────┬──────────────────┘   │
└──────────────────────────────────────────│──────────────────────┘
                                           │
┌──────────────────────────────────────────│──────────────────────┐
│                        API SERVER        │                       │
│                                          ▼                       │
│  FastAPI (Uvicorn)    ┌──────────────────────────────────────┐   │
│                       │ REST endpoints:                       │   │
│                       │  GET  /api/alerts    — recent alerts  │   │
│                       │  GET  /api/stats     — counts, drift  │   │
│                       │  POST /api/predict   — single predict │   │
│                       │  WS   /ws/alerts     — live stream    │   │
│                       │  GET  /              — dashboard HTML  │   │
│                       │  GET  /static/*      — JS / CSS / data│   │
│                       └──────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                           │
┌──────────────────────────────────────────│──────────────────────┐
│                     BROWSER DASHBOARD    │                       │
│                                          ▼                       │
│  Vanilla JS + HTML5 Canvas                                       │
│                                                                  │
│  ┌──────────────────────────────┐  ┌──────────────────────────┐ │
│  │   Sky Map Canvas             │  │   Sidebar                │ │
│  │                              │  │                          │ │
│  │  Layers (back → front):      │  │  • Live Alerts feed      │ │
│  │  1. Dark sky background      │  │  • Search / filter       │ │
│  │  2. Milky Way band           │  │  • Inspector panel:      │ │
│  │  3. RA/Dec grid              │  │    – RA, Dec, confidence │ │
│  │  4. ~9,000 Hipparcos stars   │  │    – Class probabilities │ │
│  │  5. 88 constellation lines   │  │    – SHAP bar chart      │ │
│  │  6. Alert dots (by class)    │  │    – Class explanation   │ │
│  │  7. Selection highlight      │  │                          │ │
│  └──────────────────────────────┘  └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | What it uses |
|---|---|
| Alert ingestion | ALeRCE ZTF REST API · Fink broker REST API · MockSource (offline) |
| Feature engineering | NumPy · pandas — 15 photometric light-curve features |
| Classifier | LightGBM 4.3 · scikit-learn |
| Explainability | SHAP 0.45 — top-5 feature contributions per prediction |
| API server | FastAPI 0.111 · Uvicorn · WebSocket |
| Database | SQLite · SQLAlchemy 2.0 |
| Frontend | Pure vanilla HTML5 Canvas + CSS + JavaScript — no frameworks |
| Config | PyYAML · python-dotenv |

---

## Quickstart

### Prerequisites

- Python **3.11.9**
- Internet connection for live ZTF data (or use `--synthetic` to run fully offline)

### 1 · Install

```bash
git clone <repo-url> rubin-skymap
cd rubin-skymap

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2 · Generate sky map assets (one time)

This downloads ~580 KB of star and constellation data for the sky map background:

```bash
python scripts/build_sky_assets.py
```

### 3 · Launch everything

```bash
./run.sh
```

This single command:
- Checks if a trained model exists; trains one on synthetic data if not
- Starts the API server at `http://localhost:8000`
- Starts the alert consumer (pulls live ZTF data from ALeRCE every 8 s)
- Opens the dashboard in your browser

```bash
./run.sh --api-only    # API server only, skip the consumer
./run.sh --stop        # kill all running rubin-skymap processes
```

### 4 · Manual step-by-step (if you prefer)

```bash
# Generate ~2,000 synthetic training objects (no network needed)
python scripts/download_plasticc.py --synthetic

# Train the LightGBM classifier → saves to models/lgbm_v1.txt
python scripts/train_model.py

# Start the API server
uvicorn rubin_skymap.serving.api:app --host 0.0.0.0 --port 8000

# In a second terminal: start the alert consumer
python scripts/run_consumer.py
```

Open **http://localhost:8000** in your browser.

---

## Dashboard

The dashboard is a single dark-theme HTML page served by FastAPI. Here is what each part does:

| Part | What you see | What it means |
|---|---|---|
| **Header** | App name, alert count, rate, top class, UTC clock | Live summary of all classified alerts so far |
| **Legend bar** | Coloured class filters + sky layer toggles | Click a class to hide/show it on the map and list |
| **Sky map** | Dark canvas with stars, constellations, alert dots | Equatorial projection — RA on X axis, Dec on Y axis |
| **Alert dot** | Coloured circle on the map | One classified ZTF alert. Colour = predicted class |
| **Live Alerts panel** | Scrolling list of recent alerts | Streams in real time over WebSocket |
| **Inspector panel** | Shown when you click a dot or list row | Full details: coordinates, model confidence, probability bars, SHAP chart |
| **Status indicator** | Green = connected, Red = reconnecting | WebSocket health — auto-reconnects with backoff |

### Sky layer toggles

| Toggle | What it controls |
|---|---|
| **Milky Way** | Soft diffuse band showing the galactic plane |
| **Stars** | ~9,000 Hipparcos background stars, sized by visual magnitude |
| **Constellations** | 88 IAU constellation line segments |
| **Labels** | Constellation name text at each constellation's centroid |

---

## Transient Classes

The model classifies each alert into one of these eight categories:

| Class | Colour | What it is |
|---|---|---|
| **SN Ia** | Red | Type Ia supernova — a white dwarf explodes. The "standard candle" used to measure dark energy and the expansion of the universe. Light curve rises over ~20 days then fades over ~60 days. |
| **SN II** | Orange | Core-collapse supernova — a massive star's core implodes. Has a long ~80-day brightness plateau powered by hydrogen recombination. |
| **SN Ibc** | Yellow | Stripped core-collapse — same physics as SN II but the star lost its outer layers before exploding. Faster rise and decline. |
| **SLSN** | Green | Super-luminous supernova — 10 to 100 times brighter than a normal supernova. Extremely rare. Possibly powered by a rapidly spinning neutron star (magnetar). |
| **Kilonova** | Blue | Neutron star merger — two neutron stars collide. The heavy elements (gold, platinum) in your jewellery were made in events like this. Rises and fades in just 2–5 days. |
| **AGN** | Purple | Active Galactic Nucleus — a supermassive black hole actively swallowing material. Varies stochastically over years. Never fully fades. |
| **RRL** | Pink | RR Lyrae variable star — an old, low-mass star that physically pulsates with a very regular period of 0.2–1 day. Used to map the Milky Way's structure. |
| **Other** | Grey | Unclassified — model confidence below threshold. Could be a tidal disruption event, microlensing, eclipsing binary, or something genuinely new. |

---

## API Reference

The FastAPI server exposes these endpoints:

| Method | Endpoint | What it does |
|---|---|---|
| `GET` | `/health` | Returns `{"status": "ok"}` — use for uptime monitoring |
| `GET` | `/api/alerts?limit=200` | Returns the most recent N predictions from the database |
| `GET` | `/api/stats` | Returns total count, per-class breakdown, and drift scores |
| `POST` | `/api/predict` | Classifies a single alert you provide as JSON |
| `WS` | `/ws/alerts` | WebSocket — receive a JSON message every time a new alert is classified |

### Example: classify a single alert

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "object_id": "ZTF21example",
    "ra": 5.5,
    "dec": -10.2,
    "jd": [2459001.5, 2459003.5, 2459007.5, 2459012.5, 2459018.5],
    "mag": [19.1, 18.4, 18.0, 18.3, 18.9],
    "magerr": [0.05, 0.04, 0.03, 0.04, 0.05],
    "filter": ["g", "r", "g", "r", "g"]
  }'
```

---

## Data Sources

### ALeRCE (default — no login required)

The consumer polls the [ALeRCE ZTF REST API](https://api.alerce.online/ztf/v1) every 8 seconds. It cycles through supernova, AGN, and variable-star query types so the dashboard gets a diverse mix of real events.

```yaml
# configs/dev.yaml
ingest:
  mode: alerce
  poll_interval_sec: 8.0
  batch_size: 10
```

### Fink Broker (alternative)

```yaml
ingest:
  mode: fink
```

Add credentials to `.env` if using private Fink streams (copy `.env.example` first):

```env
FINK_USER=your_username
FINK_PASSWORD=your_password
```

### Mock / Fully offline

```yaml
ingest:
  mode: mock
```

Generates synthetic light curves locally. No network needed. Good for demos and development.

---

## Sky Map Assets

The sky map background uses three pre-generated JSON files served as static assets:

| File | Contents | Size |
|---|---|---|
| `frontend/data/stars.json` | ~9,000 brightest Hipparcos stars with RA, Dec, and magnitude | ~540 KB |
| `frontend/data/constellations.json` | 88 IAU constellation line segments (HIP star pairs) | ~28 KB |
| `frontend/data/milkyway.json` | 361-point galactic-plane centre-line in equatorial coords | ~6 KB |

Generate them once by running:

```bash
python scripts/build_sky_assets.py
```

The script fetches data from the HYG database and Stellarium. If the network is unavailable it falls back to an embedded minimal dataset (50 brightest stars, 5 major constellations) so the dashboard still loads.

### How the Milky Way is drawn

The galactic band is drawn on an offscreen canvas in three clean layers, then composited at 55% opacity onto the sky:

- **Layer 1** — A wide, heavily blurred (`blur(18px)`) cool-white stroke covering the full band width. This represents the diffuse glow of millions of unresolved stars.
- **Layer 2** — A narrower, moderately blurred (`blur(4px)`) brighter spine down the centre, representing the denser galactic plane.
- **Layer 3** — A very thin `destination-out` (erasing) stroke along the exact centre, creating a faint dark notch that hints at the real interstellar dust lane that blocks light in the actual Milky Way.

The result is subtle — background context, not the main event.

---

## Training on Real PLAsTiCC Data

For a production-quality model, train on the real [PLAsTiCC](https://plasticc.org/) dataset:

```bash
# Download ~500 MB from Zenodo and convert to parquet
python scripts/download_plasticc.py --real

# Retrain
python scripts/train_model.py --data data/processed/plasticc_train.parquet
```

---

## Project Layout

```
rubin-skymap/
│
├── configs/
│   ├── base.yaml              — all default settings
│   └── dev.yaml               — dev overrides (mode, poll interval, DB path)
│
├── rubin_skymap/              — main Python package
│   ├── config.py              — YAML loader with deep-merge and dotenv support
│   ├── logging_setup.py
│   ├── ingest/
│   │   ├── alerce_source.py   — ALeRCE ZTF REST API client
│   │   ├── fink_source.py     — Fink broker REST API client
│   │   ├── mock_source.py     — synthetic parametric light-curve generator
│   │   └── schemas.py         — Pydantic alert schema
│   ├── features/
│   │   └── lightcurve.py      — 15 photometric feature extractors
│   ├── models/
│   │   ├── train.py           — LightGBM training pipeline
│   │   ├── predict.py         — inference wrapper (loads model once, thread-safe)
│   │   └── explain.py         — SHAP TreeExplainer wrapper
│   ├── db/
│   │   ├── models.py          — SQLAlchemy ORM table definitions
│   │   └── session.py         — engine/session factory + CRUD helpers
│   ├── serving/
│   │   ├── api.py             — FastAPI app: REST routes + WebSocket + static files
│   │   └── bus.py             — asyncio in-process broadcast bus
│   └── monitoring/
│       └── drift.py           — Kolmogorov–Smirnov drift scoring vs training data
│
├── scripts/
│   ├── build_sky_assets.py    — generate stars / constellations / milkyway JSON
│   ├── download_plasticc.py   — synthetic generator + Zenodo downloader
│   ├── train_model.py         — training entry point
│   └── run_consumer.py        — consumer loop entry point
│
├── frontend/
│   ├── index.html             — single-page dashboard
│   ├── styles.css             — dark theme styles
│   ├── app.js                 — all JS: canvas rendering, WebSocket, inspector
│   └── data/
│       ├── stars.json         — Hipparcos star positions (generated)
│       ├── constellations.json — IAU constellation lines (generated)
│       └── milkyway.json      — galactic plane polyline (generated)
│
├── tests/
│   ├── test_api.py
│   ├── test_features.py
│   ├── test_ingest.py
│   ├── test_model.py
│   └── test_sky_assets.py     — schema validation for generated data files
│
├── data/
│   ├── raw/                   — raw parquet input files
│   └── processed/             — feature parquets used for drift monitoring
│
├── models/
│   └── lgbm_v1.txt            — trained LightGBM model (generated by train_model.py)
│
├── run.sh                     — one-command launcher
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── pyproject.toml
```

---

## Configuration Reference

All configuration lives in `configs/base.yaml` (defaults). Override any value in `configs/dev.yaml`, or point to a different file via the `RUBIN_SKYMAP_ENV` environment variable.

| Key | Default | What it controls |
|---|---|---|
| `ingest.mode` | `alerce` | Alert source: `alerce` · `fink` · `mock` |
| `ingest.poll_interval_sec` | `8.0` | Seconds between poll cycles |
| `ingest.batch_size` | `10` | Alerts fetched per cycle |
| `ingest.alerce.min_detections` | `5` | Minimum ZTF detections to include an object |
| `ingest.alerce.min_probability` | `0.0` | Minimum ALeRCE classifier probability gate |
| `model.num_leaves` | `31` | LightGBM tree complexity |
| `model.learning_rate` | `0.05` | Gradient boosting learning rate |
| `model.n_estimators` | `300` | Maximum number of boosting rounds |
| `model.random_state` | `42` | Global random seed for reproducibility |
| `server.port` | `8000` | API server port |
| `db.url` | `sqlite:///./rubin_skymap.db` | SQLAlchemy database URL |

---

## Docker

```bash
# Build and start the API server in a container (port 8000)
docker compose up --build
```

The container runs the **API server only**. Start the consumer on the host:

```bash
python scripts/run_consumer.py
```

> CORS is set to `["*"]` for development. Restrict `server.cors_origins` in `configs/base.yaml` before deploying publicly.

---

## Tests, Lint, and Type Check

```bash
# Run the test suite (fully offline)
pytest -q

# Lint with ruff
ruff check .

# Type check with mypy
mypy rubin_skymap
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Model not found at models/lgbm_v1.txt` | Run `python scripts/train_model.py` (or just `./run.sh`) |
| `Training data not found` | Run `python scripts/download_plasticc.py --synthetic` first |
| Port 8000 already in use | `lsof -ti:8000 \| xargs kill` or change `server.port` in config |
| Sky map shows no stars or constellations | Run `python scripts/build_sky_assets.py` to generate the data files |
| Sky map shows no alert dots | Make sure the consumer is running; check `/api/alerts` returns rows |
| Dashboard status dot is red | API server is not reachable — it will auto-reconnect every 3 s |
| `ModuleNotFoundError: rubin_skymap` | Run all commands from the repo root with the venv activated |
| All alerts are the same class | Old mock rows in DB — `DELETE FROM alerts WHERE object_id LIKE 'MOCK-%'` |

---

## Reproducibility

- All random seeds are controlled by `model.random_state` in config (default 42)
- All file paths are defined in `configs/base.yaml` — no hardcoded paths anywhere in library code
- `requirements.txt` is pinned to exact versions
- Synthetic data generation is fully deterministic
- Training features are saved to `data/processed/features_train.parquet` for later drift monitoring

---

## License

MIT — see [LICENSE](LICENSE.md).
