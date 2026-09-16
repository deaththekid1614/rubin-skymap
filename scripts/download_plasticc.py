"""scripts/download_plasticc.py — Download or generate PLAsTiCC training data.

Usage
-----
    python scripts/download_plasticc.py --synthetic   # fast, no network (default)
    python scripts/download_plasticc.py --real        # downloads from Zenodo
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path so `rubin_skymap` is importable when the
# script is run from the repo root.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from rubin_skymap.config import cfg_get, load_config
from rubin_skymap.ingest.mock_source import MockSource
from rubin_skymap.logging_setup import setup_logging

_log = logging.getLogger(__name__)

# PLAsTiCC label codes → class names.
_PLASTICC_LABELS: dict[int, str] = {
    90: "SN Ia",
    42: "SN II",
    62: "SN Ibc",
    95: "SLSN",
    64: "Kilonova",
    88: "AGN",
    92: "RRL",
}

_ZENODO_FILES = [
    "plasticc_train_lightcurves.csv.gz",
    "plasticc_train_metadata.csv.gz",
]

# All LSST passband IDs used in PLAsTiCC (0=u,1=g,2=r,3=i,4=z,5=y).
_PASSBAND_TO_FILTER: dict[int, str] = {0: "u", 1: "g", 2: "r", 3: "i", 4: "z", 5: "y"}


# ---------------------------------------------------------------------------
# Synthetic mode
# ---------------------------------------------------------------------------


def _build_synthetic(cfg: dict) -> None:
    """Generate synthetic labelled light curves and save to parquet.

    Parameters
    ----------
    cfg:
        Loaded configuration dictionary.
    """
    n_total: int = int(cfg_get(cfg, "data.synthetic_n", 2000))
    out_path: str = cfg_get(cfg, "data.parquet_synthetic", "data/raw/plasticc_synthetic.parquet")
    classes = list(_PLASTICC_LABELS.values())

    print(f"Generating {n_total} synthetic objects …")
    source = MockSource(n_per_batch=1, classes=classes, seed=42)

    records: list[dict] = []
    classes * (n_total // len(classes) + 1)

    # We drive MockSource iteration manually to collect n_total objects.
    iter(source)

    # Patch source to emit one specific class at a time for balance.
    import uuid

    from rubin_skymap.ingest.mock_source import _TEMPLATES, _generate_lc

    rng = np.random.default_rng(42)
    per_class = n_total // len(classes)
    remainder = n_total % len(classes)

    for cls_idx, cls_name in enumerate(classes):
        n_this = per_class + (1 if cls_idx < remainder else 0)
        tmpl = _TEMPLATES.get(cls_name, _TEMPLATES["SN Ia"])
        for _ in range(n_this):
            jd, mag, magerr, flt = _generate_lc(cls_name, tmpl, rng)
            oid = f"SYN-{cls_name.replace(' ', '')}-{uuid.uuid4().hex[:8]}"
            ra = float(rng.uniform(0.0, 24.0))
            dec = float(np.clip(rng.normal(0.0, 40.0), -89.9, 89.9))
            for j, m, me, f in zip(jd, mag, magerr, flt):
                records.append({
                    "object_id": oid,
                    "class": cls_name,
                    "ra": ra,
                    "dec": dec,
                    "jd": j,
                    "mag": m,
                    "magerr": me,
                    "filter": f,
                })

    cols = ["object_id", "class", "ra", "dec", "jd", "mag", "magerr", "filter"]
    df = pd.DataFrame(records, columns=cols)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, engine="pyarrow")

    # Print summary.
    print(f"\nSaved {len(df)} observation rows → {out_path}")
    print("\nClass distribution:")
    obj_counts = df.groupby("class")["object_id"].nunique()
    for cls_name, cnt in obj_counts.items():
        print(f"  {cls_name:<12} {cnt:>5} objects")
    print(f"\nTotal unique objects: {df['object_id'].nunique()}")


# ---------------------------------------------------------------------------
# Real mode (Zenodo download)
# ---------------------------------------------------------------------------


def _download_file(url: str, dest: Path) -> None:
    """Stream-download *url* to *dest* with a tqdm progress bar.

    Parameters
    ----------
    url:
        HTTP(S) URL to download.
    dest:
        Destination file path (parent must exist).
    """
    print(f"Downloading {dest.name} …")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=dest.name
    ) as bar:
        for chunk in resp.iter_content(chunk_size=65536):
            fh.write(chunk)
            bar.update(len(chunk))
    size = dest.stat().st_size
    assert size > 0, f"Downloaded file {dest} is empty!"
    print(f"  → {dest} ({size / 1e6:.1f} MB)")


def _build_real(cfg: dict) -> None:
    """Download PLAsTiCC training set from Zenodo and convert to parquet.

    Parameters
    ----------
    cfg:
        Loaded configuration dictionary.
    """
    zenodo_base: str = cfg_get(cfg, "data.zenodo_base", "https://zenodo.org/records/2539456/files")
    raw_dir = Path(cfg_get(cfg, "data.raw_dir", "data/raw")) / "zenodo"
    processed_dir = Path(cfg_get(cfg, "data.processed_dir", "data/processed"))
    out_path = Path(cfg_get(cfg, "data.parquet_train", "data/processed/plasticc_train.parquet"))

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Download files.
    for fname in _ZENODO_FILES:
        dest = raw_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  Already downloaded: {fname}")
        else:
            url = f"{zenodo_base}/{fname}?download=1"
            _download_file(url, dest)

    # Load and merge.
    print("\nLoading CSVs …")
    lc_path = raw_dir / "plasticc_train_lightcurves.csv.gz"
    meta_path = raw_dir / "plasticc_train_metadata.csv.gz"

    lc_df = pd.read_csv(lc_path, compression="gzip")
    meta_df = pd.read_csv(meta_path, compression="gzip")

    # Rename columns to our canonical schema.
    lc_df = lc_df.rename(columns={
        "object_id": "object_id",
        "mjd": "jd",
        "flux": "mag",
        "flux_err": "magerr",
        "passband": "passband_id",
    })

    # Map passband int → filter string.
    lc_df["filter"] = lc_df["passband_id"].map(_PASSBAND_TO_FILTER).fillna("g")

    # Map target → class name.
    meta_df["class"] = meta_df["target"].map(_PLASTICC_LABELS).fillna("Other")

    # Merge.
    merged = lc_df.merge(meta_df[["object_id", "class", "ra", "decl"]], on="object_id", how="left")
    merged = merged.rename(columns={"decl": "dec"})

    # Subset columns.
    out_df = merged[["object_id", "class", "ra", "dec", "jd", "mag", "magerr", "filter"]].copy()
    out_df = out_df.dropna(subset=["object_id", "jd", "mag"])

    # Convert ra from degrees to hours.
    out_df["ra"] = out_df["ra"] / 15.0

    out_df.to_parquet(str(out_path), index=False, engine="pyarrow")

    print(f"\nSaved {len(out_df)} observation rows → {out_path}")
    print("\nClass distribution:")
    obj_counts = out_df.groupby("class")["object_id"].nunique()
    for cls_name, cnt in obj_counts.items():
        print(f"  {cls_name:<12} {cnt:>5} objects")
    print(f"\nTotal unique objects: {out_df['object_id'].nunique()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download or generate PLAsTiCC training data."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--synthetic",
        action="store_true",
        default=True,
        help="Generate synthetic data (default, no network required).",
    )
    group.add_argument(
        "--real",
        action="store_true",
        default=False,
        help="Download real PLAsTiCC data from Zenodo.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for download_plasticc.py."""
    args = _parse_args()
    cfg = load_config()
    setup_logging(cfg_get(cfg, "app.log_level", "INFO"))

    if args.real:
        print("=== PLAsTiCC Real Download Mode ===")
        _build_real(cfg)
    else:
        print("=== PLAsTiCC Synthetic Mode ===")
        _build_synthetic(cfg)


if __name__ == "__main__":
    main()
