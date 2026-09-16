"""Synthetic alert generator that yields realistic-looking light curves.

Each call to ``MockSource.__iter__`` yields ``Alert`` objects indefinitely.
Light-curve shapes are parametric and seeded for reproducibility.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from typing import Iterator

import numpy as np

from rubin_skymap.ingest.schemas import Alert

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-class parametric templates
# ---------------------------------------------------------------------------
# Each template is a dict with keys used by _generate_lc.
_TEMPLATES: dict[str, dict] = {
    "SN Ia": {
        "peak_mag": 19.0,
        "amplitude": 3.0,
        "rise_days": 15.0,
        "decay_days": 40.0,
        "shape": "sn",
        "duration": 60.0,
        "n_obs": (25, 40),
    },
    "SN II": {
        "peak_mag": 19.5,
        "amplitude": 2.5,
        "rise_days": 20.0,
        "decay_days": 60.0,
        "shape": "plateau",
        "duration": 90.0,
        "n_obs": (20, 35),
    },
    "SN Ibc": {
        "peak_mag": 19.2,
        "amplitude": 2.8,
        "rise_days": 10.0,
        "decay_days": 30.0,
        "shape": "sn",
        "duration": 50.0,
        "n_obs": (20, 35),
    },
    "SLSN": {
        "peak_mag": 17.5,
        "amplitude": 4.5,
        "rise_days": 30.0,
        "decay_days": 80.0,
        "shape": "sn",
        "duration": 120.0,
        "n_obs": (30, 45),
    },
    "Kilonova": {
        "peak_mag": 21.0,
        "amplitude": 2.0,
        "rise_days": 2.0,
        "decay_days": 5.0,
        "shape": "sn",
        "duration": 12.0,
        "n_obs": (15, 25),
    },
    "AGN": {
        "peak_mag": 18.5,
        "amplitude": 0.8,
        "rise_days": 200.0,
        "decay_days": 200.0,
        "shape": "agn",
        "duration": 180.0,
        "n_obs": (30, 50),
    },
    "RRL": {
        "peak_mag": 19.0,
        "amplitude": 0.5,
        "rise_days": 0.2,
        "decay_days": 0.3,
        "shape": "rrl",
        "duration": 5.0,
        "n_obs": (25, 40),
    },
}

_BANDS = ["g", "r", "i", "z"]
_BAND_OFFSETS: dict[str, float] = {"g": 0.3, "r": 0.0, "i": -0.2, "z": -0.4}


def _sn_mag(
    t: np.ndarray,
    t0: float,
    peak_mag: float,
    rise: float,
    decay: float,
    amplitude: float,
) -> np.ndarray:
    """Compute supernova-like magnitude curve at times *t*.

    Parameters
    ----------
    t:
        Array of observation times (days relative to start).
    t0:
        Time of peak brightness.
    peak_mag:
        Magnitude at peak (minimum value = brightest).
    rise:
        Rise-time scale in days.
    decay:
        Decay-time scale in days.
    amplitude:
        Total magnitude swing.

    Returns
    -------
    np.ndarray
        Magnitude at each time *t*.
    """
    baseline = peak_mag + amplitude
    mag = np.where(
        t <= t0,
        baseline - amplitude * np.exp(-0.5 * ((t - t0) / rise) ** 2)
        + amplitude * (1 - np.exp(-0.5 * ((t - t0) / rise) ** 2)),
        peak_mag + amplitude * (1 - np.exp(-(t - t0) / decay)),
    )
    # Simpler: Gaussian rise, exponential decay.
    mag = peak_mag + amplitude * (
        1
        - np.exp(-0.5 * np.where(t <= t0, ((t - t0) / (rise + 1e-6)) ** 2, 0))
    )
    decay_part = peak_mag + amplitude * (1 - np.exp(-(t - t0) / (decay + 1e-6)))
    mag = np.where(t <= t0, mag, decay_part)
    return mag.astype(float)


def _plateau_mag(
    t: np.ndarray,
    t0: float,
    peak_mag: float,
    rise: float,
    amplitude: float,
) -> np.ndarray:
    """SN II plateau: rise then flat then drop.

    Parameters
    ----------
    t, t0, peak_mag, rise, amplitude:
        Same as :func:`_sn_mag`.

    Returns
    -------
    np.ndarray
    """
    plateau_end = t0 + rise * 2.0
    drop_scale = rise * 0.5
    mag = np.where(
        t <= t0,
        peak_mag + amplitude * np.exp(-((t - t0) / (rise + 1e-6)) ** 2),
        np.where(
            t <= plateau_end,
            peak_mag,
            peak_mag + amplitude * (1 - np.exp(-(t - plateau_end) / (drop_scale + 1e-6))),
        ),
    )
    return mag.astype(float)


def _agn_mag(
    t: np.ndarray,
    peak_mag: float,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """AGN-like slowly varying stochastic curve.

    Parameters
    ----------
    t:
        Observation times.
    peak_mag:
        Mean magnitude.
    amplitude:
        RMS variation scale.
    rng:
        NumPy random generator.

    Returns
    -------
    np.ndarray
    """
    noise = np.cumsum(rng.normal(0, amplitude / 10.0, size=len(t)))
    noise -= noise.mean()
    noise = np.clip(noise, -amplitude / 2, amplitude / 2)
    return (peak_mag + noise).astype(float)


def _rrl_mag(
    t: np.ndarray,
    peak_mag: float,
    amplitude: float,
    period: float = 0.5,
) -> np.ndarray:
    """RR Lyrae-like sinusoidal light curve.

    Parameters
    ----------
    t:
        Observation times.
    peak_mag:
        Mean magnitude.
    amplitude:
        Half-amplitude of oscillation.
    period:
        Period in days.

    Returns
    -------
    np.ndarray
    """
    return (peak_mag + amplitude * np.sin(2 * math.pi * t / period)).astype(float)


def _generate_lc(
    cls_name: str,
    tmpl: dict,
    rng: np.random.Generator,
) -> tuple[list[float], list[float], list[float], list[str]]:
    """Generate a synthetic multi-band light curve for a given class.

    Parameters
    ----------
    cls_name:
        Transient class name.
    tmpl:
        Template parameters dict from ``_TEMPLATES``.
    rng:
        NumPy random generator (already seeded).

    Returns
    -------
    tuple of (jd, mag, magerr, filter)
    """
    duration: float = float(tmpl["duration"])
    n_obs: int = int(rng.integers(tmpl["n_obs"][0], tmpl["n_obs"][1] + 1))
    # Sparse random cadence.
    t_sorted = np.sort(rng.uniform(0.0, duration, size=n_obs))
    t0 = float(tmpl["rise_days"])

    shape = tmpl["shape"]
    peak_mag: float = float(tmpl["peak_mag"]) + rng.normal(0, 0.3)
    amplitude: float = float(tmpl["amplitude"])
    rise: float = float(tmpl["rise_days"])
    decay: float = float(tmpl["decay_days"])

    # Assign random bands
    bands = rng.choice(_BANDS, size=n_obs)

    jd_out: list[float] = []
    mag_out: list[float] = []
    magerr_out: list[float] = []
    flt_out: list[str] = []

    for idx, (t_val, band) in enumerate(zip(t_sorted, bands)):
        t_arr = np.array([t_val])
        if shape == "sn":
            m_arr = _sn_mag(t_arr, t0, peak_mag, rise, decay, amplitude)
        elif shape == "plateau":
            m_arr = _plateau_mag(t_arr, t0, peak_mag, rise, amplitude)
        elif shape == "agn":
            # AGN: generate all at once but we only need this one point — use noise
            m_arr = np.array([peak_mag + rng.normal(0, amplitude / 4.0)])
        elif shape == "rrl":
            m_arr = _rrl_mag(t_arr, peak_mag, amplitude / 2.0)
        else:
            m_arr = np.array([peak_mag])

        band_offset = _BAND_OFFSETS.get(band, 0.0)
        # Kilonova: redder (larger offset in redder bands)
        if cls_name == "Kilonova":
            band_offset += _BANDS.index(band) * 0.5

        mag_val = float(m_arr[0]) + band_offset + rng.normal(0, 0.05)
        magerr_val = float(rng.uniform(0.02, 0.15))

        jd_out.append(float(t_val + 2459000.5))  # Arbitrary JD offset
        mag_out.append(round(mag_val, 4))
        magerr_out.append(round(magerr_val, 4))
        flt_out.append(str(band))

    return jd_out, mag_out, magerr_out, flt_out


class MockSource:
    """Infinite iterator of synthetic ``Alert`` objects.

    Parameters
    ----------
    n_per_batch:
        How many alerts to yield per iteration step.
    classes:
        List of class names to cycle through.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_per_batch: int = 10,
        classes: list[str] | None = None,
        seed: int = 0,
    ) -> None:
        self._n_per_batch = n_per_batch
        self._classes = classes or list(_TEMPLATES.keys())
        self._seed = seed
        self._counter = 0

    def __iter__(self) -> Iterator[Alert]:
        """Yield Alert objects indefinitely, one per iteration."""
        rng = np.random.default_rng(self._seed + self._counter)
        for _ in range(self._n_per_batch):
            cls_name = self._classes[self._counter % len(self._classes)]
            tmpl = _TEMPLATES.get(cls_name, _TEMPLATES["SN Ia"])
            jd, mag, magerr, flt = _generate_lc(cls_name, tmpl, rng)

            # Sky position: uniform RA, Gaussian-biased Dec away from galactic plane
            ra = float(rng.uniform(0.0, 24.0))
            # Bias away from equator slightly for realism
            dec = float(np.clip(rng.normal(0.0, 40.0), -89.9, 89.9))

            oid = f"MOCK-{cls_name.replace(' ', '')}-{uuid.uuid4().hex[:8]}"
            alert = Alert(
                object_id=oid,
                ra=ra,
                dec=dec,
                jd=jd,
                mag=mag,
                magerr=magerr,
                filter=flt,
                ingested_at=time.time(),
            )
            self._counter += 1
            rng = np.random.default_rng(self._seed + self._counter)
            yield alert
