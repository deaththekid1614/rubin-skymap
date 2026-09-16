"""Light-curve feature extraction.

All functions are pure (no IO) and NaN-safe.  The main entry-point is
:func:`extract_features`, which maps an ``Alert`` object to a flat
``dict[str, float]``.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from rubin_skymap.ingest.schemas import Alert

_log = logging.getLogger(__name__)

# Canonical feature names in the order they will appear in the DataFrame.
FEATURE_NAMES: list[str] = [
    "n_detections",
    "time_span",
    "rise_time",
    "decay_time",
    "peak_mag",
    "amplitude",
    "mean_mag",
    "std_mag",
    "skew_mag",
    "kurt_mag",
    "color_gr_at_peak",
    "color_ri_at_peak",
    "cadence_mean",
    "cadence_std",
]


def _nan_dict() -> dict[str, float]:
    """Return a dict with every feature set to ``float('nan')``.

    Returns
    -------
    dict[str, float]
    """
    return {k: float("nan") for k in FEATURE_NAMES}


def _safe_float(value: Any) -> float:
    """Convert *value* to float, returning ``nan`` on failure.

    Parameters
    ----------
    value:
        Any object that may or may not convert to float.

    Returns
    -------
    float
    """
    try:
        result = float(value)
        return result if math.isfinite(result) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _interpolate_at(
    jd_arr: np.ndarray,
    mag_arr: np.ndarray,
    target_jd: float,
) -> float:
    """Linear interpolation of *mag_arr* at *target_jd*.

    Uses the two nearest observations.  Returns ``nan`` if fewer than two
    points are available.

    Parameters
    ----------
    jd_arr:
        Sorted Julian-date array for one band.
    mag_arr:
        Corresponding magnitude array.
    target_jd:
        The Julian date at which to interpolate.

    Returns
    -------
    float
        Interpolated magnitude or ``nan``.
    """
    if len(jd_arr) < 2:
        return float("nan")
    idx = int(np.searchsorted(jd_arr, target_jd))
    idx = max(1, min(idx, len(jd_arr) - 1))
    jd0, jd1 = jd_arr[idx - 1], jd_arr[idx]
    m0, m1 = mag_arr[idx - 1], mag_arr[idx]
    if jd1 == jd0:
        return float(m0)
    frac = (target_jd - jd0) / (jd1 - jd0)
    return float(m0 + frac * (m1 - m0))


def extract_features(alert: Alert) -> dict[str, float]:
    """Extract a fixed set of numerical features from a single alert.

    Parameters
    ----------
    alert:
        A validated :class:`~rubin_skymap.ingest.schemas.Alert` object.

    Returns
    -------
    dict[str, float]
        Mapping of feature name → float value.  Any feature that cannot be
        computed (missing band, insufficient points, …) is set to ``nan``.

    Notes
    -----
    All features are NaN-safe: the function never raises; it returns
    ``nan`` for any feature that cannot be computed.

    Feature definitions
    -------------------
    n_detections
        Total number of observations across all bands.
    time_span
        ``jd[-1] - jd[0]`` (days).
    rise_time
        ``jd[peak_idx] - jd[0]`` where ``peak_idx = argmin(mag)`` (days).
    decay_time
        ``jd[-1] - jd[peak_idx]`` (days).
    peak_mag
        ``min(mag)`` (brightest point).
    amplitude
        ``max(mag) - min(mag)``.
    mean_mag
        Arithmetic mean of all magnitudes.
    std_mag
        Standard deviation of magnitudes.
    skew_mag
        Skewness of the magnitude distribution.
    kurt_mag
        Excess kurtosis of the magnitude distribution.
    color_gr_at_peak
        g − r interpolated at ``jd[peak_idx]``.  ``nan`` if g or r band absent.
    color_ri_at_peak
        r − i interpolated at ``jd[peak_idx]``.  ``nan`` if r or i band absent.
    cadence_mean
        Mean of ``diff(jd)`` (days between consecutive sorted observations).
    cadence_std
        Std of ``diff(jd)``.
    """
    result = _nan_dict()

    try:
        jd_arr = np.asarray(alert.jd, dtype=float)
        mag_arr = np.asarray(alert.mag, dtype=float)
        flt_arr = np.asarray(alert.filter)
    except (TypeError, ValueError) as exc:
        _log.debug("extract_features: failed to convert arrays: %s", exc)
        return result

    n = len(jd_arr)
    if n == 0:
        return result

    # Basic counts and time span.
    result["n_detections"] = float(n)
    jd_sorted_idx = np.argsort(jd_arr)
    jd_s = jd_arr[jd_sorted_idx]
    mag_s = mag_arr[jd_sorted_idx]
    flt_s = flt_arr[jd_sorted_idx]

    result["time_span"] = float(jd_s[-1] - jd_s[0]) if n > 1 else 0.0

    # Peak detection (minimum magnitude = brightest).
    peak_idx = int(np.nanargmin(mag_s))
    peak_jd = float(jd_s[peak_idx])

    result["peak_mag"] = _safe_float(mag_s[peak_idx])
    result["rise_time"] = float(peak_jd - jd_s[0])
    result["decay_time"] = float(jd_s[-1] - peak_jd)
    result["amplitude"] = _safe_float(np.nanmax(mag_s) - np.nanmin(mag_s))

    # Aggregate statistics.
    result["mean_mag"] = _safe_float(np.nanmean(mag_s))
    result["std_mag"] = _safe_float(np.nanstd(mag_s))

    # Skewness (Fisher definition, unbiased via scipy-free manual calc).
    if n >= 3:
        mu = float(np.nanmean(mag_s))
        sigma = float(np.nanstd(mag_s, ddof=1))
        if sigma > 0:
            skew = float(np.nanmean(((mag_s - mu) / sigma) ** 3))
            result["skew_mag"] = skew
            kurt = float(np.nanmean(((mag_s - mu) / sigma) ** 4)) - 3.0
            result["kurt_mag"] = kurt

    # Cadence statistics.
    if n >= 2:
        gaps = np.diff(jd_s)
        result["cadence_mean"] = _safe_float(np.mean(gaps))
        result["cadence_std"] = _safe_float(np.std(gaps))

    # Per-band arrays for colour computation.
    band_jd: dict[str, np.ndarray] = {}
    band_mag: dict[str, np.ndarray] = {}
    for band in ("g", "r", "i"):
        mask = flt_s == band
        if np.any(mask):
            band_jd[band] = jd_s[mask]
            band_mag[band] = mag_s[mask]

    # color_gr_at_peak = g - r interpolated at peak_jd.
    if "g" in band_jd and "r" in band_jd:
        g_at_peak = _interpolate_at(band_jd["g"], band_mag["g"], peak_jd)
        r_at_peak = _interpolate_at(band_jd["r"], band_mag["r"], peak_jd)
        if math.isfinite(g_at_peak) and math.isfinite(r_at_peak):
            result["color_gr_at_peak"] = g_at_peak - r_at_peak

    # color_ri_at_peak = r - i interpolated at peak_jd.
    if "r" in band_jd and "i" in band_jd:
        r_at_peak = _interpolate_at(band_jd["r"], band_mag["r"], peak_jd)
        i_at_peak = _interpolate_at(band_jd["i"], band_mag["i"], peak_jd)
        if math.isfinite(r_at_peak) and math.isfinite(i_at_peak):
            result["color_ri_at_peak"] = r_at_peak - i_at_peak

    return result
