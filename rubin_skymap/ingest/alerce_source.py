"""ALeRCE (Automatic Learning for the Rapid Classification of Events) alert source.

ALeRCE is a Chilean astronomy broker that classifies ZTF public alerts in near
real time using several ML models.  This source polls the public ALeRCE REST API
(https://api.alerce.online/ztf/v1) — no authentication required.

API endpoints used
------------------
GET  /ztf/v1/objects              list objects with classification filters
GET  /ztf/v1/objects/{oid}/lightcurve  full photometric history for one object

ALeRCE class mapping
---------------------
The ``stamp_classifier`` labels map to our internal names:
  SN  → SN Ia / SN II / SN Ibc  (re-classified by our LightGBM)
  AGN → AGN
  VS  → RRL  (variable star; may include RRL, Cepheids, etc.)
  asteroid → Other

The ``lc_classifier`` (light-curve based) labels used here:
  SNIa, SNII, SNIbc, SLSN-I, Kilonova, AGN, RRLyr
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterator

import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rubin_skymap.ingest.schemas import Alert

_log = logging.getLogger(__name__)

# ALeRCE public REST API base URL.
_API_BASE = "https://api.alerce.online/ztf/v1"

# ZTF filter-id → band name (same as ZTF/Fink: g=1, r=2, i=3).
_FID_TO_BAND: dict[int, str] = {1: "g", 2: "r", 3: "i"}

# MJD → JD offset.
_MJD_OFFSET = 2_400_000.5

# ALeRCE stamp_classifier class → our internal label.
# stamp_classifier gives a coarse SN/AGN/VS/asteroid label.
STAMP_CLASS_MAP: dict[str, str] = {
    "SN":       "SN Ia",    # re-classified finely by our LightGBM model
    "AGN":      "AGN",
    "VS":       "RRL",      # variable star (RRL/Ceph/EB)
    "asteroid": "Other",
    "bogus":    "Other",
    "Unknown":  "Other",
}

# ALeRCE lc_classifier class → our internal label.
# lc_classifier is more granular but not always available.
LC_CLASS_MAP: dict[str, str] = {
    "SNIa":       "SN Ia",
    "SNIbc":      "SN Ibc",
    "SNII":       "SN II",
    "SLSN-I":     "SLSN",
    "Kilonova":   "Kilonova",
    "AGN":        "AGN",
    "QSO":        "AGN",
    "RRLyr":      "RRL",
    "CEP":        "RRL",
    "DSCT":       "RRL",
    "EB":         "RRL",
    "LPV":        "RRL",
    "Periodic":   "RRL",
    "asteroid":   "Other",
    "bogus":      "Other",
    "Unknown":    "Other",
}


def _make_retry(retries: int, backoff: float):  # type: ignore[return]
    """Return a tenacity retry decorator for ALeRCE HTTP calls."""
    return retry(
        stop=stop_after_attempt(retries),
        wait=wait_exponential(multiplier=backoff, min=2.0, max=60.0),
        retry=retry_if_exception_type(
            (requests.RequestException, ValueError, OSError)
        ),
        before_sleep=before_sleep_log(_log, logging.DEBUG),
        reraise=True,
    )


# ALeRCE stamp_classifier class labels to query one at a time.
_POLL_CLASSES: list[str] = ["SN", "AGN", "VS"]


class AlerceSource:
    """Infinite iterator that polls the ALeRCE ZTF REST API for real alerts.

    Fetches the most recent classified objects from ALeRCE, downloads their
    full light curves, and yields normalised ``Alert`` objects.

    ALeRCE processes the full ZTF public alert stream and provides one of the
    richest public classification APIs available without authentication.

    Parameters
    ----------
    api_base:
        ALeRCE API base URL.
    timeout:
        Per-request HTTP timeout in seconds.
    retries:
        Retry attempts on failure.
    backoff_base:
        Exponential backoff multiplier (seconds).
    batch_size:
        Number of objects to fetch per poll cycle.
    min_detections:
        Minimum number of ZTF detections required to process an object.
    min_probability:
        Minimum classifier probability to accept an object.
    """

    def __init__(
        self,
        api_base: str = _API_BASE,
        timeout: float = 20.0,
        retries: int = 3,
        backoff_base: float = 2.0,
        batch_size: int = 10,
        min_detections: int = 5,
        min_probability: float = 0.0,
    ) -> None:
        self._api_base       = api_base.rstrip("/")
        self._timeout        = timeout
        self._retries        = retries
        self._backoff        = backoff_base
        self._batch_size     = batch_size
        self._min_dets       = min_detections
        self._min_prob       = min_probability
        self._page           = 1
        self._class_idx      = 0
        self._seen_oids: set[str] = set()

    # ── private ──────────────────────────────────────────────────────────

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET request with retry, returns parsed JSON or raises.

        Parameters
        ----------
        path:
            URL path relative to api_base.
        params:
            Query string parameters.

        Returns
        -------
        Any
            Parsed JSON body.
        """
        url = f"{self._api_base}{path}"

        @_make_retry(self._retries, self._backoff)
        def _do() -> Any:
            resp = requests.get(url, params=params, timeout=self._timeout)
            resp.raise_for_status()
            if not resp.content:
                raise ValueError(f"Empty response from {url}")
            return resp.json()

        return _do()

    def _fetch_objects(self) -> list[dict[str, Any]]:
        """Fetch a page of classified ZTF objects from ALeRCE.

        Cycles through SN, AGN, and VS classes one at a time, querying the
        API with a per-class filter so each poll returns diverse real objects
        rather than replaying the same alphabetical default result set.

        Returns
        -------
        list[dict]
            Raw ALeRCE object records, possibly empty.
        """
        # Pick the class to query this cycle and advance the index.
        poll_class = _POLL_CLASSES[self._class_idx % len(_POLL_CLASSES)]
        self._class_idx += 1

        try:
            data = self._get(
                "/objects",
                params={
                    "page":        self._page,
                    "page_size":   self._batch_size,
                    "classifier":  "stamp_classifier",
                    "class":       poll_class,
                    "ndet":        self._min_dets,
                    "format":      "json",
                },
            )
        except Exception as exc:
            _log.warning("AlerceSource: /objects request failed for class '%s': %s", poll_class, exc)
            return []

        items: list[dict[str, Any]] = (
            data.get("items", []) if isinstance(data, dict) else []
        )

        # Filter by minimum probability (already class-filtered by API).
        filtered = [
            item for item in items
            if item.get("probability", 0.0) >= self._min_prob
        ]

        has_next = (
            data.get("has_next", False) if isinstance(data, dict) else False
        )
        self._page = self._page + 1 if has_next else 1
        _log.info(
            "AlerceSource: class=%s page %d → %d objects (%d pass prob filter)",
            poll_class, self._page - 1, len(items), len(filtered),
        )
        return filtered

    def _fetch_lightcurve(self, oid: str) -> Alert | None:
        """Fetch the full light curve for *oid* and build a normalised Alert.

        Parameters
        ----------
        oid:
            ZTF object identifier (e.g. ``"ZTF18abcdefg"``).

        Returns
        -------
        Alert or None
            ``None`` if data is missing or too sparse.
        """
        try:
            data = self._get(f"/objects/{oid}/lightcurve")
        except Exception as exc:
            _log.warning("AlerceSource: lightcurve fetch failed for '%s': %s", oid, exc)
            return None

        if not isinstance(data, dict):
            return None

        detections: list[dict] = data.get("detections", [])
        if len(detections) < self._min_dets:
            _log.debug("AlerceSource: too few detections for '%s' (%d).", oid, len(detections))
            return None

        jd_list:     list[float] = []
        mag_list:    list[float] = []
        magerr_list: list[float] = []
        flt_list:    list[str]   = []
        ra_vals:     list[float] = []
        dec_vals:    list[float] = []

        for det in detections:
            try:
                mjd       = float(det.get("mjd", 0) or 0)
                mag       = float(det.get("magpsf", 0) or 0)
                magerr    = float(det.get("sigmapsf", 0.1) or 0.1)
                fid       = int(det.get("fid", 1) or 1)
                ra_raw    = float(det.get("ra", 0) or 0)
                dec_raw   = float(det.get("dec", 0) or 0)

                band = _FID_TO_BAND.get(fid, "g")

                # Validate: MJD > 2017 (ZTF started 58000 MJD ≈ 2017),
                # magnitude in plausible range.
                if mjd > 57_000 and 10.0 < mag < 25.0:
                    jd_list.append(mjd + _MJD_OFFSET)
                    mag_list.append(mag)
                    magerr_list.append(max(0.001, abs(magerr)))
                    flt_list.append(band)
                    if ra_raw > 0:
                        ra_vals.append(ra_raw)
                    if dec_raw != 0:
                        dec_vals.append(dec_raw)

            except (TypeError, ValueError, AttributeError) as exc:
                _log.debug("AlerceSource: bad detection in '%s': %s", oid, exc)
                continue

        if len(jd_list) < self._min_dets:
            _log.debug(
                "AlerceSource: too few valid epochs for '%s' (%d), skipping.", oid, len(jd_list)
            )
            return None

        # Mean sky position from all detections.
        ra_deg  = sum(ra_vals)  / len(ra_vals)  if ra_vals  else 0.0
        dec_deg = sum(dec_vals) / len(dec_vals) if dec_vals else 0.0
        # Convert RA from degrees → decimal hours.
        ra_hours = ra_deg / 15.0

        try:
            return Alert(
                object_id=oid,
                ra=ra_hours,
                dec=dec_deg,
                jd=jd_list,
                mag=mag_list,
                magerr=magerr_list,
                filter=flt_list,
                ingested_at=time.time(),
            )
        except Exception as exc:
            _log.warning("AlerceSource: Alert construction failed for '%s': %s", oid, exc)
            return None

    # ── public ───────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[Alert]:
        """Yield real ZTF alerts from ALeRCE, one at a time.

        One call fetches one page of objects and yields alerts for each.
        Returns after exhausting the page so the consumer can sleep between
        cycles.
        """
        objects = self._fetch_objects()
        for obj in objects:
            oid = str(obj.get("oid", ""))
            if not oid or oid in self._seen_oids:
                continue

            # Map ALeRCE class → internal label.
            alerce_cls   = str(obj.get("class", "Unknown"))
            internal_cls = STAMP_CLASS_MAP.get(alerce_cls, "Other")

            alert = self._fetch_lightcurve(oid)
            if alert is None:
                continue

            self._seen_oids.add(oid)

            # Small sleep between object fetches to respect rate limits.
            time.sleep(0.5)

            _log.info(
                "AlerceSource: %s  class=%s→%s  ra=%.3fh dec=%.2f°  n=%d epochs",
                oid, alerce_cls, internal_cls, alert.ra, alert.dec, len(alert.jd),
            )
            yield alert
