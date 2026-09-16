"""Fink broker REST API alert source.

Polls the Fink portal REST API (https://api.fink-portal.org) and yields
normalised ``Alert`` objects.  Every HTTP call is wrapped with ``tenacity``
retry logic.  On any error the source logs a warning and yields nothing for
that cycle — it never crashes the consumer loop.

Real-data usage
---------------
Set ``ingest.mode: fink`` in ``configs/dev.yaml`` (or set the env var
``RUBIN_SKYMAP_ENV=prod`` with ``configs/prod.yaml``).  No credentials are
required for the public Fink REST API.  The Fink portal uses ZTF public alert
data and exposes a REST interface — no Kafka subscription needed.

The Fink "class" names used here match the labels returned by the Fink
ML classifiers (e.g. ``"SN Ia"``, ``"AGN"``).  A full list is available at
https://fink-portal.org/api under ``/api/v1/classes``.
"""

from __future__ import annotations

import logging
import os
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

# ZTF filter-id → LSST-like band name (ZTF: g=1, r=2, i=3).
_FID_TO_BAND: dict[int, str] = {1: "g", 2: "r", 3: "i"}

# Columns to request in the /latests call.
# Using both the internal (i:) and derived (d:) prefixes that Fink uses.
_LATESTS_COLS = "i:objectId,d:ra,d:dec"

# Fink class names as they appear in the broker API.
# These are the labels used in the /api/v1/latests endpoint.
FINK_CLASS_NAMES: dict[str, str] = {
    "SN Ia":   "SN Ia",
    "SN II":   "SN II",
    "SN Ibc":  "SN Ibc",
    "SLSN":    "SLSN-I",         # Fink uses "SLSN-I"
    "Kilonova":"Kilonova",
    "AGN":     "AGN",
    "RRL":     "RRLyr",          # Fink uses "RRLyr"
    "Other":   "Unknown",
}


def _make_retry(retries: int, backoff_base: float):  # type: ignore[return]
    """Return a configured tenacity retry decorator.

    Parameters
    ----------
    retries:
        Maximum number of attempts before giving up.
    backoff_base:
        Exponential backoff multiplier in seconds.
    """
    return retry(
        stop=stop_after_attempt(retries),
        wait=wait_exponential(multiplier=backoff_base, min=1.0, max=60.0),
        retry=retry_if_exception_type((requests.RequestException, ValueError, OSError)),
        before_sleep=before_sleep_log(_log, logging.DEBUG),
        reraise=True,
    )


def test_connection(api_base: str, timeout: float = 8.0) -> bool:
    """Return True if the Fink API is reachable.

    Sends a minimal request to /api/v1/classes to check connectivity.
    Does not raise — always returns bool.

    Parameters
    ----------
    api_base:
        Base URL of the Fink portal.
    timeout:
        Socket timeout in seconds.

    Returns
    -------
    bool
    """
    try:
        resp = requests.get(
            f"{api_base.rstrip('/')}/api/v1/classes",
            timeout=timeout,
        )
        return resp.status_code == 200
    except Exception as exc:
        _log.debug("Fink connectivity check failed: %s", exc)
        return False


class FinkSource:
    """Infinite iterator that polls the Fink broker REST API.

    Cycles through the configured list of transient classes, fetching the
    N most recent alerts for each class per poll cycle.  Light-curve data
    is fetched from ``/api/v1/objects`` for each object ID returned by
    ``/api/v1/latests``.

    Parameters
    ----------
    api_base:
        Base URL, e.g. ``"https://api.fink-portal.org"``.
    timeout:
        Per-request HTTP timeout in seconds.
    retries:
        How many times to retry a failed request.
    backoff_base:
        Exponential backoff base multiplier (seconds).
    classes:
        Internal class labels to poll (mapped to Fink labels automatically).
    batch_size:
        Number of recent objects to fetch per class per cycle.
    """

    def __init__(
        self,
        api_base: str,
        timeout: float,
        retries: int,
        backoff_base: float,
        classes: list[str],
        batch_size: int = 5,
    ) -> None:
        self._api_base   = api_base.rstrip("/")
        self._timeout    = timeout
        self._retries    = retries
        self._backoff    = backoff_base
        self._classes    = classes
        self._batch_size = batch_size
        self._class_idx  = 0

        # Optional HTTP basic auth (not required for public Fink API).
        user = os.environ.get("FINK_USER", "")
        pwd  = os.environ.get("FINK_PASSWORD", "")
        self._auth: tuple[str, str] | None = (user, pwd) if (user and pwd) else None

    # ─── private ───────────────────────────────────────────────────────────

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Execute an HTTP request and return parsed JSON.

        Parameters
        ----------
        method:
            ``"GET"`` or ``"POST"``.
        path:
            URL path, e.g. ``"/api/v1/latests"``.
        **kwargs:
            Passed through to ``requests.request``.

        Returns
        -------
        Any
            Parsed JSON body.

        Raises
        ------
        requests.RequestException
            On network or HTTP error.
        ValueError
            When the response body is not valid JSON.
        """
        url = f"{self._api_base}{path}"
        kwargs.setdefault("timeout", self._timeout)
        if self._auth:
            kwargs["auth"] = self._auth

        resp = requests.request(method, url, **kwargs)
        resp.raise_for_status()
        data = resp.json()
        return data

    def _fetch_latest_oids(self, fink_class: str) -> list[str]:
        """Fetch the most recent ZTF object IDs for *fink_class*.

        Parameters
        ----------
        fink_class:
            Fink broker class label (e.g. ``"SN Ia"``).

        Returns
        -------
        list[str]
            Object IDs, empty list on failure.
        """
        do_request = _make_retry(self._retries, self._backoff)(self._request)
        try:
            data = do_request(
                "POST",
                "/api/v1/latests",
                json={
                    "class": fink_class,
                    "n": str(self._batch_size),
                    "columns": _LATESTS_COLS,
                },
            )
        except Exception as exc:
            _log.warning("FinkSource: /latests failed for '%s': %s", fink_class, exc)
            return []

        if not isinstance(data, list):
            _log.debug("FinkSource: unexpected /latests response type: %s", type(data))
            return []

        oids: list[str] = []
        for row in data:
            # Fink returns keys like "i:objectId"
            oid = (
                row.get("i:objectId")
                or row.get("objectId")
                or row.get("d:objectId")
            )
            if oid:
                oids.append(str(oid))
        _log.debug("FinkSource: found %d OIDs for class '%s'", len(oids), fink_class)
        return oids

    def _fetch_lightcurve(self, oid: str) -> Alert | None:
        """Fetch the full photometric history for *oid* and build an Alert.

        Uses the ``/api/v1/objects`` endpoint which returns one JSON row per
        observation epoch, including the ZTF passband filter ID, Julian date,
        magnitude (PSF), and magnitude error.

        Parameters
        ----------
        oid:
            ZTF object identifier, e.g. ``"ZTF21abcdefg"``.

        Returns
        -------
        Alert or None
        """
        do_request = _make_retry(self._retries, self._backoff)(self._request)
        try:
            data = do_request(
                "POST",
                "/api/v1/objects",
                json={"objectId": oid, "output-format": "json"},
            )
        except Exception as exc:
            _log.warning("FinkSource: /objects failed for '%s': %s", oid, exc)
            return None

        if not isinstance(data, list) or len(data) == 0:
            _log.debug("FinkSource: empty /objects response for '%s'", oid)
            return None

        jd_list:     list[float] = []
        mag_list:    list[float] = []
        magerr_list: list[float] = []
        flt_list:    list[str]   = []
        ra_deg:  float = 0.0
        dec_deg: float = 0.0

        for row in data:
            try:
                jd_val    = float(row.get("i:jd",      0) or 0)
                mag_val   = float(row.get("i:magpsf",   0) or 0)
                magerr_val = float(row.get("i:sigmapsf", 0.1) or 0.1)
                fid        = int(row.get("i:fid", 1) or 1)
                band       = _FID_TO_BAND.get(fid, "g")

                # RA/Dec: Fink returns degrees in both "d:ra"/"d:dec" (derived)
                # and "i:ra"/"i:dec" (raw ZTF).
                ra_raw  = row.get("d:ra")  or row.get("i:ra")  or 0.0
                dec_raw = row.get("d:dec") or row.get("i:dec") or 0.0
                ra_deg  = float(ra_raw)
                dec_deg = float(dec_raw)

                # Only keep valid photometry.
                if jd_val > 2_400_000 and 10 < mag_val < 25:
                    jd_list.append(jd_val)
                    mag_list.append(mag_val)
                    magerr_list.append(max(0.001, abs(magerr_val)))
                    flt_list.append(band)

            except (TypeError, ValueError, AttributeError) as exc:
                _log.debug("FinkSource: malformed row in '%s': %s", oid, exc)
                continue

        if len(jd_list) < 3:
            _log.debug(
                "FinkSource: too few valid epochs for '%s' (%d), skipping.", oid, len(jd_list)
            )
            return None

        # Convert RA from degrees to decimal hours.
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
            _log.warning("FinkSource: Alert construction failed for '%s': %s", oid, exc)
            return None

    # ─── public ────────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[Alert]:
        """Yield one Alert at a time, cycling through configured classes.

        One call to ``__iter__`` processes a single class then returns,
        so the consumer can sleep between cycles.  The next call picks up
        the next class.
        """
        cls_internal = self._classes[self._class_idx % len(self._classes)]
        self._class_idx += 1

        # Map internal class name to Fink API label.
        fink_label = FINK_CLASS_NAMES.get(cls_internal, cls_internal)
        _log.info("FinkSource: polling '%s' (Fink label: '%s')", cls_internal, fink_label)

        oids = self._fetch_latest_oids(fink_label)
        for oid in oids:
            alert = self._fetch_lightcurve(oid)
            if alert is not None:
                _log.info(
                    "FinkSource: yielding %s  ra=%.2fh dec=%.2f°  n=%d epochs",
                    oid, alert.ra, alert.dec, len(alert.jd),
                )
                yield alert
