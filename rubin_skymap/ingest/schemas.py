"""Pydantic schema for a normalised alert object."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, field_validator


class Alert(BaseModel):
    """A normalised astronomical alert carrying a multi-band light curve.

    All array fields must have the same length.

    Attributes
    ----------
    object_id:
        Unique source identifier (e.g. ``"ZTF21abcdefg"`` or ``"MOCK-SN Ia-a1b2c3d4"``).
    ra:
        Right ascension in decimal hours ``[0, 24)``.
    dec:
        Declination in decimal degrees ``[-90, 90]``.
    jd:
        Julian dates of each observation.
    mag:
        Apparent magnitudes (smaller = brighter).
    magerr:
        1-sigma magnitude uncertainties (positive).
    filter:
        Photometric band for each observation; one of ``{u,g,r,i,z,y}``.
    ingested_at:
        Unix timestamp (UTC) when the alert was received.
    """

    object_id: str
    ra: float
    dec: float
    jd: list[float]
    mag: list[float]
    magerr: list[float]
    filter: list[str]
    ingested_at: float = 0.0

    @field_validator("filter", mode="before")
    @classmethod
    def _validate_filters(cls, value: list[str]) -> list[str]:
        """Ensure every filter value is one of the six LSST bands."""
        allowed = {"u", "g", "r", "i", "z", "y"}
        for band in value:
            if band not in allowed:
                raise ValueError(f"Unknown filter '{band}'; must be one of {allowed}")
        return value

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Alert":
        """Construct an Alert from a plain dictionary.

        Parameters
        ----------
        d:
            Mapping with keys matching the Alert fields.

        Returns
        -------
        Alert
        """
        if d.get("ingested_at", 0.0) == 0.0:
            d = {**d, "ingested_at": time.time()}
        return cls(**d)

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of this alert.

        Returns
        -------
        dict
        """
        return self.model_dump()
