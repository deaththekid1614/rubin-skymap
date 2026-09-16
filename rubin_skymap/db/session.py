"""Database session management and CRUD helpers.

All public functions accept the database URL as a parameter so they can be
used from any context (API server, consumer, tests) without shared global
state.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator

from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from rubin_skymap.db.models import AlertRow, Base

_log = logging.getLogger(__name__)

# Engine cache keyed by URL string.
_engines: dict[str, Any] = {}
_session_factories: dict[str, sessionmaker] = {}


def _get_engine(url: str):  # type: ignore[return]
    """Return a cached SQLAlchemy engine for *url*.

    Parameters
    ----------
    url:
        SQLAlchemy connection string.
    """
    if url not in _engines:
        connect_args = {}
        if url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        _engines[url] = create_engine(url, connect_args=connect_args)
        _session_factories[url] = sessionmaker(
            bind=_engines[url], expire_on_commit=False
        )
    return _engines[url]


def init_db(url: str) -> None:
    """Create all tables defined in the ORM models.

    Safe to call multiple times (uses ``CREATE TABLE IF NOT EXISTS``).

    Parameters
    ----------
    url:
        SQLAlchemy connection string (e.g. ``"sqlite:///./rubin_skymap.db"``).
    """
    engine = _get_engine(url)
    Base.metadata.create_all(bind=engine)
    _log.info("Database initialised at %s", url)


@contextmanager
def session_scope(url: str) -> Generator[Session, None, None]:
    """Context manager that yields a database ``Session``.

    Commits on clean exit, rolls back on exception.

    Parameters
    ----------
    url:
        SQLAlchemy connection string.

    Yields
    ------
    Session
        An active SQLAlchemy session.
    """
    _get_engine(url)
    factory = _session_factories[url]
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def insert_alert(url: str, payload: dict[str, Any]) -> None:
    """Insert one prediction record into the ``alerts`` table.

    Silently ignores duplicate ``object_id`` (``IntegrityError``).

    Parameters
    ----------
    url:
        SQLAlchemy connection string.
    payload:
        Dict with keys matching ``AlertRow`` columns.
    """
    created_at = payload.get("created_at")
    if isinstance(created_at, str):
        try:
            created_at = datetime.fromisoformat(created_at)
        except ValueError:
            created_at = datetime.now(timezone.utc)
    elif not isinstance(created_at, datetime):
        created_at = datetime.now(timezone.utc)

    row = AlertRow(
        object_id=str(payload["object_id"]),
        ra=float(payload["ra"]),
        dec=float(payload["dec"]),
        predicted_class=str(payload["predicted_class"]),
        predicted_prob=float(payload["predicted_prob"]),
        features_json=str(payload.get("features_json", "{}")),
        shap_json=str(payload.get("shap_json", "[]")),
        created_at=created_at,
    )
    try:
        with session_scope(url) as session:
            session.add(row)
    except IntegrityError:
        _log.debug("Duplicate object_id '%s', skipping insert.", payload["object_id"])


def latest_alerts(url: str, limit: int = 200) -> list[dict[str, Any]]:
    """Return the most recent *limit* prediction records, newest first.

    Parameters
    ----------
    url:
        SQLAlchemy connection string.
    limit:
        Maximum number of rows to return.

    Returns
    -------
    list[dict]
        Each dict has the same keys as ``AlertRow`` columns (``id`` excluded).
    """
    with session_scope(url) as session:
        stmt = (
            select(AlertRow)
            .order_by(AlertRow.created_at.desc())
            .limit(limit)
        )
        rows = session.execute(stmt).scalars().all()
        return [_row_to_dict(r) for r in rows]


def class_counts(url: str) -> dict[str, int]:
    """Return the count of predictions per class.

    Parameters
    ----------
    url:
        SQLAlchemy connection string.

    Returns
    -------
    dict[str, int]
        Mapping of class label → count.
    """
    with session_scope(url) as session:
        stmt = select(AlertRow.predicted_class, func.count(AlertRow.id)).group_by(
            AlertRow.predicted_class
        )
        results = session.execute(stmt).all()
        return {str(cls): int(cnt) for cls, cnt in results}


def _row_to_dict(row: AlertRow) -> dict[str, Any]:
    """Convert an ``AlertRow`` ORM object to a plain serialisable dict.

    Parameters
    ----------
    row:
        ORM row instance.

    Returns
    -------
    dict[str, Any]
    """
    return {
        "object_id": row.object_id,
        "ra": row.ra,
        "dec": row.dec,
        "predicted_class": row.predicted_class,
        "predicted_prob": row.predicted_prob,
        "features_json": row.features_json,
        "shap_json": row.shap_json,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }
