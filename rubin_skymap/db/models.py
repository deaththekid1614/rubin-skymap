"""SQLAlchemy 2.0 ORM model for the ``alerts`` table."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    pass


class AlertRow(Base):
    """Persisted prediction record for one astronomical alert.

    Attributes
    ----------
    id:
        Auto-incrementing primary key.
    object_id:
        Unique source identifier.  Indexed for fast lookups.
    ra:
        Right ascension in decimal hours.
    dec:
        Declination in decimal degrees.
    predicted_class:
        String label of the winning class.
    predicted_prob:
        Probability of the predicted class.
    features_json:
        JSON-encoded feature dict.
    shap_json:
        JSON-encoded SHAP explanation list.
    created_at:
        UTC timestamp of when the prediction was persisted.
    """

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    object_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    ra: Mapped[float] = mapped_column(Float, nullable=False)
    dec: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_class: Mapped[str] = mapped_column(String(64), nullable=False)
    predicted_prob: Mapped[float] = mapped_column(Float, nullable=False)
    features_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    shap_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    __table_args__ = (Index("ix_alerts_object_id", "object_id"),)
