"""Database engine configuration and connection-pool lifecycle."""

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from app.config import Settings


def normalize_database_url(url: str) -> str:
    """Use SQLAlchemy's psycopg dialect for PostgreSQL URLs."""
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def create_database_engine(settings: Settings) -> Engine:
    """Create a pre-pinged PostgreSQL pool, rejecting missing DB configuration."""
    if settings.database_url is None:
        raise RuntimeError("DATABASE_URL must be configured")

    url = normalize_database_url(settings.database_url.get_secret_value())

    return create_engine(
        url,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        pool_timeout=settings.database_pool_timeout_seconds,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_use_lifo=True,
        connect_args={"connect_timeout": settings.database_connect_timeout_seconds},
    )
