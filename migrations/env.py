"""
Alembic Environment Configuration – EcoPackAI
==============================================

This module is executed by Alembic whenever a migration command is run.
It configures the SQLAlchemy engine, connects to the target database,
and runs migrations in either *offline* (SQL-script) or *online*
(live-connection) mode.

The database URL is resolved in this order:

1. ``ECOPACKAI_DATABASE_URL`` environment variable (highest priority).
2. ``sqlalchemy.url`` value in ``alembic.ini``.
"""

from __future__ import annotations

import logging
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool
from sqlalchemy.orm import declarative_base

# ---------------------------------------------------------------------------
# Alembic Config object – provides access to values in alembic.ini
# ---------------------------------------------------------------------------
config = context.config

# Interpret the config file for Python logging if present.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

logger = logging.getLogger("alembic.env")

# ---------------------------------------------------------------------------
# Target metadata
# ---------------------------------------------------------------------------
# Import or define the declarative Base whose ``metadata`` Alembic will use
# for autogenerate support.  If you have a central ``models.py``, import its
# ``Base`` here instead of creating a new one.
#
# Example:
#   from ecopackai.models import Base
#   target_metadata = Base.metadata

Base = declarative_base()
target_metadata = Base.metadata

# ---------------------------------------------------------------------------
# URL resolution
# ---------------------------------------------------------------------------


def _get_url() -> str:
    """Return the database URL, preferring the environment variable."""
    env_url = os.environ.get("ECOPACKAI_DATABASE_URL")
    if env_url:
        logger.info("Using database URL from ECOPACKAI_DATABASE_URL env var.")
        return env_url
    url = config.get_main_option("sqlalchemy.url")
    if not url:
        raise RuntimeError(
            "No database URL configured.  Set ECOPACKAI_DATABASE_URL or "
            "sqlalchemy.url in alembic.ini."
        )
    return url


# ---------------------------------------------------------------------------
# Offline migrations  (generate SQL without a live DB connection)
# ---------------------------------------------------------------------------


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine, though an
    Engine is also acceptable here.  By skipping engine creation we don't even
    need a DBAPI to be available.

    Calls to ``context.execute()`` here emit the given string to the script
    output.
    """
    url = _get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online migrations  (run against a live database)
# ---------------------------------------------------------------------------


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we create an Engine and associate a connection with the
    context.  A transaction is used so that all migrations either succeed or
    roll back atomically.
    """
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _get_url()

    connectable = create_engine(
        configuration["sqlalchemy.url"],
        poolclass=pool.NullPool,
        future=True,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            transaction_per_migration=False,
        )

        with context.begin_transaction():
            context.run_migrations()

    connectable.dispose()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if context.is_offline_mode():
    logger.info("Running migrations in OFFLINE mode.")
    run_migrations_offline()
else:
    logger.info("Running migrations in ONLINE mode.")
    run_migrations_online()
