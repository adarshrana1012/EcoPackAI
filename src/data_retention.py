"""
data_retention.py — Data Retention Celery Task (Prompt 34)
==========================================================

Enforces data retention policies:
* Delete shipment records older than 90 days (keep aggregates).
* Anonymize product_id in records older than 30 days.

Scheduled via Celery Beat at 03:00 UTC every Sunday.

Author: EcoPackAI Team
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Celery App Configuration
# ---------------------------------------------------------------------------
try:
    from celery import Celery
    from celery.schedules import crontab

    app = Celery(
        "ecopackai_retention",
        broker="redis://localhost:6379/0",
        backend="redis://localhost:6379/1",
    )

    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        beat_schedule={
            "weekly-data-retention": {
                "task": "src.data_retention.enforce_data_retention",
                "schedule": crontab(
                    hour=3, minute=0,
                    day_of_week="sunday",
                ),
                "args": (),
            },
        },
    )
    _HAS_CELERY = True
except ImportError:
    logger.warning("Celery not installed. Task will run synchronously.")

    class _MockCelery:
        class task:
            def __init__(self, *a, **kw):
                pass
            def __call__(self, func):
                func.delay = lambda *a, **kw: func(*a, **kw)
                return func

    app = _MockCelery()
    _HAS_CELERY = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HARD_DELETE_DAYS = 90    # Delete records older than this
ANONYMIZE_DAYS = 30      # Anonymize product_id older than this
ANONYMIZED_VALUE = "ANONYMIZED"
AGGREGATE_TABLE = "shipment_aggregates"


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _get_session(conn_str: Optional[str] = None) -> Any:
    """Get a SQLAlchemy session.  Falls back to None if no DB."""
    if conn_str is None:
        return None
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        engine = create_engine(conn_str)
        Session = sessionmaker(bind=engine)
        return Session()
    except Exception as e:
        logger.warning("Could not create DB session: %s", e)
        return None


def _save_aggregate_summary(
    session: Any,
    records_deleted: int,
    oldest_date: str,
    newest_date: str,
    avg_void_pct: float,
    total_co2e_kg: float,
) -> None:
    """Save an aggregate summary before deleting records."""
    if session is None:
        logger.info(
            "Aggregate summary (no DB): deleted=%d, period=%s to %s, "
            "avg_void=%.1f%%, co2e=%.2fkg",
            records_deleted, oldest_date, newest_date,
            avg_void_pct, total_co2e_kg,
        )
        return

    try:
        from sqlalchemy import text
        session.execute(text(f"""
            INSERT INTO {AGGREGATE_TABLE}
            (period_start, period_end, records_count, avg_void_pct,
             total_co2e_kg, created_at)
            VALUES (:start, :end, :count, :avg_void, :co2e, :created)
        """), {
            "start": oldest_date,
            "end": newest_date,
            "count": records_deleted,
            "avg_void": avg_void_pct,
            "co2e": total_co2e_kg,
            "created": datetime.now(timezone.utc).isoformat(),
        })
        session.commit()
        logger.info("Aggregate summary saved for period %s to %s.",
                     oldest_date, newest_date)
    except Exception as e:
        logger.warning("Could not save aggregate: %s", e)


# ---------------------------------------------------------------------------
# Main Task
# ---------------------------------------------------------------------------

@app.task(bind=True, name="src.data_retention.enforce_data_retention")
def enforce_data_retention(
    self: Any = None,
    conn_str: Optional[str] = None,
    hard_delete_days: int = HARD_DELETE_DAYS,
    anonymize_days: int = ANONYMIZE_DAYS,
) -> Dict[str, Any]:
    """Enforce data retention policy on shipment records.

    Steps
    -----
    1. Compute aggregate summaries for records about to be deleted.
    2. Delete shipment records older than 90 days.
    3. Anonymize product_id in records older than 30 days.
    4. Log all actions with counts.

    Parameters
    ----------
    conn_str : str, optional
        Database connection string.
    hard_delete_days : int
        Days after which records are hard-deleted.
    anonymize_days : int
        Days after which product_id is anonymized.

    Returns
    -------
    dict
        Retention enforcement outcome.
    """
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logger.info("=" * 60)
    logger.info("DATA RETENTION — run_id=%s", run_id)
    logger.info("=" * 60)

    result = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hard_delete_days": hard_delete_days,
        "anonymize_days": anonymize_days,
        "status": "started",
        "records_deleted": 0,
        "records_anonymized": 0,
    }

    session = _get_session(conn_str)

    try:
        delete_cutoff = datetime.now(timezone.utc) - timedelta(days=hard_delete_days)
        anonymize_cutoff = datetime.now(timezone.utc) - timedelta(days=anonymize_days)

        if session is not None:
            from sqlalchemy import text, func
            from src.dashboard_queries import ShipmentRecord

            # --- Step 1: Aggregate summaries before deletion ---------------
            old_records = (
                session.query(
                    func.count(ShipmentRecord.id),
                    func.min(ShipmentRecord.packed_at),
                    func.max(ShipmentRecord.packed_at),
                    func.avg(ShipmentRecord.void_volume_pct),
                    func.sum(ShipmentRecord.co2e_kg),
                )
                .filter(ShipmentRecord.packed_at < delete_cutoff)
                .first()
            )

            records_to_delete = int(old_records[0] or 0)

            if records_to_delete > 0:
                _save_aggregate_summary(
                    session,
                    records_to_delete,
                    str(old_records[1]),
                    str(old_records[2]),
                    float(old_records[3] or 0),
                    float(old_records[4] or 0),
                )

                # --- Step 2: Hard delete old records -----------------------
                deleted = (
                    session.query(ShipmentRecord)
                    .filter(ShipmentRecord.packed_at < delete_cutoff)
                    .delete(synchronize_session="fetch")
                )
                session.commit()
                result["records_deleted"] = deleted
                logger.info(
                    "Deleted %d records older than %d days (cutoff: %s).",
                    deleted, hard_delete_days, delete_cutoff.isoformat(),
                )

            # --- Step 3: Anonymize product_id in older records -------------
            anonymized = (
                session.query(ShipmentRecord)
                .filter(
                    ShipmentRecord.packed_at < anonymize_cutoff,
                    ShipmentRecord.product_id != ANONYMIZED_VALUE,
                )
                .update(
                    {ShipmentRecord.product_id: ANONYMIZED_VALUE},
                    synchronize_session="fetch",
                )
            )
            session.commit()
            result["records_anonymized"] = anonymized
            logger.info(
                "Anonymized product_id in %d records older than %d days.",
                anonymized, anonymize_days,
            )

        else:
            logger.info(
                "No DB connection — simulating retention enforcement. "
                "Would delete records before %s and anonymize before %s.",
                delete_cutoff.isoformat(), anonymize_cutoff.isoformat(),
            )
            result["simulated"] = True

        result["status"] = "completed"

    except Exception as e:
        logger.error("Data retention failed: %s", e, exc_info=True)
        result["status"] = "failed"
        result["error"] = str(e)
    finally:
        if session is not None:
            session.close()

    logger.info("Retention outcome: %s", json.dumps(result, indent=2, default=str))
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    result = enforce_data_retention()
    print(json.dumps(result, indent=2, default=str))
