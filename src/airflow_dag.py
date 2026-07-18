"""
EcoPackAI Airflow DAG – Daily Data Preprocessing & Ingestion Pipeline
=====================================================================

This DAG orchestrates the end-to-end data flow for the EcoPackAI platform:

1. **ingest_raw_data** – Reads a raw CSV from the configured data-source path,
   persists it as a dated Parquet snapshot, and pushes the output path via XCom.
2. **validate_schema** – Pulls the ingested file path from XCom and runs schema
   validation (column names, dtypes, null checks, value-range constraints).
3. **run_feature_engineering** – Applies the EcoPackAI feature pipeline (derived
   columns, encodings, scaling) and writes a feature-store Parquet file.
4. **upload_to_postgres** – Loads the feature-engineered data into the
   ``products`` table in PostgreSQL via SQLAlchemy bulk insert.

All inter-task communication uses **file paths over XCom** (never raw
DataFrames) to keep the metadata DB lightweight.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("ecopackai.dag")

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
DATA_SOURCE_PATH: str = Variable.get(
    "ecopackai_data_source_path",
    default_var="/opt/airflow/data/raw/products.csv",
)
STAGING_DIR: str = Variable.get(
    "ecopackai_staging_dir",
    default_var="/opt/airflow/data/staging",
)
FEATURE_STORE_DIR: str = Variable.get(
    "ecopackai_feature_store_dir",
    default_var="/opt/airflow/data/features",
)
POSTGRES_CONN_STR: str = Variable.get(
    "ecopackai_postgres_conn",
    default_var="postgresql://ecopackai:ecopackai@localhost:5432/ecopackai",
)

# Required columns and their expected pandas dtypes after ingestion.
EXPECTED_SCHEMA: Dict[str, str] = {
    "product_id": "object",
    "length_cm": "float64",
    "width_cm": "float64",
    "height_cm": "float64",
    "weight_g": "float64",
    "material_type": "object",
    "fragility_label": "int64",
}


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def _ingest_raw_data(**context: Any) -> str:
    """Read a raw CSV from the configured data-source path and persist it as a
    dated Parquet snapshot in the staging directory.

    The output file path is pushed to XCom so that downstream tasks can locate
    the snapshot without hard-coding paths.

    Parameters
    ----------
    **context : Any
        Airflow task-instance context (automatically injected).

    Returns
    -------
    str
        Absolute path to the staged Parquet file (also pushed via XCom).

    Raises
    ------
    FileNotFoundError
        If the configured CSV source does not exist.
    RuntimeError
        If the CSV is empty (zero rows).
    """
    ti = context["ti"]
    execution_date: str = context["ds"]  # YYYY-MM-DD

    logger.info("Ingestion started for execution date %s", execution_date)
    logger.info("Reading raw CSV from %s", DATA_SOURCE_PATH)

    source = Path(DATA_SOURCE_PATH)
    if not source.exists():
        raise FileNotFoundError(
            f"Raw data file not found at {DATA_SOURCE_PATH}. "
            "Ensure the data-source path Airflow Variable is configured."
        )

    try:
        df = pd.read_csv(source)
    except pd.errors.ParserError as exc:
        raise RuntimeError(
            f"Failed to parse CSV at {DATA_SOURCE_PATH}: {exc}"
        ) from exc

    if df.empty:
        raise RuntimeError(
            f"Raw CSV at {DATA_SOURCE_PATH} contains zero rows. "
            "Aborting pipeline to prevent downstream errors."
        )

    logger.info(
        "Ingested %d rows and %d columns from %s",
        len(df),
        len(df.columns),
        DATA_SOURCE_PATH,
    )

    # Persist as dated Parquet snapshot
    staging_dir = Path(STAGING_DIR)
    staging_dir.mkdir(parents=True, exist_ok=True)

    output_path = str(staging_dir / f"products_{execution_date}.parquet")
    df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info("Staged Parquet snapshot written to %s", output_path)

    # Push file path (not the DataFrame) to XCom
    ti.xcom_push(key="staged_parquet_path", value=output_path)
    return output_path


def _validate_schema(**context: Any) -> str:
    """Validate the schema of the staged Parquet file against the expected
    column definitions for the EcoPackAI products dataset.

    Checks performed:
    * Presence of all required columns.
    * Data-type compatibility for each column.
    * No null values in required columns.
    * Positive-value constraints on dimension and weight columns.

    Parameters
    ----------
    **context : Any
        Airflow task-instance context.

    Returns
    -------
    str
        The validated file path (passed through for downstream tasks).

    Raises
    ------
    ValueError
        If any schema check fails.
    FileNotFoundError
        If the upstream staged file is missing.
    """
    ti = context["ti"]

    staged_path: str = ti.xcom_pull(
        task_ids="ingest_raw_data", key="staged_parquet_path"
    )
    if not staged_path or not Path(staged_path).exists():
        raise FileNotFoundError(
            f"Staged Parquet file not found at '{staged_path}'. "
            "Ensure the ingest_raw_data task completed successfully."
        )

    logger.info("Validating schema for %s", staged_path)
    df = pd.read_parquet(staged_path)

    # --- Column presence ---
    missing_cols = set(EXPECTED_SCHEMA.keys()) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {sorted(missing_cols)}"
        )

    # --- Dtype compatibility ---
    dtype_errors: list[str] = []
    for col, expected_dtype in EXPECTED_SCHEMA.items():
        actual_dtype = str(df[col].dtype)
        if actual_dtype != expected_dtype:
            dtype_errors.append(
                f"Column '{col}': expected {expected_dtype}, got {actual_dtype}"
            )
    if dtype_errors:
        raise ValueError(
            "Schema dtype mismatches:\n  " + "\n  ".join(dtype_errors)
        )

    # --- Null checks ---
    null_counts = df[list(EXPECTED_SCHEMA.keys())].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        raise ValueError(
            f"Null values detected in required columns:\n{cols_with_nulls}"
        )

    # --- Positive-value constraints ---
    positive_cols = ["length_cm", "width_cm", "height_cm", "weight_g"]
    for col in positive_cols:
        if (df[col] <= 0).any():
            bad_count = int((df[col] <= 0).sum())
            raise ValueError(
                f"Column '{col}' has {bad_count} non-positive values. "
                "All dimensions and weight must be > 0."
            )

    logger.info("Schema validation passed for %d rows.", len(df))

    # Forward the same path to the next task
    ti.xcom_push(key="validated_parquet_path", value=staged_path)
    return staged_path


def _run_feature_engineering(**context: Any) -> str:
    """Apply the EcoPackAI feature-engineering pipeline to the validated data.

    Derived features:
    * ``volume_cm3`` – product bounding-box volume.
    * ``density_g_per_cm3`` – weight / volume.
    * ``surface_area_cm2`` – outer surface area of the bounding box.
    * ``aspect_ratio`` – max(dimension) / min(dimension).
    * ``material_encoded`` – ordinal encoding of ``material_type``.

    The enriched DataFrame is written to the feature-store directory as Parquet.

    Parameters
    ----------
    **context : Any
        Airflow task-instance context.

    Returns
    -------
    str
        Absolute path to the feature-store Parquet file.

    Raises
    ------
    FileNotFoundError
        If the upstream validated file is missing.
    RuntimeError
        If feature computation fails.
    """
    ti = context["ti"]
    execution_date: str = context["ds"]

    validated_path: str = ti.xcom_pull(
        task_ids="validate_schema", key="validated_parquet_path"
    )
    if not validated_path or not Path(validated_path).exists():
        raise FileNotFoundError(
            f"Validated Parquet file not found at '{validated_path}'. "
            "Ensure validate_schema completed successfully."
        )

    logger.info("Running feature engineering on %s", validated_path)
    df = pd.read_parquet(validated_path)

    try:
        # Volume (cm³)
        df["volume_cm3"] = df["length_cm"] * df["width_cm"] * df["height_cm"]

        # Density (g / cm³)
        df["density_g_per_cm3"] = df["weight_g"] / df["volume_cm3"]

        # Surface area (cm²)
        df["surface_area_cm2"] = 2.0 * (
            df["length_cm"] * df["width_cm"]
            + df["width_cm"] * df["height_cm"]
            + df["height_cm"] * df["length_cm"]
        )

        # Aspect ratio
        dim_cols = df[["length_cm", "width_cm", "height_cm"]]
        df["aspect_ratio"] = dim_cols.max(axis=1) / dim_cols.min(axis=1)

        # Ordinal encoding for material_type
        material_order = {
            "paper": 0,
            "cardboard": 1,
            "biodegradable_plastic": 2,
            "recycled_plastic": 3,
            "plastic": 4,
            "foam": 5,
        }
        df["material_encoded"] = (
            df["material_type"]
            .str.lower()
            .map(material_order)
            .fillna(-1)
            .astype(int)
        )

    except Exception as exc:
        raise RuntimeError(
            f"Feature engineering failed: {exc}"
        ) from exc

    # Persist to feature store
    feature_dir = Path(FEATURE_STORE_DIR)
    feature_dir.mkdir(parents=True, exist_ok=True)

    output_path = str(feature_dir / f"features_{execution_date}.parquet")
    df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info(
        "Feature-engineered data (%d rows, %d cols) written to %s",
        len(df),
        len(df.columns),
        output_path,
    )

    ti.xcom_push(key="features_parquet_path", value=output_path)
    return output_path


def _upload_to_postgres(**context: Any) -> None:
    """Load feature-engineered data into the ``products`` table in PostgreSQL.

    Uses SQLAlchemy ``to_sql`` with ``method='multi'`` and chunked inserts for
    efficient bulk loading.  Rows are **appended** – the table is never
    truncated so historical snapshots are preserved.

    Parameters
    ----------
    **context : Any
        Airflow task-instance context.

    Raises
    ------
    FileNotFoundError
        If the upstream feature file is missing.
    RuntimeError
        If the database write fails.
    """
    ti = context["ti"]

    features_path: str = ti.xcom_pull(
        task_ids="run_feature_engineering", key="features_parquet_path"
    )
    if not features_path or not Path(features_path).exists():
        raise FileNotFoundError(
            f"Features Parquet file not found at '{features_path}'. "
            "Ensure run_feature_engineering completed successfully."
        )

    logger.info("Uploading features from %s to PostgreSQL", features_path)
    df = pd.read_parquet(features_path)

    engine: Engine | None = None
    try:
        engine = create_engine(
            POSTGRES_CONN_STR,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )

        rows_written = df.to_sql(
            name="products",
            con=engine,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=500,
        )

        logger.info(
            "Successfully uploaded %d rows to 'products' table.",
            len(df),
        )

    except Exception as exc:
        raise RuntimeError(
            f"PostgreSQL upload failed: {exc}"
        ) from exc
    finally:
        if engine is not None:
            engine.dispose()
            logger.info("Database engine disposed.")


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

default_args: Dict[str, Any] = {
    "owner": "ecopackai",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
    "email": ["mlops@ecopackai.com"],
    "depends_on_past": False,
}

with DAG(
    dag_id="ecopackai_data_pipeline",
    description="EcoPackAI daily data preprocessing and ingestion pipeline",
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["ecopackai", "data-pipeline", "etl"],
    doc_md=__doc__,
) as dag:

    ingest_raw_data = PythonOperator(
        task_id="ingest_raw_data",
        python_callable=_ingest_raw_data,
        doc_md="Read raw CSV, persist as dated Parquet, push path via XCom.",
    )

    validate_schema = PythonOperator(
        task_id="validate_schema",
        python_callable=_validate_schema,
        doc_md="Validate column presence, dtypes, nulls, and value ranges.",
    )

    run_feature_engineering = PythonOperator(
        task_id="run_feature_engineering",
        python_callable=_run_feature_engineering,
        doc_md="Derive volume, density, surface area, aspect ratio, encodings.",
    )

    upload_to_postgres = PythonOperator(
        task_id="upload_to_postgres",
        python_callable=_upload_to_postgres,
        doc_md="Bulk-insert feature data into PostgreSQL products table.",
    )

    # Linear dependency chain
    ingest_raw_data >> validate_schema >> run_feature_engineering >> upload_to_postgres
