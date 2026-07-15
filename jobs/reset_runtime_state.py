#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
import psycopg2

from jobs.ml_utils import create_spark_session, load_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("reset_runtime_state")


def reset_environment(config_path: str | None = None) -> None:
    """Drops the Iceberg table completely from disk and clears the PostgreSQL cache table."""
    cfg = load_config(config_path)

    # 1. Resolve Target Table Identifiers
    catalog = cfg.get("catalog", {}).get("spark_catalog_name", "nyc")
    predictions_table = f"{catalog}.predictions.hourly_demand_predictions"

    logger.info("Starting absolute state and table file purge across the platform...")

    # 2. Drop Apache Iceberg Data Table & Delete Disk Objects Completely
    logger.info(
        f"Connecting to PySpark cluster to destroy Iceberg table: {predictions_table}"
    )
    try:
        spark = create_spark_session(cfg, app_name="MLStatePurgeUtility")

        if spark.catalog.tableExists(predictions_table):
            # Migrated from 'DELETE FROM' (which keeps files) to 'DROP TABLE' (which removes table files completely)
            logger.info(
                f"💥 Table found. Executing structural DROP TABLE on {predictions_table}..."
            )
            spark.sql(f"DROP TABLE IF EXISTS {predictions_table}")
            logger.info(
                "🟢 Apache Iceberg table schema and raw data files completely purged from warehouse."
            )
        else:
            logger.warning(
                f"Iceberg target table {predictions_table} does not exist. Skipping."
            )

    except Exception as e:
        logger.error(f"❌ Iceberg dropping sequence aborted: {str(e)}")
        raise

    # 3. Purge PostgreSQL Operational Cache Table
    logger.info("Connecting to operational network service cache layer via Psycopg2...")
    pg_conn_string = (
        "host=operational_cache_postgres "
        "dbname=operational_cache "
        "user=cache_user "
        "password=cache_password "
        "port=5432"
    )

    try:
        connection = psycopg2.connect(pg_conn_string)
        cursor = connection.cursor()

        # We DROP the table entirely here as well to match the absolute reset intent
        logger.info(
            "🧹 Executing structural DROP TABLE on PostgreSQL: hourly_demand_predictions..."
        )
        cursor.execute("DROP TABLE IF EXISTS hourly_demand_predictions;")

        connection.commit()
        logger.info("🟢 PostgreSQL operational cache table safely destroyed.")

    except Exception as e:
        if "connection" in locals():
            connection.rollback()
        logger.error(f"❌ PostgreSQL purging phase encountered an exception: {str(e)}")
        raise
    finally:
        if "cursor" in locals():
            cursor.close()
        if "connection" in locals():
            connection.close()

    logger.info(
        "🚀 Done! Table metadata schemas and backing warehouse records are completely gone."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=None, help="Path to config/local.yml mapping configurations"
    )
    args = parser.parse_args()

    try:
        reset_environment(config_path=args.config)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
