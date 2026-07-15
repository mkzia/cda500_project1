#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
import psycopg2
from psycopg2.extras import DictCursor
import pandas as pd
from pyspark.sql import SparkSession

# Import source of truth business constraints from your shared architecture module
from jobs.ml_features import DemandThresholds, class_from_ride_count, calculate_penalty

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ml_feedback_loop")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real-now",
        required=True,
        help="ISO timestamp string of the current processed hour",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    current_hour = pd.Timestamp(args.real_now).floor("h")
    logger.info(
        f"Starting feedback loop settlement evaluating target context date: {current_hour}"
    )

    conn_string = (
        "host=operational_cache_postgres "
        "dbname=operational_cache "
        "user=cache_user "
        "password=cache_password "
        "port=5432"
    )

    connection = psycopg2.connect(conn_string)
    cursor = connection.cursor(cursor_factory=DictCursor)

    try:
        # 1. Fetch only pending records whose targeted simulation hour has arrived
        cursor.execute(
            """
            SELECT
                pickup_location_id,
                prediction_hour_ts,
                predicted_class,
                baseline_predicted_class,
                threshold_q25,
                threshold_q75
            FROM hourly_demand_predictions
            WHERE actual_class IS NULL
              AND prediction_hour_ts <= %s
            ORDER BY prediction_hour_ts ASC
        """,
            (current_hour.to_pydatetime(),),
        )

        pending_rows = cursor.fetchall()

        if not pending_rows:
            logger.info(
                "No pending data entries found requiring actuals backfill evaluation."
            )
            return

        logger.info(
            f"Found {len(pending_rows)} historical records awaiting metric updates. Initializing Spark layer..."
        )

        # 2. Spin up local PySpark cluster environment matching master configuration profiles
        spark = (
            SparkSession.builder.appName("MLOpsFeedbackEvaluator")
            .config("spark.sql.catalog.nyc", "org.apache.iceberg.spark.SparkCatalog")
            .config("spark.sql.catalog.nyc.type", "hadoop")
            .config("spark.sql.catalog.nyc.warehouse", "/opt/airflow/data/warehouse")
            .config(
                "spark.sql.extensions",
                "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
            )
            .getOrCreate()
        )

        # 3. Iterate through un-evaluated records and fetch true values from the Iceberg Gold layer
        for row in pending_rows:
            loc_id = int(row["pickup_location_id"])
            pred_hour = row["prediction_hour_ts"]
            pred_class = int(row["predicted_class"])
            baseline_class = int(row["baseline_predicted_class"])

            # Explicitly mapping names from DictCursor columns directly to prevent tuple mismatch
            thresholds = DemandThresholds(
                q25=float(row["threshold_q25"]), q75=float(row["threshold_q75"])
            )

            # Format timestamp string safely for the Iceberg SQL engine context parser
            pred_hour_str = pd.Timestamp(pred_hour).strftime("%Y-%m-%d %H:%M:%S")

            iceberg_actual_df = spark.sql(f"""
                SELECT ride_count FROM nyc.gold.hourly_ml_observations
                WHERE pickup_location_id = {loc_id} AND pickup_hour_ts = '{pred_hour_str}'
            """).toPandas()

            if iceberg_actual_df.empty:
                logger.warning(
                    f"Ground truth data not yet settled in Iceberg lake for location={loc_id} at {pred_hour_str}. Skipping."
                )
                continue

            actual_count = float(iceberg_actual_df.iloc[0]["ride_count"])

            # Dynamically determine the true class using synchronized model metadata rules
            actual_cls = class_from_ride_count(actual_count, thresholds)

            # Evaluate operational penalty tracking metrics using the unified shared function
            model_penalty = calculate_penalty(
                actual_class=actual_cls, predicted_class=pred_class
            )
            baseline_penalty = calculate_penalty(
                actual_class=actual_cls, predicted_class=baseline_class
            )

            # 4. Update the row atomically inside PostgreSQL
            cursor.execute(
                """
                UPDATE hourly_demand_predictions
                SET
                    actual_class = %s,
                    actual_ride_count = %s,
                    model_penalty = %s,
                    baseline_penalty = %s
                WHERE pickup_location_id = %s AND prediction_hour_ts = %s
            """,
                (
                    actual_cls,
                    actual_count,
                    model_penalty,
                    baseline_penalty,
                    loc_id,
                    pred_hour,
                ),
            )

            # Commit immediately per record loop to clear transaction states incrementally
            connection.commit()
            logger.info(
                f"✅ Settled row in Postgres: Location {loc_id} @ {pred_hour_str} -> Actual Count: {actual_count:.0f} (Model Penalty: {model_penalty})"
            )

    except Exception as e:
        connection.rollback()
        logger.error(f"❌ Feedback loop calculation aborted due to exception: {str(e)}")
        raise
    finally:
        cursor.close()
        connection.close()
        logger.info("PostgreSQL transaction network space safely closed down.")


if __name__ == "__main__":
    main()
