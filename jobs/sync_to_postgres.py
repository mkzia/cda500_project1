#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from datetime import timezone
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
from pyiceberg.expressions import And, EqualTo
from pyiceberg.table import StaticTable

from jobs.ml_features import shifted_simulation_time
from jobs.ml_utils import get_models_dir, load_config, read_json

# Added evaluation contract fields to match the updated Iceberg schema
POSTGRES_COLUMNS = [
    "pickup_location_id",
    "prediction_hour_ts",
    "scored_at_ts",
    "model_name",
    "model_version",
    "predicted_class",
    "predicted_label",
    "prob_class_0",
    "prob_class_1",
    "prob_class_2",
    "baseline_predicted_class",
    "mlflow_run_id",
    "feature_window_end_ts",
    "threshold_q25",
    "threshold_q75",
    "penalty_version",
    "actual_class",
    "actual_ride_count",
    "model_penalty",
    "baseline_penalty",
]


def as_utc_datetime(value: pd.Timestamp):
    """
    Convert a Pandas timestamp to a timezone-aware UTC Python datetime.
    """
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def sync_predictions_to_postgres(
    config_path: str | None,
    real_now: str | None,
    months_back: int,
) -> None:
    cfg = load_config(config_path)

    # 1. Recreate the simulation timestamp used by the scoring job.
    feature_window_end_ts = shifted_simulation_time(
        real_now=real_now,
        months_back=months_back,
    )
    prediction_hour_ts = feature_window_end_ts + pd.Timedelta(hours=1)
    prediction_hour_utc = as_utc_datetime(prediction_hour_ts)

    # Load the location used by the champion model.
    models_dir = get_models_dir(cfg)
    metadata = read_json(models_dir / "champion_metadata.json")
    pickup_location_id = int(metadata["pickup_location_id"])

    # 2. Load the local Iceberg table.
    warehouse_path = Path(cfg["catalog"]["warehouse_path"])
    table_path = warehouse_path / "predictions" / "hourly_demand_predictions"
    metadata_dir = table_path / "metadata"

    if not table_path.exists() or not metadata_dir.exists():
        raise FileNotFoundError(
            f"Iceberg table components missing at path: {table_path}"
        )

    print(f"🕵️ Loading Iceberg table from path: {table_path}")
    iceberg_table = StaticTable.from_metadata(str(table_path))

    print(
        f"🕵️ Scanning Iceberg for location={pickup_location_id}, "
        f"prediction_hour_ts={prediction_hour_utc.isoformat()}"
    )

    row_filter = And(
        EqualTo("pickup_location_id", pickup_location_id),
        EqualTo("prediction_hour_ts", prediction_hour_utc),
    )

    arrow_table = iceberg_table.scan(row_filter=row_filter).to_arrow()

    if arrow_table.num_rows == 0:
        print(f"⚠️ No matching prediction row found in Iceberg.")
        return

    pandas_record = arrow_table.to_pandas()

    # Ensure timestamps remain localized to UTC for standard Postgres compatibility
    timestamp_columns = ["prediction_hour_ts", "scored_at_ts", "feature_window_end_ts"]
    for column in timestamp_columns:
        if column in pandas_record.columns:
            pandas_record[column] = pd.to_datetime(pandas_record[column], utc=True)

    missing_columns = [
        col for col in POSTGRES_COLUMNS if col not in pandas_record.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Iceberg record is missing required columns: {missing_columns}"
        )

    # Enforce precise column ordering match
    pandas_record = pandas_record[POSTGRES_COLUMNS]

    if len(pandas_record) > 1:
        raise ValueError(f"Expected 1 row, got {len(pandas_record)}")

    # 3. Synchronize the row to PostgreSQL (Atomic Multi-Process Safe Upsert)
    conn_string = "host=operational_cache_postgres dbname=operational_cache user=cache_user password=cache_password port=5432"

    connection = psycopg2.connect(conn_string)
    cursor = connection.cursor()

    try:
        # Added threshold_q25, threshold_q75, and penalty_version tracking properties to DDL contract
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hourly_demand_predictions (
                pickup_location_id BIGINT,
                prediction_hour_ts TIMESTAMPTZ,
                scored_at_ts TIMESTAMPTZ,
                model_name VARCHAR,
                model_version VARCHAR,
                predicted_class BIGINT,
                predicted_label VARCHAR,
                prob_class_0 DOUBLE PRECISION,
                prob_class_1 DOUBLE PRECISION,
                prob_class_2 DOUBLE PRECISION,
                baseline_predicted_class BIGINT,
                mlflow_run_id VARCHAR,
                feature_window_end_ts TIMESTAMPTZ,
                threshold_q25 DOUBLE PRECISION,
                threshold_q75 DOUBLE PRECISION,
                penalty_version VARCHAR,
                actual_class BIGINT,
                actual_ride_count BIGINT,
                model_penalty DOUBLE PRECISION,
                baseline_penalty DOUBLE PRECISION,
                PRIMARY KEY (pickup_location_id, prediction_hour_ts)
            );
            """)

        # ---------------------------------------------------------
        # SAFE TYPE CONVERSION & SANITIZATION
        # ---------------------------------------------------------
        raw_dict = pandas_record.iloc[0].to_dict()
        sanitized_values = []

        for col in POSTGRES_COLUMNS:
            val = raw_dict[col]

            # Convert pandas/numpy types to native python primitives for psycopg2
            if isinstance(val, pd.Timestamp):
                val = val.to_pydatetime()
            elif hasattr(val, "item"):  # Unpacks np.int64, np.float64, etc.
                val = val.item()

            # Clear NaN floating structures out of integer columns
            if pd.isna(val):
                val = None

            sanitized_values.append(val)

        row_data = tuple(sanitized_values)

        # ---------------------------------------------------------
        # Expanded the insert statement block and upsert conflict resolutions to handle metadata fields
        upsert_query = """
            INSERT INTO hourly_demand_predictions (
                pickup_location_id, prediction_hour_ts, scored_at_ts, model_name, model_version,
                predicted_class, predicted_label, prob_class_0, prob_class_1, prob_class_2,
                baseline_predicted_class, mlflow_run_id, feature_window_end_ts,
                threshold_q25, threshold_q75, penalty_version,
                actual_class, actual_ride_count, model_penalty, baseline_penalty
            ) VALUES %s
            ON CONFLICT (pickup_location_id, prediction_hour_ts) DO UPDATE SET
                scored_at_ts = EXCLUDED.scored_at_ts,
                model_name = EXCLUDED.model_name,
                model_version = EXCLUDED.model_version,
                predicted_class = EXCLUDED.predicted_class,
                predicted_label = EXCLUDED.predicted_label,
                prob_class_0 = EXCLUDED.prob_class_0,
                prob_class_1 = EXCLUDED.prob_class_1,
                prob_class_2 = EXCLUDED.prob_class_2,
                baseline_predicted_class = EXCLUDED.baseline_predicted_class,
                mlflow_run_id = EXCLUDED.mlflow_run_id,
                feature_window_end_ts = EXCLUDED.feature_window_end_ts,
                threshold_q25 = EXCLUDED.threshold_q25,
                threshold_q75 = EXCLUDED.threshold_q75,
                penalty_version = EXCLUDED.penalty_version,
                actual_class = EXCLUDED.actual_class,
                actual_ride_count = EXCLUDED.actual_ride_count,
                model_penalty = EXCLUDED.model_penalty,
                baseline_penalty = EXCLUDED.baseline_penalty;
        """

        print(
            "🚀 Merging contract-safe prediction row into Postgres operational cache via safe ON CONFLICT block."
        )
        execute_values(cursor, upsert_query, [row_data])
        connection.commit()

    except Exception as e:
        connection.rollback()
        print(f"❌ DATABASE ERROR: {str(e)}", file=sys.stderr)
        raise
    finally:
        cursor.close()
        connection.close()

    print("GFV 🟢 Prediction successfully synchronized from Iceberg to PostgreSQL.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--real-now", default=None)
    parser.add_argument("--months-back", type=int, default=2)

    args = parser.parse_args()
    sync_predictions_to_postgres(
        config_path=args.config,
        real_now=args.real_now,
        months_back=args.months_back,
    )


if __name__ == "__main__":
    main()
