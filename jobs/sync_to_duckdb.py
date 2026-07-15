from __future__ import annotations

import argparse
from datetime import timezone
from pathlib import Path

import duckdb
import pandas as pd
from pyiceberg.expressions import And, EqualTo
from pyiceberg.table import StaticTable

from jobs.ml_features import shifted_simulation_time
from jobs.ml_utils import get_models_dir, load_config, read_json

DUCKDB_COLUMNS = [
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
    "actual_class",
    "actual_ride_count",
    "model_penalty",
    "baseline_penalty",
]


def as_utc_datetime(value: pd.Timestamp):
    """
    Convert a Pandas timestamp to a timezone-aware UTC Python datetime.

    Naive timestamps are interpreted as UTC without changing the clock time.
    Timezone-aware timestamps are converted to UTC.
    """
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp.to_pydatetime()


def sync_predictions_to_duckdb(
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

    if not table_path.exists():
        raise FileNotFoundError(f"Iceberg table directory does not exist: {table_path}")

    if not metadata_dir.exists():
        raise FileNotFoundError(
            f"Iceberg metadata directory does not exist: {metadata_dir}"
        )

    print(f"🕵️ Loading Iceberg table from path: {table_path}")

    iceberg_table = StaticTable.from_metadata(str(table_path))

    print(
        "🕵️ Scanning Iceberg for "
        f"location={pickup_location_id}, "
        f"prediction_hour_ts={prediction_hour_utc.isoformat()}"
    )

    row_filter = And(
        EqualTo(
            "pickup_location_id",
            pickup_location_id,
        ),
        EqualTo(
            "prediction_hour_ts",
            prediction_hour_utc,
        ),
    )

    arrow_table = iceberg_table.scan(
        row_filter=row_filter,
    ).to_arrow()

    if arrow_table.num_rows == 0:
        print(
            "⚠️ No matching prediction row found in Iceberg for "
            f"location={pickup_location_id}, "
            f"prediction_hour_ts={prediction_hour_utc.isoformat()}."
        )

        diagnostic_rows = iceberg_table.scan(
            row_filter=EqualTo(
                "pickup_location_id",
                pickup_location_id,
            ),
            selected_fields=(
                "pickup_location_id",
                "prediction_hour_ts",
            ),
            limit=10,
        ).to_arrow()

        print("Diagnostic Iceberg rows for this location:")
        print(diagnostic_rows.to_pandas().to_string(index=False))
        return

    pandas_record = arrow_table.to_pandas()

    # Ensure timestamps are compatible with DuckDB TIMESTAMP columns.
    timestamp_columns = [
        "prediction_hour_ts",
        "scored_at_ts",
        "feature_window_end_ts",
    ]

    for column in timestamp_columns:
        if column not in pandas_record.columns:
            continue

        pandas_record[column] = pd.to_datetime(
            pandas_record[column],
            utc=True,
        ).dt.tz_localize(None)

    missing_columns = [
        column for column in DUCKDB_COLUMNS if column not in pandas_record.columns
    ]

    if missing_columns:
        raise ValueError(
            "Iceberg prediction record is missing required columns: "
            f"{missing_columns}"
        )

    # Preserve the exact DuckDB table column order.
    pandas_record = pandas_record[DUCKDB_COLUMNS]

    # There should be only one row for the idempotent table key.
    if len(pandas_record) > 1:
        raise ValueError(
            "Expected one prediction row, but Iceberg returned "
            f"{len(pandas_record)} rows for "
            f"location={pickup_location_id}, "
            f"prediction_hour_ts={prediction_hour_utc.isoformat()}."
        )

    # 3. Synchronize the row to DuckDB.
    duckdb_path = Path("/opt/airflow/data/predictions.duckdb")
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect(str(duckdb_path))

    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS hourly_demand_predictions (
                pickup_location_id INTEGER,
                prediction_hour_ts TIMESTAMP,
                scored_at_ts TIMESTAMP,
                model_name VARCHAR,
                model_version VARCHAR,
                predicted_class INTEGER,
                predicted_label VARCHAR,
                prob_class_0 DOUBLE,
                prob_class_1 DOUBLE,
                prob_class_2 DOUBLE,
                baseline_predicted_class INTEGER,
                mlflow_run_id VARCHAR,
                feature_window_end_ts TIMESTAMP,
                actual_class INTEGER,
                actual_ride_count INTEGER,
                model_penalty DOUBLE,
                baseline_penalty DOUBLE,
                PRIMARY KEY (
                    pickup_location_id,
                    prediction_hour_ts
                )
            )
            """)

        connection.begin()

        try:
            print("🚀 Merging prediction row into DuckDB serving cache.")

            connection.execute(
                "INSERT OR REPLACE INTO hourly_demand_predictions SELECT * FROM pandas_record"
            )

            connection.commit()

        except Exception:
            connection.rollback()
            raise

    finally:
        connection.close()

    print("🟢 Prediction successfully synchronized " "from Iceberg to DuckDB.")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=None,
    )
    parser.add_argument(
        "--real-now",
        default=None,
    )
    parser.add_argument(
        "--months-back",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    sync_predictions_to_duckdb(
        config_path=args.config,
        real_now=args.real_now,
        months_back=args.months_back,
    )


if __name__ == "__main__":
    main()
