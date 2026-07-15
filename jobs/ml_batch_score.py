from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from jobs.ml_features import (
    LABEL_MAP,
    COST_MATRIX,
    DemandThresholds,
    build_one_prediction_feature_row,
    calculate_penalty,
    load_location_history,
    shifted_simulation_time,
)
from jobs.ml_utils import (
    create_spark_session,
    get_gold_ml_table,
    get_models_dir,
    load_config,
    read_json,
)


def cost_sensitive_predict_single(model: Any, X: pd.DataFrame) -> int:
    """Chooses the demand class category that minimizes total expected asymmetric cost."""
    probabilities = model.predict_proba(X)
    full_probabilities = np.zeros((1, 3), dtype=float)

    for column_index, class_value in enumerate(model.classes_):
        full_probabilities[0, int(class_value)] = probabilities[0, column_index]

    # Expected Cost Matrix = Class Probabilities Vector @ Business Cost Loss Matrix Array
    expected_costs = full_probabilities @ COST_MATRIX
    return int(expected_costs.argmin(axis=1)[0])


def _probabilities(
    model: Any, X: pd.DataFrame
) -> tuple[float | None, float | None, float | None]:
    if not hasattr(model, "predict_proba"):
        return None, None, None
    probs = model.predict_proba(X)[0]
    classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1, 2]
    prob_map = {int(c): float(p) for c, p in zip(classes, probs)}
    return prob_map.get(0), prob_map.get(1), prob_map.get(2)


def ensure_predictions_table_exists(spark: SparkSession, table_name: str) -> None:
    """Dynamically establishes the Iceberg target table including evaluation metadata contract schema rows."""
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            pickup_location_id INT,
            prediction_hour_ts TIMESTAMP,
            scored_at_ts TIMESTAMP,
            model_name STRING,
            model_version STRING,
            predicted_class INT,
            predicted_label STRING,
            prob_class_0 DOUBLE,
            prob_class_1 DOUBLE,
            prob_class_2 DOUBLE,
            baseline_predicted_class INT,
            mlflow_run_id STRING,
            feature_window_end_ts TIMESTAMP,
            threshold_q25 DOUBLE,
            threshold_q75 DOUBLE,
            penalty_version STRING,
            actual_class INT,
            actual_ride_count INT,
            model_penalty DOUBLE,
            baseline_penalty DOUBLE
        ) USING iceberg
        PARTITIONED BY (pickup_location_id)
        TBLPROPERTIES ('format-version'='2')
    """)


def run_batch_score(
    config_path: str | None, real_now: str | None, months_back: int
) -> None:
    cfg = load_config(config_path)
    spark = create_spark_session(cfg, app_name="MLBatchDemandScoreIceberg")
    gold_table = get_gold_ml_table(cfg)

    catalog = cfg.get("catalog", {}).get("spark_catalog_name", "nyc")
    predictions_namespace = f"{catalog}.predictions"
    predictions_table = f"{predictions_namespace}.hourly_demand_predictions"

    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {predictions_namespace}")
    ensure_predictions_table_exists(spark, predictions_table)

    models_dir = get_models_dir(cfg)
    model_path = models_dir / "champion_model.pkl"
    metadata_path = models_dir / "champion_metadata.json"

    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing champion artifacts. Expected {model_path} and {metadata_path}."
        )

    model = joblib.load(model_path)
    meta = read_json(metadata_path)

    pickup_location_id = int(meta["pickup_location_id"])
    thresholds = DemandThresholds(q25=float(meta["q25"]), q75=float(meta["q75"]))
    feature_columns = list(meta["features"])

    feature_window_end_ts = shifted_simulation_time(
        real_now=real_now, months_back=months_back
    )
    prediction_hour_ts = feature_window_end_ts + pd.Timedelta(hours=1)

    print(f"Real run time supplied: {real_now or '<current wall-clock time>'}")
    print(f"Simulated feature window end: {feature_window_end_ts}")
    print(f"Prediction target hour (t+1): {prediction_hour_ts}")

    spark_df = spark.table(gold_table)

    # Named argument renamed from 'end_exclusive_ts' to 'end_ts' to avoid TypeError crashes
    history_pdf = load_location_history(
        spark_df,
        pickup_location_id=pickup_location_id,
        end_ts=prediction_hour_ts,
    )

    X_live, row_meta = build_one_prediction_feature_row(
        history_pdf=history_pdf,
        pickup_location_id=pickup_location_id,
        feature_window_end_ts=feature_window_end_ts,
        thresholds=thresholds,
        feature_columns=feature_columns,
    )

    # Replaced default scikit-learn maximum probability boundary choices with cost-sensitive ones
    predicted_class = cost_sensitive_predict_single(model, X_live)
    p0, p1, p2 = _probabilities(model, X_live)

    actual_class = row_meta["actual_class"]
    actual_ride_count = row_meta["actual_ride_count"]
    baseline_class = int(row_meta["baseline_predicted_class"])

    model_penalty = calculate_penalty(actual_class, predicted_class)
    baseline_penalty = calculate_penalty(actual_class, baseline_class)

    schema = StructType(
        [
            StructField("pickup_location_id", IntegerType(), True),
            StructField("prediction_hour_ts", TimestampType(), True),
            StructField("scored_at_ts", TimestampType(), True),
            StructField("model_name", StringType(), True),
            StructField("model_version", StringType(), True),
            StructField("predicted_class", IntegerType(), True),
            StructField("predicted_label", StringType(), True),
            StructField("prob_class_0", DoubleType(), True),
            StructField("prob_class_1", DoubleType(), True),
            StructField("prob_class_2", DoubleType(), True),
            StructField("baseline_predicted_class", IntegerType(), True),
            StructField("mlflow_run_id", StringType(), True),
            StructField("feature_window_end_ts", TimestampType(), True),
            StructField("threshold_q25", DoubleType(), True),
            StructField("threshold_q75", DoubleType(), True),
            StructField("penalty_version", StringType(), True),
            StructField("actual_class", IntegerType(), True),
            StructField("actual_ride_count", IntegerType(), True),
            StructField("model_penalty", DoubleType(), True),
            StructField("baseline_penalty", DoubleType(), True),
        ]
    )

    # Contract parameter columns populated explicitly into payload log dictionaries
    record_dict = {
        "pickup_location_id": pickup_location_id,
        "prediction_hour_ts": prediction_hour_ts.to_pydatetime(),
        "scored_at_ts": datetime.utcnow(),
        "model_name": str(meta.get("model_name")),
        "model_version": str(meta.get("model_version", "local-v2-cost-sensitive")),
        "predicted_class": predicted_class,
        "predicted_label": str(LABEL_MAP[predicted_class]),
        "prob_class_0": float(p0) if p0 is not None else None,
        "prob_class_1": float(p1) if p1 is not None else None,
        "prob_class_2": float(p2) if p2 is not None else None,
        "baseline_predicted_class": baseline_class,
        "mlflow_run_id": str(meta.get("mlflow_run_id")),
        "feature_window_end_ts": feature_window_end_ts.to_pydatetime(),
        "threshold_q25": float(thresholds.q25),
        "threshold_q75": float(thresholds.q75),
        "penalty_version": str(meta.get("penalty_version", "asymmetric-v1")),
        "actual_class": int(actual_class) if not pd.isna(actual_class) else None,
        "actual_ride_count": (
            int(actual_ride_count) if not pd.isna(actual_ride_count) else None
        ),
        "model_penalty": float(model_penalty) if not pd.isna(model_penalty) else None,
        "baseline_penalty": (
            float(baseline_penalty) if not pd.isna(baseline_penalty) else None
        ),
    }

    pandas_bridge_df = pd.DataFrame([record_dict])
    spark_record_df = spark.createDataFrame(pandas_bridge_df, schema=schema)

    print(
        f"🧹 Executing idempotent purge for Location {pickup_location_id} at {prediction_hour_ts}..."
    )
    spark.sql(f"""
        DELETE FROM {predictions_table}
        WHERE pickup_location_id = {pickup_location_id}
          AND prediction_hour_ts = CAST('{prediction_hour_ts.strftime("%Y-%m-%d %H:%M:%S")}' AS TIMESTAMP)
    """)

    print(
        f"🚀 Pushing next-hour batch prediction into warehouse layer: {predictions_table}"
    )
    spark_record_df.writeTo(predictions_table).append()
    print("🟢 Prediction successfully committed to Iceberg storage container view.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--real-now", default=None, help="Real run timestamp, e.g. 2026-07-08T17:00:00"
    )
    parser.add_argument("--months-back", type=int, default=2)
    args = parser.parse_args()
    run_batch_score(
        config_path=args.config, real_now=args.real_now, months_back=args.months_back
    )


if __name__ == "__main__":
    main()
