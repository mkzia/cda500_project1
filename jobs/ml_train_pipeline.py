#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

# Configure clean logging output
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ml_train_pipeline")

# Resolve PROJECT_ROOT pathing mechanics
PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "notebooks" or PROJECT_ROOT.name == "jobs":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from jobs.ml_utils import (
    load_config,
    create_spark_session,
    get_gold_ml_table,
    get_models_dir,
)
from jobs.ml_features import (
    select_highest_volume_location,
    load_location_history,
    prepare_model_frames,
)
from jobs.ml_experiments import (
    run_six_experiments,
    select_champion,
    log_test_metrics_to_champion_run,
    export_champion,
)


def run_training_pipeline(
    config_path: str | None = None, optuna_trials: int = 20
) -> None:
    """Executes the complete production training lifecycle, optimizing for asymmetric business costs."""
    # 1. Initialize configurations and cluster sessions
    config_file = (
        Path(config_path) if config_path else PROJECT_ROOT / "config" / "local.yml"
    )
    cfg = load_config(config_file)

    spark = create_spark_session(cfg, app_name="MLOpsDemandTrainingPipeline")
    gold_table = get_gold_ml_table(cfg)
    logger.info(f"Targeting ML Feature Mart Observation Layer: {gold_table}")

    # 2. Load Observations and Isolate Core Hotspot
    gold_df = spark.table(gold_table)
    pickup_location_id = select_highest_volume_location(gold_df)
    logger.info(f"Isolated highest-volume zone candidate ID: {pickup_location_id}")

    history_pdf = load_location_history(gold_df, pickup_location_id=pickup_location_id)

    # 3. Construct Leakage-Safe Time-Series Frames
    logger.info(
        "Initializing chronological structural window features and threshold splits..."
    )
    train_df, val_df, test_df, thresholds, feature_columns = prepare_model_frames(
        history_pdf
    )

    logger.info(f"Loaded Contract Thresholds: {thresholds}")
    logger.info(
        f"Train Shape: {train_df.shape} | Val Shape: {val_df.shape} | Test Shape: {test_df.shape}"
    )

    X_train = train_df[feature_columns]
    y_train = train_df["demand_class"]

    X_val = val_df[feature_columns]
    y_val = val_df["demand_class"]

    X_test = test_df[feature_columns]
    y_test = test_df["demand_class"]

    # 4. Run Asymmetric Loss Function Space Hyperparameter Tuning Matrix
    artifacts_dir = PROJECT_ROOT / "artifacts" / "ml"
    logger.info(
        f"Spawning hyperparameter optimization trials (Count: {optuna_trials})..."
    )

    results = run_six_experiments(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        thresholds=thresholds,
        pickup_location_id=pickup_location_id,
        artifacts_dir=artifacts_dir,
        optuna_trials=optuna_trials,
    )

    # 5. Parse Metric Frame Diagnostics
    summary = pd.DataFrame(
        [
            {
                "run_name": r.run_name,
                "run_id": r.run_id,
                "model_family": r.model_family,
                "tuned": r.tuned,
                **r.val_metrics,
            }
            for r in results
        ]
    ).sort_values("val_mean_penalty", ascending=True)

    print(
        "\n"
        + "=" * 80
        + "\nLOCAL EXPERIMENT MATRIX RUN RESULTS (Sorted by val_mean_penalty):\n"
        + "=" * 80
    )
    print(summary.to_string(index=False))
    print("=" * 80 + "\n")

    # 6. Select Champion and Retrain on Combined Data (Train + Val)
    champion = select_champion(results)
    logger.info(f"🏆 Active Champion Selected: {champion.run_name} [{champion.run_id}]")

    final_model, test_metrics = log_test_metrics_to_champion_run(
        champion=champion,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        artifacts_dir=artifacts_dir,
    )

    # 7. Export Validated Assets to Scoring Path Container
    models_dir = get_models_dir(cfg)
    export_champion(
        final_model=final_model,
        champion=champion,
        models_dir=models_dir,
        thresholds=thresholds,
        feature_columns=feature_columns,
        pickup_location_id=pickup_location_id,
        test_metrics=test_metrics,
    )
    logger.info("🟢 Training pipeline successfully finished execution.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Production ML Pipeline Training Driver."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the local.yml configuration metadata file.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of Optuna trials to allocate per tuned family.",
    )
    args = parser.parse_args()

    run_training_pipeline(config_path=args.config, optuna_trials=args.trials)


if __name__ == "__main__":
    main()
