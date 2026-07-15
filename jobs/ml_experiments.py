from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from jobs.ml_features import DemandThresholds, COST_MATRIX, mean_operational_penalty
from jobs.ml_utils import write_json


@dataclass
class ExperimentResult:
    run_id: str
    run_name: str
    model_family: str
    tuned: bool
    model: Pipeline
    params: dict[str, Any]
    val_metrics: dict[str, float]


def make_logistic_regression(params: dict[str, Any] | None = None) -> Pipeline:
    params = params or {}
    clf = LogisticRegression(
        max_iter=2000,
        random_state=42,
        **params,
    )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", clf),
        ]
    )


def make_random_forest(params: dict[str, Any] | None = None) -> Pipeline:
    params = params or {}
    clf = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", clf),
        ]
    )


def make_lightgbm(params: dict[str, Any] | None = None) -> Pipeline:
    params = params or {}
    clf = LGBMClassifier(random_state=42, verbosity=-1, **params)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", clf),
        ]
    )


def cost_sensitive_predict(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Maps classifier probabilities to predictions that minimize expected asymmetric operational cost.

    Expected Cost Matrix = Class Probabilities x Business Cost Matrix Matrix
    """
    probabilities = model.predict_proba(X)
    full_probabilities = np.zeros((len(X), 3), dtype=float)

    for column_index, class_value in enumerate(model.classes_):
        full_probabilities[:, int(class_value)] = probabilities[:, column_index]

    # Matrix multiplication: shape (N, 3) @ (3, 3) -> (N, 3) expected costs per choice
    expected_costs = full_probabilities @ COST_MATRIX
    return expected_costs.argmin(axis=1)


def evaluate_classifier(
    model: Pipeline, X: pd.DataFrame, y: pd.Series, prefix: str
) -> dict[str, float]:
    """Generates diagnostic scores using asymmetric business objective boundaries."""
    preds = cost_sensitive_predict(model, X)
    actuals = np.asarray(y, dtype=int)

    total_penalty = float(COST_MATRIX[actuals, preds].sum())
    mean_penalty = mean_operational_penalty(y, preds)

    return {
        f"{prefix}_mean_penalty": mean_penalty,
        f"{prefix}_total_penalty": total_penalty,
        f"{prefix}_accuracy": float(accuracy_score(y, preds)),
        f"{prefix}_precision_macro": float(
            precision_score(y, preds, average="macro", zero_division=0)
        ),
        f"{prefix}_recall_macro": float(
            recall_score(y, preds, average="macro", zero_division=0)
        ),
        f"{prefix}_macro_f1": float(
            f1_score(y, preds, average="macro", zero_division=0)
        ),
    }


def save_confusion_matrix(
    model: Pipeline, X: pd.DataFrame, y: pd.Series, path: Path, title: str
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    preds = cost_sensitive_predict(model, X)
    cm = confusion_matrix(y, preds, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Low", "Normal", "High"]
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(values_format="d", ax=ax, cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _objective_for_family(
    family: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        if family == "LogisticRegression":
            params = {
                "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
                "class_weight": trial.suggest_categorical(
                    "class_weight", [None, "balanced"]
                ),
                "solver": "lbfgs",
            }
            model = make_logistic_regression(params)
        elif family == "RandomForest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 24),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "class_weight": trial.suggest_categorical(
                    "class_weight", [None, "balanced"]
                ),
            }
            model = make_random_forest(params)
        elif family == "LightGBM":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.2, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "max_depth": trial.suggest_int("max_depth", 3, 16),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "class_weight": trial.suggest_categorical(
                    "class_weight", [None, "balanced"]
                ),
            }
            model = make_lightgbm(params)
        else:
            raise ValueError(f"Unknown family: {family}")

        model.fit(X_train, y_train)
        preds = cost_sensitive_predict(model, X_val)

        # Optuna now minimizes the actual financial metrics that power your dashboard!
        return mean_operational_penalty(y_true=y_val, y_pred=preds)

    return objective


def tune_family(
    family: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 20,
) -> dict[str, Any]:
    # Changed direction to minimize penalty metrics
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(
        _objective_for_family(family, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    return dict(study.best_params)


def build_model(family: str, params: dict[str, Any] | None = None) -> Pipeline:
    if family == "LogisticRegression":
        return make_logistic_regression(params)
    if family == "RandomForest":
        return make_random_forest(params)
    if family == "LightGBM":
        return make_lightgbm(params)
    raise ValueError(f"Unknown family: {family}")


def run_one_experiment(
    run_name: str,
    family: str,
    tuned: bool,
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    thresholds: DemandThresholds,
    pickup_location_id: int,
    artifacts_dir: Path,
) -> ExperimentResult:
    model = build_model(family, params)

    simulated_run_id = (
        f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_name.lower()}"
    )
    run_dir = artifacts_dir / simulated_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\nStarting Local Experiment: {run_name}\n{'='*60}")

    model.fit(X_train, y_train)
    val_metrics = evaluate_classifier(model, X_val, y_val, prefix="val")

    logged_params = {
        "model_family": family,
        "tuned": str(tuned),
        "pickup_location_id": int(pickup_location_id),
        "q25": float(thresholds.q25),
        "q75": float(thresholds.q75),
    }
    for k, v in params.items():
        logged_params[f"param_{k}"] = v

    write_json(run_dir / "params.json", logged_params)
    write_json(run_dir / "metrics.json", val_metrics)

    print(f"Logged Parameters:\n{json.dumps(logged_params, indent=2)}")
    print(f"Logged Validation Metrics:\n{json.dumps(val_metrics, indent=2)}")

    cm_path = run_dir / "confusion_matrix.png"
    save_confusion_matrix(model, X_val, y_val, cm_path, title=run_name)

    return ExperimentResult(
        run_id=simulated_run_id,
        run_name=run_name,
        model_family=family,
        tuned=tuned,
        model=model,
        params=params,
        val_metrics=val_metrics,
    )


def run_six_experiments(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    thresholds: DemandThresholds,
    pickup_location_id: int,
    artifacts_dir: Path,
    optuna_trials: int = 20,
) -> list[ExperimentResult]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    plan = [
        ("EXP-01_LogisticRegression_Baseline", "LogisticRegression", False),
        ("EXP-02_LogisticRegression_Optuna", "LogisticRegression", True),
        ("EXP-03_RandomForest_Baseline", "RandomForest", False),
        ("EXP-04_RandomForest_Optuna", "RandomForest", True),
        ("EXP-05_LightGBM_Baseline", "LightGBM", False),
        ("EXP-06_LightGBM_Optuna", "LightGBM", True),
    ]

    results: list[ExperimentResult] = []
    for run_name, family, tuned in plan:
        params = (
            tune_family(family, X_train, y_train, X_val, y_val, n_trials=optuna_trials)
            if tuned
            else {}
        )

        result = run_one_experiment(
            run_name=run_name,
            family=family,
            tuned=tuned,
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            thresholds=thresholds,
            pickup_location_id=pickup_location_id,
            artifacts_dir=artifacts_dir,
        )
        results.append(result)

    return results


def select_champion(results: list[ExperimentResult]) -> ExperimentResult:
    if not results:
        raise ValueError("No experiment results were supplied.")
    # Target choice selection changes from max(macro_f1) to min(val_mean_penalty)
    return min(results, key=lambda r: r.val_metrics["val_mean_penalty"])


def log_test_metrics_to_champion_run(
    champion: ExperimentResult,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    artifacts_dir: Path,
) -> tuple[Pipeline, dict[str, float]]:
    """Retrains the active champion on train+validation data before test evaluation."""
    print(
        f"\n{'='*60}\nRetraining Champion on Train + Validation Sets: {champion.run_name}\n{'='*60}"
    )

    # 1. Expand historical scope to capture trailing structural trends
    X_train_final = pd.concat([X_train, X_val], ignore_index=True)
    y_train_final = pd.concat([y_train, y_val], ignore_index=True)

    final_fitted_model = build_model(champion.model_family, champion.params)
    final_fitted_model.fit(X_train_final, y_train_final)

    # 2. Touch the final test set EXACTLY ONCE
    test_metrics = evaluate_classifier(
        final_fitted_model, X_test, y_test, prefix="test"
    )

    run_dir = artifacts_dir / champion.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = run_dir / "metrics.json"
    existing_metrics = {}
    if metrics_file.exists():
        with metrics_file.open("r") as f:
            existing_metrics = json.load(f)

    combined_metrics = {**existing_metrics, **test_metrics}
    write_json(metrics_file, combined_metrics)

    print(f"Logged Retrained Test Metrics:\n{json.dumps(test_metrics, indent=2)}")

    cm_path = run_dir / "test_confusion_matrix.png"
    save_confusion_matrix(
        final_fitted_model,
        X_test,
        y_test,
        cm_path,
        title=f"{champion.run_name} FINAL TEST",
    )

    return final_fitted_model, test_metrics


def export_champion(
    final_model: Pipeline,
    champion: ExperimentResult,
    models_dir: Path,
    thresholds: DemandThresholds,
    feature_columns: list[str],
    pickup_location_id: int,
    test_metrics: dict[str, float],
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "champion_model.pkl"
    metadata_path = models_dir / "champion_metadata.json"

    # Save out the complete model fitted on training + validation
    joblib.dump(final_model, model_path)

    metadata = {
        "pickup_location_id": int(pickup_location_id),
        "model_name": champion.model_family,
        "model_version": "local-v2-cost-sensitive",
        "penalty_version": "asymmetric-v1",  # Added contract tracking column
        "mlflow_run_id": champion.run_id,
        "run_name": champion.run_name,
        "tuned": champion.tuned,
        "best_params": champion.params,
        "q25": float(thresholds.q25),
        "q75": float(thresholds.q75),
        "features": feature_columns,
        "test_metrics": test_metrics,
    }
    write_json(metadata_path, metadata)
    print(f"\nSuccessfully exported finalized model pipeline to {models_dir}")
