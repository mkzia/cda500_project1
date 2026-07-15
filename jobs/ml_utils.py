from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from pyspark.sql import SparkSession


def project_root() -> Path:
    """Return the project root from either local notebooks or /opt/airflow containers."""
    env_root = os.getenv("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    cwd = Path.cwd().resolve()
    if cwd.name == "notebooks":
        return cwd.parent
    if cwd.name in {"jobs", "dags"}:
        return cwd.parent
    return cwd


def load_config(config_path: str | Path | os.PathLike | None = None) -> dict[str, Any]:
    """Load config/local.yml by default."""
    if config_path is None:
        root = project_root()
        local_candidate = root / "config" / "local.yml"
        airflow_candidate = Path("/opt/airflow/config/local.yml")
        config_path = airflow_candidate if airflow_candidate.exists() else local_candidate

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r") as f:
        return yaml.safe_load(f)


def _resolve_path(path_value: str, root: Path) -> str:
    """Resolve local relative paths while leaving container absolute paths alone."""
    if path_value.startswith("/"):
        return path_value
    return str((root / path_value).resolve())


def get_catalog_name(cfg: dict[str, Any]) -> str:
    return cfg.get("catalog", {}).get("spark_catalog_name", "nyc")


def get_gold_ml_table(cfg: dict[str, Any]) -> str:
    catalog = get_catalog_name(cfg)
    return cfg.get("tables", {}).get("gold_ml_observations", f"{catalog}.gold.hourly_ml_observations")


def get_models_dir(cfg: dict[str, Any] | None = None) -> Path:
    root = project_root()
    if cfg:
        value = cfg.get("ml", {}).get("models_dir", "models")
    else:
        value = os.getenv("MODELS_DIR", "models")
    return Path(_resolve_path(value, root))


def get_predictions_dir(cfg: dict[str, Any] | None = None) -> Path:
    root = project_root()
    if cfg:
        value = cfg.get("ml", {}).get(
            "predictions_dir", "data/predictions/hourly_demand_predictions"
        )
    else:
        value = os.getenv(
            "PREDICTIONS_DIR", "data/predictions/hourly_demand_predictions"
        )
    return Path(_resolve_path(value, root))


def create_spark_session(
    cfg: dict[str, Any], app_name: str = "TaxiDemandML"
) -> SparkSession:
    """Create a Spark session configured for the same Iceberg catalog used by the course project."""
    catalog_cfg = cfg.get("catalog", {})
    catalog = catalog_cfg.get("spark_catalog_name", "nyc")
    catalog_type = catalog_cfg.get("type", "hadoop")
    warehouse_path = catalog_cfg.get("warehouse_path", "data/warehouse")
    warehouse_path = _resolve_path(warehouse_path, project_root())

    builder = (
        SparkSession.builder.appName(app_name)
        .config(f"spark.sql.catalog.{catalog}", "org.apache.iceberg.spark.SparkCatalog")
        .config(f"spark.sql.catalog.{catalog}.type", catalog_type)
        .config(f"spark.sql.catalog.{catalog}.warehouse", warehouse_path)
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )
    )

    # Optional Spark configuration from config/local.yml.
    for key, value in cfg.get("spark", {}).items():
        builder = builder.config(key, value)

    return builder.getOrCreate()


def read_json(path: str | Path | os.PathLike) -> dict[str, Any]:
    with Path(path).open("r") as f:
        return json.load(f)


def write_json(path: str | Path | os.PathLike, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)