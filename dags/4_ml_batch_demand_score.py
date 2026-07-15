from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime

from airflow.sdk import dag, task
from airflow.sdk.exceptions import AirflowFailException


@dag(
    dag_id="4_ml_batch_demand_score",
    start_date=datetime(2026, 7, 10, 0, 0),
    schedule="@hourly",
    catchup=False,
    max_active_runs=1,
    tags=["ml", "batch-inference", "taxi-demand"],
)
def ml_batch_demand_score_dag():

    @task(task_id="score_next_simulated_hour")
    def score_next_simulated_hour(**context) -> str:
        log = logging.getLogger("airflow.task")
        dag_run = context.get("dag_run")
        logical_date = context.get("logical_date") or context.get("execution_date")

        conf = getattr(dag_run, "conf", {}) or {}
        real_now = conf.get("real_now")
        if real_now is None and logical_date is not None:
            real_now = logical_date.isoformat()

        months_back = int(conf.get("months_back", 2))

        cmd = [
            sys.executable,
            "/opt/airflow/jobs/ml_batch_score.py",
            "--config",
            "/opt/airflow/config/local.yml",
            "--real-now",
            str(real_now),
            "--months-back",
            str(months_back),
        ]

        log.info("Running Spark Iceberg batch scoring command: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        log.info("Spark job stdout:\n%s", result.stdout)

        if result.returncode != 0:
            log.error("Spark job failed. Stderr:\n%s", result.stderr)
            raise AirflowFailException(
                f"Spark scoring script failed with code {result.returncode}"
            )

        return str(real_now)

    @task(task_id="sync_iceberg_to_postgres_cache")
    def sync_iceberg_to_postgres_cache(real_now: str, **context) -> str:
        log = logging.getLogger("airflow.task")
        dag_run = context.get("dag_run")
        conf = getattr(dag_run, "conf", {}) or {}
        months_back = int(conf.get("months_back", 2))

        cmd = [
            sys.executable,
            "/opt/airflow/jobs/sync_to_postgres.py",
            "--config",
            "/opt/airflow/config/local.yml",
            "--real-now",
            real_now,
            "--months-back",
            str(months_back),
        ]

        log.info("Running CQRS operational cache sync to Postgres: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        log.info("PostgreSQL sync stdout:\n%s", result.stdout)

        if result.returncode != 0:
            log.error("PostgreSQL cache replication failed. Stderr:\n%s", result.stderr)
            raise AirflowFailException(
                f"Cache sync script failed with code {result.returncode}"
            )

        log.info("🟢 CQRS Operational Sync Complete.")
        return real_now

    @task(task_id="evaluate_past_predictions_feedback")
    def evaluate_past_predictions_feedback(real_now: str, **context):
        log = logging.getLogger("airflow.task")

        cmd = [
            sys.executable,
            "/opt/airflow/jobs/ml_evaluate_feedback.py",
            "--real-now",
            real_now,
        ]

        log.info(
            "Running MLOps Evaluation Loop and Penalty Assessment: %s", " ".join(cmd)
        )
        result = subprocess.run(cmd, capture_output=True, text=True)
        log.info("Evaluation loop stdout:\n%s", result.stdout)

        if result.returncode != 0:
            log.error("Feedback evaluation loop failed. Stderr:\n%s", result.stderr)
            raise AirflowFailException(
                f"Evaluation script failed with code {result.returncode}"
            )

        log.info(
            "🏁 Feedback evaluation successfully settled metrics for the processed timeline."
        )

    # Main orchestration lineage
    real_now_ts = score_next_simulated_hour()
    sync_completed_ts = sync_iceberg_to_postgres_cache(real_now_ts)
    evaluate_past_predictions_feedback(sync_completed_ts)


ml_batch_demand_score_dag()
