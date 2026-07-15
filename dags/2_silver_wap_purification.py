import sys
import logging
import subprocess
from datetime import datetime
from airflow.sdk import dag, task, Asset
from airflow.sdk.exceptions import AirflowFailException
from airflow.sdk.exceptions import AirflowSkipException

BRONZE_TAXI_DATA = Asset(
    name="warehouse/bronze/yellow_taxi_raw",
    uri="iceberg://bronze.yellow_taxi_raw",
)
SILVER_TAXI_DATA = Asset(
    name="warehouse/silver/yellow_taxi_trips",
    uri="iceberg://silver.yellow_taxi_trips",
)

@dag(
    dag_id="2_silver_wap_purification",
    start_date=datetime(2026, 4, 1),
    schedule=[BRONZE_TAXI_DATA],
    catchup=False,
    max_active_runs=1,
)
def silver_wap_purification_dag():

    @task(task_id="invoke_silver_wap_job", outlets=[SILVER_TAXI_DATA])
    def invoke_silver_wap_job(**context):
        log = logging.getLogger("airflow.task")

        events = context.get("triggering_asset_events", {}).get(BRONZE_TAXI_DATA, [])
        if not events:
            raise AirflowFailException("Missing upstream Bronze asset event metadata.")

        lineage = dict(events[0].extra)

        cmd = [
            sys.executable, "/opt/airflow/jobs/silver_wap.py",
            "--config", "/opt/airflow/config/local.yml",
            "--source_filename", str(lineage["source_filename"]),
            "--ingestion_run_id", str(lineage["ingestion_run_id"]),
            "--target_year", str(lineage["source_year"]),
            "--target_month", str(lineage["source_month"]),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        log.info("Silver Spark job stdout:\n%s", result.stdout)

        if result.returncode != 0:
            log.error("Silver Spark job stderr:\n%s", result.stderr)
            raise AirflowFailException(f"Silver WAP-style job failed with exit code {result.returncode}")

        # 🛑 IDEMPOTENT ORCHESTRATION GUARD
        # Check if the compute job indicated that the file was a historical duplicate.
        if "Idempotency Skip" in result.stdout:
            log.info("🛑 Spark job detected a duplicate drop. Short-circuiting Airflow context to prevent downstream triggers.")
            raise AirflowSkipException("Skipping downstream updates because this data file has already been processed.")

        # This will only be reached and emitted if it was a genuine new run
        context["outlet_events"][SILVER_TAXI_DATA].extra = lineage

    invoke_silver_wap_job()

silver_wap_purification_dag()