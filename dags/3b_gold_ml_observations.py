import sys
import subprocess
from datetime import datetime
from airflow.sdk import dag, task, Asset
from airflow.sdk.exceptions import AirflowFailException

SILVER_TAXI_DATA = Asset(
    name="warehouse/silver/yellow_taxi_trips",
    uri="iceberg://silver.yellow_taxi_trips",
)

@dag(
    dag_id="3b_gold_ml_observations",
    start_date=datetime(2026, 4, 1),
    schedule=[SILVER_TAXI_DATA],
    catchup=False,
    max_active_runs=1,
)
def gold_ml_observations_dag():

    @task(task_id="run_ml_observations_compile")
    def run_ml_observations_compile(**context):
        events = context.get("triggering_asset_events", {}).get(SILVER_TAXI_DATA, [])
        if not events:
            raise AirflowFailException("Missing upstream Silver asset event metadata.")

        lineage = events[0].extra

        cmd = [
            sys.executable, "/opt/airflow/jobs/gold_ml.py",
            "--config", "/opt/airflow/config/local.yml",
            "--year", str(lineage["source_year"]),
            "--month", str(lineage["source_month"]),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)

        if result.returncode != 0:
            raise AirflowFailException(f"ML observation mart compilation failed:\n{result.stderr}")

    run_ml_observations_compile()

gold_ml_observations_dag()