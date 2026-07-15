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
    dag_id="3a_gold_bi_marts",
    start_date=datetime(2026, 4, 1),
    schedule=[SILVER_TAXI_DATA],
    catchup=False,
    max_active_runs=1,
)
def gold_bi_marts_dag():

    @task(task_id="run_bi_aggregations")
    def run_bi_aggregations(**context):
        events = context.get("triggering_asset_events", {}).get(SILVER_TAXI_DATA, [])
        if not events:
            raise AirflowFailException("Missing upstream Silver asset event metadata.")

        lineage = events[0].extra

        cmd = [
            sys.executable, "/opt/airflow/jobs/gold_bi.py",
            "--config", "/opt/airflow/config/local.yml",
            "--year", str(lineage["source_year"]),
            "--month", str(lineage["source_month"]),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)

        if result.returncode != 0:
            raise AirflowFailException(f"BI mart compilation failed:\n{result.stderr}")

    run_bi_aggregations()

gold_bi_marts_dag()