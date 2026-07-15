from datetime import datetime
import logging

from airflow.sdk import dag, task


@dag(
    dag_id="hello_world_dag",
    start_date=datetime(2026, 6, 1),
    schedule=None,
    catchup=False,
    tags=["basics", "infrastructure"],
)
def hello_world_dag():

    @task
    def say_hello():
        log = logging.getLogger("airflow.task")
        log.info("👋 Hello, Airflow 3.x Engine!")

    @task
    def say_goodbye():
        log = logging.getLogger("airflow.task")
        log.info("🏃‍♂️ Logging out of the task container.")

    say_hello() >> say_goodbye()


hello_world_dag()