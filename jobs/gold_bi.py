import argparse
import yaml
from pyspark.sql import SparkSession

def month_bounds(year: int, month: int) -> tuple[str, str]:
    if year < 2000 or year > 2100: raise ValueError(f"Invalid year: {year}")
    if month < 1 or month > 12: raise ValueError(f"Invalid month: {month}")

    start_date = f"{year}-{month:02}-01 00:00:00"
    if month == 12:
        end_date = f"{year + 1}-01-01 00:00:00"
    else:
        end_date = f"{year}-{month + 1:02}-01 00:00:00"
    return start_date, end_date

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    args = parser.parse_args()

    start_date, end_date = month_bounds(args.year, args.month)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    catalog = cfg["catalog"]["spark_catalog_name"]

    spark = (SparkSession.builder
        .appName("GoldBIMartsJob")
        .config(f"spark.sql.catalog.{catalog}", "org.apache.iceberg.spark.SparkCatalog")
        .config(f"spark.sql.catalog.{catalog}.type", cfg["catalog"]["type"])
        .config(f"spark.sql.catalog.{catalog}.warehouse", cfg["catalog"]["warehouse_path"])
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .getOrCreate())

    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {catalog}.gold")

    gold_table = f"{catalog}.gold.hourly_fleet_utilization"
    silver_table = f"{catalog}.silver.yellow_taxi_trips"

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {gold_table} (
            target_year INT, target_month INT,
            pickup_hour_ts TIMESTAMP, pickup_location_id INT,
            trip_count BIGINT, avg_distance DOUBLE
        ) USING iceberg
    """)

    spark.sql(f"DELETE FROM {gold_table} WHERE target_year = {args.year} AND target_month = {args.month}")

    spark.sql(f"""
        INSERT INTO {gold_table}
        SELECT
            {args.year} AS target_year,
            {args.month} AS target_month,
            date_trunc('hour', pickup_datetime) AS pickup_hour_ts,
            pickup_location_id,
            COUNT(*) AS trip_count,
            AVG(trip_distance) AS avg_distance
        FROM {silver_table}
        WHERE pickup_datetime >= timestamp('{start_date}') AND pickup_datetime < timestamp('{end_date}')
        GROUP BY 1, 2, 3, 4
    """)
    print("🟢 Gold BI hourly fleet utilization mart rebuilt for target month.")

if __name__ == "__main__":
    main()