
import argparse
import sys
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
        .appName("GoldMLObservationsJob")
        .config(f"spark.sql.catalog.{catalog}", "org.apache.iceberg.spark.SparkCatalog")
        .config(f"spark.sql.catalog.{catalog}.type", cfg["catalog"]["type"])
        .config(f"spark.sql.catalog.{catalog}.warehouse", cfg["catalog"]["warehouse_path"])
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .getOrCreate())

    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {catalog}.gold")

    gold_table = f"{catalog}.gold.hourly_ml_observations"
    silver_table = f"{catalog}.silver.yellow_taxi_trips"

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {gold_table} (
            target_year INT, target_month INT,
            pickup_location_id INT, pickup_hour_ts TIMESTAMP, ride_count INT
        ) USING iceberg
    """)

    spark.sql(f"DELETE FROM {gold_table} WHERE target_year = {args.year} AND target_month = {args.month}")

    spark.sql(f"""
        INSERT INTO {gold_table}
        WITH time_skeleton AS (
            SELECT explode(sequence(timestamp('{start_date}'), timestamp('{end_date}') - interval '1 hour', interval '1 hour')) AS pickup_hour_ts
        ),
        zone_skeleton AS (
            SELECT explode(sequence(2, 263)) AS pickup_location_id
        ),
        dense_grid AS (
            SELECT z.pickup_location_id, t.pickup_hour_ts FROM zone_skeleton z CROSS JOIN time_skeleton t
        ),
        actual_aggregates AS (
            SELECT
                pickup_location_id,
                date_trunc('hour', pickup_datetime) AS pickup_hour_ts,
                CAST(COUNT(*) AS INT) AS actual_rides
            FROM {silver_table}
            WHERE pickup_datetime >= timestamp('{start_date}') AND pickup_datetime < timestamp('{end_date}')
            GROUP BY 1, 2
        )
        SELECT
            {args.year} AS target_year,
            {args.month} AS target_month,
            g.pickup_location_id,
            g.pickup_hour_ts,
            CAST(coalesce(a.actual_rides, 0) AS INT) AS ride_count
        FROM dense_grid g
        LEFT JOIN actual_aggregates a
          ON g.pickup_location_id = a.pickup_location_id AND g.pickup_hour_ts = a.pickup_hour_ts
    """)

    final_cols = spark.table(gold_table).columns
    forbidden_leakage_fields = ["dropoff_datetime", "tip_amount", "fare_amount", "passenger_count", "trip_distance"]

    for forbidden in forbidden_leakage_fields:
        if forbidden in final_cols:
            print(f"❌ Target leakage field detected in ML mart schema: {forbidden}")
            sys.exit(1)

    print("🟢 Gold ML hourly observation mart rebuilt. Leakage check passed.")

if __name__ == "__main__":
    main()