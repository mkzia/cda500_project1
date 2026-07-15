import argparse
import sys
import hashlib
import yaml
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import lit, current_timestamp, col

SOURCE_COLUMN_TYPES = {
    "VendorID": "int", "tpep_pickup_datetime": "timestamp", "tpep_dropoff_datetime": "timestamp",
    "passenger_count": "double", "trip_distance": "double", "RatecodeID": "double",
    "store_and_fwd_flag": "string", "PULocationID": "int", "DOLocationID": "int",
    "payment_type": "int", "fare_amount": "double", "extra": "double", "mta_tax": "double",
    "tip_amount": "double", "tolls_amount": "double", "improvement_surcharge": "double",
    "total_amount": "double", "congestion_surcharge": "double", "airport_fee": "double",
}

def validate_year_month(year: int, month: int):
    if year < 2000 or year > 2100: raise ValueError(f"Invalid year parameter configuration: {year}")
    if month < 1 or month > 12: raise ValueError(f"Invalid month parameter configuration: {month}")

def calculate_file_sha256(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source_filepath", required=True)
    parser.add_argument("--source_filename", required=True)
    parser.add_argument("--source_year", type=int, required=True)
    parser.add_argument("--source_month", type=int, required=True)
    parser.add_argument("--run_id", required=True)
    args = parser.parse_args()

    validate_year_month(args.source_year, args.source_month)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    catalog = cfg["catalog"]["spark_catalog_name"]
    computed_checksum = calculate_file_sha256(args.source_filepath)
    print(f"🔒 File checksum: {computed_checksum}")

    spark = (SparkSession.builder
        .appName("BronzeIngestionJob")
        .config(f"spark.sql.catalog.{catalog}", "org.apache.iceberg.spark.SparkCatalog")
        .config(f"spark.sql.catalog.{catalog}.type", cfg["catalog"]["type"])
        .config(f"spark.sql.catalog.{catalog}.warehouse", cfg["catalog"]["warehouse_path"])
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .getOrCreate())

    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {catalog}.audit")
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {catalog}.bronze")

    ledger_table = f"{catalog}.audit.ingestion_ledger"
    bronze_table = f"{catalog}.bronze.yellow_taxi_raw"

    spark.sql(f"CREATE TABLE IF NOT EXISTS {ledger_table} (source_filename STRING, file_checksum STRING, ingestion_run_id STRING, status STRING, finished_at TIMESTAMP) USING iceberg")
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {bronze_table} (
            VendorID INT, tpep_pickup_datetime TIMESTAMP, tpep_dropoff_datetime TIMESTAMP,
            passenger_count DOUBLE, trip_distance DOUBLE, RatecodeID DOUBLE, store_and_fwd_flag STRING,
            PULocationID INT, DOLocationID INT, payment_type INT, fare_amount DOUBLE, extra DOUBLE, mta_tax DOUBLE,
            tip_amount DOUBLE, tolls_amount DOUBLE, improvement_surcharge DOUBLE, total_amount DOUBLE,
            congestion_surcharge DOUBLE, airport_fee DOUBLE, source_filename STRING, source_checksum STRING,
            source_year INT, source_month INT, ingestion_run_id STRING, ingested_at_utc TIMESTAMP
        ) USING iceberg PARTITIONED BY (source_year, source_month)
    """)

    dup_ledger = spark.table(ledger_table).where((col("file_checksum") == computed_checksum) & (col("status") == "SUCCESS")).count()
    if dup_ledger > 0:
        print("🟢 Idempotency skip: Filename has already completed its processing in Silver. Exiting cleanly.")
        sys.exit(0)  # Short-circuit cleanly!

    dup_table = spark.table(bronze_table).where(col("source_checksum") == computed_checksum).count()
    if dup_table > 0:
        print("Idempotency guard: checksum already present in table. Re-aligning ledger.")
        ledger_df = spark.createDataFrame([Row(source_filename=args.source_filename, file_checksum=computed_checksum, ingestion_run_id=args.run_id, status="SUCCESS")]).withColumn("finished_at", current_timestamp())
        ledger_df.writeTo(ledger_table).append()
        sys.exit(0)

    df_raw = spark.read.parquet(args.source_filepath)

    for required_col, spark_type in SOURCE_COLUMN_TYPES.items():
        if required_col not in df_raw.columns:
            df_raw = df_raw.withColumn(required_col, lit(None).cast(spark_type))

    df_bronze = df_raw.select(
        col("VendorID").cast("int").alias("VendorID"),
        col("tpep_pickup_datetime").cast("timestamp").alias("tpep_pickup_datetime"),
        col("tpep_dropoff_datetime").cast("timestamp").alias("tpep_dropoff_datetime"),
        col("passenger_count").cast("double").alias("passenger_count"),
        col("trip_distance").cast("double").alias("trip_distance"),
        col("RatecodeID").cast("double").alias("RatecodeID"),
        col("store_and_fwd_flag").cast("string").alias("store_and_fwd_flag"),
        col("PULocationID").cast("int").alias("PULocationID"),
        col("DOLocationID").cast("int").alias("DOLocationID"),
        col("payment_type").cast("int").alias("payment_type"),
        col("fare_amount").cast("double").alias("fare_amount"),
        col("extra").cast("double").alias("extra"),
        col("mta_tax").cast("double").alias("mta_tax"),
        col("tip_amount").cast("double").alias("tip_amount"),
        col("tolls_amount").cast("double").alias("tolls_amount"),
        col("improvement_surcharge").cast("double").alias("improvement_surcharge"),
        col("total_amount").cast("double").alias("total_amount"),
        col("congestion_surcharge").cast("double").alias("congestion_surcharge"),
        col("airport_fee").cast("double").alias("airport_fee"),
        lit(args.source_filename).cast("string").alias("source_filename"),
        lit(computed_checksum).cast("string").alias("source_checksum"),
        lit(args.source_year).cast("int").alias("source_year"),
        lit(args.source_month).cast("int").alias("source_month"),
        lit(args.run_id).cast("string").alias("ingestion_run_id"),
        current_timestamp().alias("ingested_at_utc"),
    )

    df_bronze.writeTo(bronze_table).append()

    ledger_df = spark.createDataFrame([Row(source_filename=args.source_filename, file_checksum=computed_checksum, ingestion_run_id=args.run_id, status="SUCCESS")]).withColumn("finished_at", current_timestamp())
    ledger_df.writeTo(ledger_table).append()
    print("🟢 Bronze Ingestion complete.")

if __name__ == "__main__":
    main()