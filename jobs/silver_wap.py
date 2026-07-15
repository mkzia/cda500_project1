import argparse
import sys
import hashlib
import yaml
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col, lit, current_timestamp

def month_bounds(year: int, month: int) -> tuple[str, str]:
    if year < 2000 or year > 2100: raise ValueError(f"Invalid year tracking parameter: {year}")
    if month < 1 or month > 12: raise ValueError(f"Invalid month tracking parameter: {month}")

    start_date = f"{year}-{month:02}-01 00:00:00"
    if month == 12:
        end_date = f"{year + 1}-01-01 00:00:00"
    else:
        end_date = f"{year}-{month + 1:02}-01 00:00:00"
    return start_date, end_date

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source_filename", required=True)
    parser.add_argument("--ingestion_run_id", required=True)
    parser.add_argument("--target_year", type=int, required=True)
    parser.add_argument("--target_month", type=int, required=True)
    args = parser.parse_args()

    start_date, end_date = month_bounds(args.target_year, args.target_month)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    catalog = cfg["catalog"]["spark_catalog_name"]
    safe_run_hash = hashlib.sha256(args.ingestion_run_id.encode("utf-8")).hexdigest()[:16]
    candidate_table_name = f"{catalog}.silver.candidate_{safe_run_hash}"

    spark = (SparkSession.builder
        .appName("SilverWAPPurificationJob")
        .config(f"spark.sql.catalog.{catalog}", "org.apache.iceberg.spark.SparkCatalog")
        .config(f"spark.sql.catalog.{catalog}.type", cfg["catalog"]["type"])
        .config(f"spark.sql.catalog.{catalog}.warehouse", cfg["catalog"]["warehouse_path"])
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .getOrCreate())

    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {catalog}.audit")
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {catalog}.bronze")
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {catalog}.silver")
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {catalog}.quarantine")

    bronze_table = f"{catalog}.bronze.yellow_taxi_raw"
    silver_table = f"{catalog}.silver.yellow_taxi_trips"
    silver_ledger_table = f"{catalog}.audit.silver_publish_ledger"
    quarantine_table = f"{catalog}.quarantine.yellow_taxi_trips"

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {silver_table} (
            pickup_datetime TIMESTAMP, dropoff_datetime TIMESTAMP,
            pickup_location_id INT, dropoff_location_id INT,
            trip_distance DOUBLE, fare_amount DOUBLE, tip_amount DOUBLE,
            payment_type INT, passenger_count INT,
            source_filename STRING, source_checksum STRING,
            source_year INT, source_month INT,
            ingestion_run_id STRING, published_at_utc TIMESTAMP
        ) USING iceberg PARTITIONED BY (days(pickup_datetime))
    """)

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {quarantine_table} (
            VendorID INT, tpep_pickup_datetime TIMESTAMP, tpep_dropoff_datetime TIMESTAMP,
            passenger_count DOUBLE, trip_distance DOUBLE, RatecodeID DOUBLE, store_and_fwd_flag STRING,
            PULocationID INT, DOLocationID INT, payment_type INT, fare_amount DOUBLE, extra DOUBLE, mta_tax DOUBLE,
            tip_amount DOUBLE, tolls_amount DOUBLE, improvement_surcharge DOUBLE, total_amount DOUBLE,
            congestion_surcharge DOUBLE, airport_fee DOUBLE, source_filename STRING, source_checksum STRING,
            source_year INT, source_month INT, ingestion_run_id STRING, wap_run_id STRING, quarantined_at TIMESTAMP
        ) USING iceberg
    """)

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {silver_ledger_table} (
            ingestion_run_id STRING, source_filename STRING, status STRING, verified_at TIMESTAMP
        ) USING iceberg
    """)

    already_published = spark.table(silver_ledger_table).where((col("ingestion_run_id") == args.ingestion_run_id) & (col("status") == "SUCCESS")).count()
    if already_published > 0:
        print(f"Idempotency guard: Silver ledger already has SUCCESS. Skipping.")
        sys.exit(0)

    already_in_silver = spark.table(silver_table).where(col("ingestion_run_id") == args.ingestion_run_id).count()
    if already_in_silver > 0:
        print(f"Idempotency guard: Silver already contains run_id. Re-aligning ledger.")
        ledger_df = spark.createDataFrame([Row(ingestion_run_id=args.ingestion_run_id, source_filename=args.source_filename, status="SUCCESS")]).withColumn("verified_at", current_timestamp())
        ledger_df.writeTo(silver_ledger_table).append()
        sys.exit(0)

    # 1. Fetch current runtime batch framework data rows
    df_raw = spark.table(bronze_table).where(col("ingestion_run_id") == args.ingestion_run_id)

    # 🔄 2. IDEMPOTENT MULTI-LAYER FALLBACK LOGIC
    # If the current run token yielded 0 rows, check if the file was skipped by Bronze as a duplicate.
    if df_raw.limit(1).count() == 0:
        print(f"ℹ️ Current run ID matched 0 rows. Checking ledger for original successful execution of: {args.source_filename}")
        ledger_table = f"{catalog}.audit.ingestion_ledger"

        orig_run = spark.table(ledger_table).where(
            (col("source_filename") == args.source_filename) & (col("status") == "SUCCESS")
        ).select("ingestion_run_id").orderBy(col("finished_at").asc()).first()

        if orig_run:
            historical_run_id = orig_run["ingestion_run_id"]
            print(f"🔄 Duplicate file detected. Tracking history back to original historical run ID: {historical_run_id}")

            # Check if this original run token has already been compiled into the Silver layer
            historical_published = spark.table(silver_ledger_table).where(
                (col("ingestion_run_id") == historical_run_id) & (col("status") == "SUCCESS")
            ).count()

            if historical_published > 0:
                print("🟢 Idempotency Skip: Filename has already completed its processing in Silver. Exiting cleanly.")
                sys.exit(0)  # Short-circuit cleanly!

            # If the original run was skipped but not pushed to Silver yet, grab that data
            df_raw = spark.table(bronze_table).where(col("ingestion_run_id") == historical_run_id)
            args.ingestion_run_id = historical_run_id

    # 3. Ultimate strict lineage gate if no records can be resolved historically
    if df_raw.limit(1).count() == 0:
        print(f"❌ [UPDATED] No Bronze rows found for ingestion_run_id={args.ingestion_run_id}")
        sys.exit(1)

    quantiles_amt = df_raw.approxQuantile("total_amount", [0.999], 0.001)
    quantiles_dist = df_raw.approxQuantile("trip_distance", [0.999], 0.001)

    if not quantiles_amt or not quantiles_dist:
        print("❌ Quality gate failure: unable to compute quantiles.")
        sys.exit(1)

    max_total_amount = float(quantiles_amt[0])
    max_distance = float(quantiles_dist[0])

    valid_condition = (
        col("tpep_pickup_datetime").isNotNull() & col("tpep_dropoff_datetime").isNotNull() &
        col("PULocationID").isNotNull() & col("DOLocationID").isNotNull() &
        col("total_amount").isNotNull() & col("trip_distance").isNotNull() & col("passenger_count").isNotNull() &
        (col("tpep_dropoff_datetime").cast("long") > col("tpep_pickup_datetime").cast("long")) &
        (col("total_amount") >= 3.0) & (col("total_amount") <= max_total_amount) &
        (col("trip_distance") > 0.0) & (col("trip_distance") <= max_distance) &
        (~col("PULocationID").isin([1, 264, 265])) &
        (col("tpep_pickup_datetime") >= start_date) & (col("tpep_pickup_datetime") < end_date) &
        (col("passenger_count") >= 1) & (col("passenger_count") <= 5)
    )

    df_clean = (df_raw.filter(valid_condition).select(
            col("tpep_pickup_datetime").alias("pickup_datetime"), col("tpep_dropoff_datetime").alias("dropoff_datetime"),
            col("PULocationID").cast("int").alias("pickup_location_id"), col("DOLocationID").cast("int").alias("dropoff_location_id"),
            col("trip_distance").cast("double").alias("trip_distance"), col("fare_amount").cast("double").alias("fare_amount"),
            col("tip_amount").cast("double").alias("tip_amount"), col("payment_type").cast("int").alias("payment_type"),
            col("passenger_count").cast("int").alias("passenger_count"), lit(args.source_filename).cast("string").alias("source_filename"),
            col("source_checksum").cast("string").alias("source_checksum"), col("source_year").cast("int").alias("source_year"),
            col("source_month").cast("int").alias("source_month"), lit(args.ingestion_run_id).cast("string").alias("ingestion_run_id"),
            current_timestamp().alias("published_at_utc"),
        ))

    spark.sql(f"DROP TABLE IF EXISTS {candidate_table_name}")

    try:
        df_clean.writeTo(candidate_table_name).using("iceberg").create()

        total_count = df_raw.count()
        valid_count = spark.read.table(candidate_table_name).count()
        drop_rate = ((total_count - valid_count) / total_count) * 100 if total_count > 0 else 0.0

        print(f"📊 Quality audit: Total={total_count}, Valid={valid_count}, Dropped={drop_rate:.2f}%")

        if drop_rate > 35.0:
            print(f"❌ WAP-style gate breach: dropped row rate {drop_rate:.2f}% exceeded 35% threshold.")
            sys.exit(1)

        df_quarantine = df_raw.filter(~valid_condition).select(
            col("VendorID").cast("int").alias("VendorID"), col("tpep_pickup_datetime").cast("timestamp").alias("tpep_pickup_datetime"),
            col("tpep_dropoff_datetime").cast("timestamp").alias("tpep_dropoff_datetime"), col("passenger_count").cast("double").alias("passenger_count"),
            col("trip_distance").cast("double").alias("trip_distance"), col("RatecodeID").cast("double").alias("RatecodeID"),
            col("store_and_fwd_flag").cast("string").alias("store_and_fwd_flag"), col("PULocationID").cast("int").alias("PULocationID"),
            col("DOLocationID").cast("int").alias("DOLocationID"), col("payment_type").cast("int").alias("payment_type"),
            col("fare_amount").cast("double").alias("fare_amount"), col("extra").cast("double").alias("extra"),
            col("mta_tax").cast("double").alias("mta_tax"), col("tip_amount").cast("double").alias("tip_amount"),
            col("tolls_amount").cast("double").alias("tolls_amount"), col("improvement_surcharge").cast("double").alias("improvement_surcharge"),
            col("total_amount").cast("double").alias("total_amount"), col("congestion_surcharge").cast("double").alias("congestion_surcharge"),
            col("airport_fee").cast("double").alias("airport_fee"), col("source_filename").cast("string").alias("source_filename"),
            col("source_checksum").cast("string").alias("source_checksum"), col("source_year").cast("int").alias("source_year"),
            col("source_month").cast("int").alias("source_month"), col("ingestion_run_id").cast("string").alias("ingestion_run_id"),
            lit(safe_run_hash).cast("string").alias("wap_run_id"), current_timestamp().alias("quarantined_at")
        )
        df_quarantine.writeTo(quarantine_table).append()

        print("🚀 Promoting valid candidate rows into production Silver after audit approval.")
        spark.read.table(candidate_table_name).writeTo(silver_table).append()

        ledger_df = spark.createDataFrame([Row(ingestion_run_id=args.ingestion_run_id, source_filename=args.source_filename, status="SUCCESS")]).withColumn("verified_at", current_timestamp())
        ledger_df.writeTo(silver_ledger_table).append()

    finally:
        spark.sql(f"DROP TABLE IF EXISTS {candidate_table_name}")
        print("🧹 Temporary Candidate staging table purged from metadata registry.")

if __name__ == "__main__":
    main()