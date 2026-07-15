from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat, lit, lpad, count

# 1. Initialize Spark Session (Ensure Iceberg catalog is configured)
spark = (
    SparkSession.builder.appName("Gold_Layer_Audit")
    .config("spark.sql.catalog.nyc", "org.apache.iceberg.spark.SparkCatalog")
    .config("spark.sql.catalog.nyc.type", "hadoop")
    .config("spark.sql.catalog.nyc.warehouse", "/opt/airflow/data/warehouse")
    .config(
        "spark.sql.extensions",
        "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
    )
    .getOrCreate()
)

# 2. Query the Gold Table
gold_ml_df = spark.table("nyc.gold.hourly_ml_observations")

# 3. Aggregate counts by YYYY-MM
coverage_report = (
    gold_ml_df.withColumn(
        "year_month",
        concat(col("target_year"), lit("-"), lpad(col("target_month"), 2, "0")),
    )
    .groupBy("year_month")
    .agg(count("*").alias("observation_count"))
    .orderBy(col("year_month").asc())
)

print("📊 Gold Layer Coverage Report (Expected: > 0 per month):")
coverage_report.show(truncate=False)

# 4. Optional: Assert completeness if you expect a specific range
# (e.g., ensure no gaps between 2025-01 and 2026-05)
