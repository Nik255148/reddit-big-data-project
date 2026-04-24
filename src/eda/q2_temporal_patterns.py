"""
src/eda/q2_temporal_patterns.py

EDA Q2: What time of day and day of week correlate with viral posts?

Computes engagement metrics broken down by:
  (a) Hour of day (0-23 UTC)
  (b) Day of week (1=Sunday ... 7=Saturday in Spark)

A post is viral if score >= 1000 OR num_comments >= 100.

Run from master:
    spark-submit \
      --master spark://MASTER_PRIVATE_IP:7077 \
      --deploy-mode client \
      --executor-memory 4G --driver-memory 2G --total-executor-cores 6 \
      src/eda/q2_temporal_patterns.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyspark.sql import functions as F
from common import build_spark, S3_SUBMISSIONS, DEFAULT_DEV_MONTH

S3_RESULTS = "s3a://nik-datsbd-s2026/results"
ABS_SCORE_THRESHOLD = 1000
ABS_COMMENTS_THRESHOLD = 100


def main():
    spark = build_spark("EDA_Q2_Temporal")

    print("=" * 60)
    print("EDA Q2: Temporal Patterns of Viral Posts")
    print("=" * 60)

    path = S3_SUBMISSIONS + DEFAULT_DEV_MONTH
    print(f"Reading: {path}")
    df = spark.read.parquet(path)

    clean = (
        df
        .where(F.col("author") != "[deleted]")
        .where(~F.col("stickied"))
        .where(F.col("distinguished").isNull())
        .where(~F.col("locked"))
        .where(~F.col("quarantine"))
        .withColumn("ts", F.to_timestamp(F.col("created_utc")))
        .withColumn("hour_utc", F.hour("ts"))
        .withColumn("dayofweek", F.dayofweek("ts"))
        .withColumn("is_viral",
            ((F.col("score") >= ABS_SCORE_THRESHOLD) |
             (F.col("num_comments") >= ABS_COMMENTS_THRESHOLD)).cast("int"))
    )

    clean.cache()
    total = clean.count()
    print(f"\nTotal cleaned posts: {total:,}")

    # (a) By hour of day
    print("\n--- By Hour of Day (UTC) ---")
    by_hour = (
        clean.groupBy("hour_utc")
        .agg(
            F.count("*").alias("total_posts"),
            F.sum("is_viral").alias("viral_posts"),
            F.avg("score").alias("avg_score"),
            F.avg("num_comments").alias("avg_comments"),
        )
        .withColumn("viral_rate_pct",
                    F.round(F.col("viral_posts") / F.col("total_posts") * 100, 3))
        .withColumn("avg_score", F.round("avg_score", 2))
        .withColumn("avg_comments", F.round("avg_comments", 2))
        .orderBy("hour_utc")
    )
    by_hour.show(24, truncate=False)
    (by_hour.coalesce(1)
        .write.mode("overwrite").option("header", True)
        .csv(f"{S3_RESULTS}/q2_by_hour"))

    # (b) By day of week
    print("\n--- By Day of Week ---")
    day_map = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday",
               5: "Thursday", 6: "Friday", 7: "Saturday"}
    mapping_expr = F.create_map(
        [F.lit(x) for pair in day_map.items() for x in pair]
    )
    by_day = (
        clean.groupBy("dayofweek")
        .agg(
            F.count("*").alias("total_posts"),
            F.sum("is_viral").alias("viral_posts"),
            F.avg("score").alias("avg_score"),
            F.avg("num_comments").alias("avg_comments"),
        )
        .withColumn("day_name", mapping_expr[F.col("dayofweek")])
        .withColumn("viral_rate_pct",
                    F.round(F.col("viral_posts") / F.col("total_posts") * 100, 3))
        .withColumn("avg_score", F.round("avg_score", 2))
        .withColumn("avg_comments", F.round("avg_comments", 2))
        .orderBy("dayofweek")
    )
    by_day.show(7, truncate=False)
    (by_day.coalesce(1)
        .write.mode("overwrite").option("header", True)
        .csv(f"{S3_RESULTS}/q2_by_dayofweek"))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total cleaned posts: {total:,}")
    print(f"Results: {S3_RESULTS}/q2_by_hour/")
    print(f"Results: {S3_RESULTS}/q2_by_dayofweek/")

    spark.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
