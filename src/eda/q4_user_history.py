"""
src/eda/q4_user_history.py

EDA Q4: How does a user's posting history relate to virality?
Optimized version - uses groupBy instead of Window functions.

Run from master:
    spark-submit \
      --master spark://MASTER_PRIVATE_IP:7077 \
      --deploy-mode client \
      --executor-memory 4G --driver-memory 2G --total-executor-cores 6 \
      src/eda/q4_user_history.py
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
    spark = build_spark("EDA_Q4_UserHistory")

    print("=" * 60)
    print("EDA Q4: User Posting History vs Virality")
    print("=" * 60)

    path = S3_SUBMISSIONS  # full dataset
    print(f"Reading: {path}")
    df = spark.read.parquet(path)

    clean = (
        df
        .where(F.col("author") != "[deleted]")
        .where(F.col("author") != "AutoModerator")
        .where(~F.col("stickied"))
        .where(F.col("distinguished").isNull())
        .where(~F.col("locked"))
        .where(~F.col("quarantine"))
        .withColumn("is_viral",
            ((F.col("score") >= ABS_SCORE_THRESHOLD) |
             (F.col("num_comments") >= ABS_COMMENTS_THRESHOLD)).cast("int"))
    )

    clean.cache()
    total = clean.count()
    print(f"\nTotal cleaned posts: {total:,}")

    # ------------------------------------------------------------------
    # Step 1: Compute per-author stats with a single groupBy (no Window)
    # ------------------------------------------------------------------
    print("\n--- Computing per-author stats ---")
    author_stats = (
        clean.groupBy("author")
        .agg(
            F.count("*").alias("post_count"),
            F.sum("is_viral").alias("viral_posts"),
            F.avg("score").alias("avg_score"),
            F.avg("num_comments").alias("avg_comments"),
            F.max("score").alias("max_score"),
        )
        .withColumn("viral_rate_pct",
                    F.round(F.col("viral_posts") / F.col("post_count") * 100, 3))
        .withColumn("avg_score", F.round("avg_score", 2))
        .withColumn("avg_comments", F.round("avg_comments", 2))
    )

    author_stats.cache()

    # ------------------------------------------------------------------
    # Step 2: Bucket by activity level
    # ------------------------------------------------------------------
    print("\n--- Virality by Author Activity Bucket ---")
    with_bucket = author_stats.withColumn(
        "activity_bucket",
        F.when(F.col("post_count") == 1, "1 post")
         .when(F.col("post_count") <= 5, "2-5 posts")
         .when(F.col("post_count") <= 20, "6-20 posts")
         .when(F.col("post_count") <= 100, "21-100 posts")
         .otherwise("100+ posts")
    )

    by_bucket = (
        with_bucket.groupBy("activity_bucket")
        .agg(
            F.countDistinct("author").alias("unique_authors"),
            F.sum("post_count").alias("total_posts"),
            F.sum("viral_posts").alias("viral_posts"),
            F.avg("avg_score").alias("avg_score"),
            F.avg("avg_comments").alias("avg_comments"),
        )
        .withColumn("viral_rate_pct",
                    F.round(F.col("viral_posts") / F.col("total_posts") * 100, 3))
        .withColumn("avg_score", F.round("avg_score", 2))
        .withColumn("avg_comments", F.round("avg_comments", 2))
        .orderBy(F.desc("total_posts"))
    )
    by_bucket.show(truncate=False)

    (by_bucket.coalesce(1)
        .write.mode("overwrite").option("header", True)
        .csv(f"{S3_RESULTS}/q4_by_activity_bucket_full"))

    # ------------------------------------------------------------------
    # Step 3: Top 20 most active authors
    # ------------------------------------------------------------------
    print("\n--- Top 20 Most Active Authors ---")
    top_authors = (
        author_stats
        .orderBy(F.desc("post_count"))
        .limit(20)
    )
    top_authors.show(truncate=False)

    (top_authors.coalesce(1)
        .write.mode("overwrite").option("header", True)
        .csv(f"{S3_RESULTS}/q4_top_authors_full"))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total cleaned posts: {total:,}")
    print(f"Results: {S3_RESULTS}/q4_by_activity_bucket_full/")
    print(f"Results: {S3_RESULTS}/q4_top_authors_full/")

    spark.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()