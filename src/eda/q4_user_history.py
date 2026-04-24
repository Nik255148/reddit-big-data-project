"""
src/eda/q4_user_history.py

EDA Q4: How does a user's posting history relate to virality?
Do power users (frequent posters) get more engagement than first-timers?

We compute:
  (a) User activity buckets: 1 post, 2-5, 6-20, 21-100, 100+ posts/month
  (b) Average score and viral rate per activity bucket
  (c) Top 20 most active authors and their avg engagement

Input : 1 month of submissions (DEFAULT_DEV_MONTH).
Output: CSVs written to S3 results path.

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

from pyspark.sql import functions as F, Window
from common import build_spark, S3_SUBMISSIONS, DEFAULT_DEV_MONTH

S3_RESULTS = "s3a://nik-datsbd-s2026/results"
ABS_SCORE_THRESHOLD = 1000
ABS_COMMENTS_THRESHOLD = 100


def main():
    spark = build_spark("EDA_Q4_UserHistory")

    print("=" * 60)
    print("EDA Q4: User Posting History vs Virality")
    print("=" * 60)

    path = S3_SUBMISSIONS + DEFAULT_DEV_MONTH
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
    )

    # Add viral flag
    clean = clean.withColumn(
        "is_viral",
        ((F.col("score") >= ABS_SCORE_THRESHOLD) |
         (F.col("num_comments") >= ABS_COMMENTS_THRESHOLD)).cast("int")
    )

    clean.cache()
    total = clean.count()
    print(f"\nTotal cleaned posts: {total:,}")

    # ------------------------------------------------------------------
    # (a) Compute per-author post count for the month
    # ------------------------------------------------------------------
    print("\n--- Computing per-author post counts ---")

    w = Window.partitionBy("author")
    with_counts = clean.withColumn(
        "author_post_count", F.count("*").over(w)
    )

    # ------------------------------------------------------------------
    # (b) Bucket authors by activity level
    # ------------------------------------------------------------------
    with_bucket = with_counts.withColumn(
        "activity_bucket",
        F.when(F.col("author_post_count") == 1, "1 post")
         .when(F.col("author_post_count") <= 5, "2-5 posts")
         .when(F.col("author_post_count") <= 20, "6-20 posts")
         .when(F.col("author_post_count") <= 100, "21-100 posts")
         .otherwise("100+ posts")
    )

    print("\n--- Virality by Author Activity Bucket ---")
    by_bucket = (
        with_bucket.groupBy("activity_bucket")
        .agg(
            F.countDistinct("author").alias("unique_authors"),
            F.count("*").alias("total_posts"),
            F.sum("is_viral").alias("viral_posts"),
            F.avg("score").alias("avg_score"),
            F.avg("num_comments").alias("avg_comments"),
        )
        .withColumn("viral_rate_pct",
                    F.round(F.col("viral_posts") / F.col("total_posts") * 100, 3))
        .withColumn("avg_score", F.round("avg_score", 2))
        .withColumn("avg_comments", F.round("avg_comments", 2))
        .orderBy("total_posts", ascending=False)
    )
    by_bucket.show(truncate=False)

    (by_bucket.coalesce(1)
        .write.mode("overwrite").option("header", True)
        .csv(f"{S3_RESULTS}/q4_by_activity_bucket"))

    # ------------------------------------------------------------------
    # (c) Top 20 most active authors
    # ------------------------------------------------------------------
    print("\n--- Top 20 Most Active Authors ---")
    top_authors = (
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
        .orderBy(F.desc("post_count"))
        .limit(20)
    )
    top_authors.show(truncate=False)

    (top_authors.coalesce(1)
        .write.mode("overwrite").option("header", True)
        .csv(f"{S3_RESULTS}/q4_top_authors"))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total cleaned posts: {total:,}")
    print(f"Results: {S3_RESULTS}/q4_by_activity_bucket/")
    print(f"Results: {S3_RESULTS}/q4_top_authors/")

    spark.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
