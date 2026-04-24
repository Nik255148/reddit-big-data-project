"""
src/eda/q1_virality_rates.py

EDA Q1: Which subreddits have the highest virality rate?

We compute "virality rate" two ways:
  (a) ABSOLUTE — viral = score >= 1000 OR num_comments >= 100
  (b) RELATIVE — viral = score >= 10 x subreddit's median score

Both definitions are filtered to subreddits with >= 1000 posts so the
percentages are statistically meaningful. Top 25 shown in each list.

Input : 1 month of submissions from S3 (DEFAULT_DEV_MONTH).
Output: Two CSVs written under RESULTS_DIR for later plotting / reporting.

Run from master node's ~/project/ directory:
    spark-submit \
      --master spark://$MASTER_PRIVATE_IP:7077 \
      --deploy-mode client \
      --executor-memory 4G --driver-memory 2G --total-executor-cores 6 \
      src/eda/q1_virality_rates.py
"""
import sys
import os

# Make `import common` work when we spark-submit a file inside src/eda/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyspark.sql import functions as F, Window
from common import build_spark, S3_SUBMISSIONS, DEFAULT_DEV_MONTH, RESULTS_DIR


# ---- Tunable parameters ----
MIN_POSTS_PER_SUBREDDIT = 1000
ABS_SCORE_THRESHOLD = 1000
ABS_COMMENTS_THRESHOLD = 100
REL_MULTIPLIER = 10
TOP_N = 25


def main():
    spark = build_spark("EDA_Q1_Virality")

    print("=" * 60)
    print("EDA Q1: Virality Rates by Subreddit")
    print("=" * 60)

    path = S3_SUBMISSIONS
    print(f"Reading: {path}")
    df = spark.read.parquet(path)

    # Filter out noise that distorts virality metrics:
    #   - [deleted] authors (we can't analyze user behavior for them)
    #   - stickied/distinguished (mod announcements, not organic content)
    #   - locked/quarantine (community-level interventions)
    clean = (
        df
        .where(F.col("author") != "[deleted]")
        .where(~F.col("stickied"))
        .where(F.col("distinguished").isNull())
        .where(~F.col("locked"))
        .where(~F.col("quarantine"))
    )

    # We scan `clean` multiple times below. Caching avoids re-reading from S3.
    clean.cache()
    total_clean = clean.count()
    print(f"\nPosts after cleaning: {total_clean:,}")

    # ------------------------------------------------------------------
    # (a) ABSOLUTE virality rate per subreddit
    # A post is viral if it crossed a hard threshold — same bar for every sub.
    # ------------------------------------------------------------------
    print("\n--- Computing ABSOLUTE virality rates ---")

    abs_flagged = clean.withColumn(
        "is_viral_abs",
        ((F.col("score") >= ABS_SCORE_THRESHOLD) |
         (F.col("num_comments") >= ABS_COMMENTS_THRESHOLD)).cast("int"),
    )

    abs_by_sub = (
        abs_flagged.groupBy("subreddit")
        .agg(
            F.count("*").alias("total_posts"),
            F.sum("is_viral_abs").alias("viral_posts"),
            F.avg("score").alias("avg_score"),
            F.avg("num_comments").alias("avg_comments"),
        )
        .where(F.col("total_posts") >= MIN_POSTS_PER_SUBREDDIT)
        .withColumn(
            "virality_rate_pct",
            F.round(F.col("viral_posts") / F.col("total_posts") * 100, 3),
        )
        .orderBy(F.desc("virality_rate_pct"))
    )

    print(f"\nTop {TOP_N} subreddits by ABSOLUTE virality rate")
    print(f"(viral = score >= {ABS_SCORE_THRESHOLD} OR num_comments >= {ABS_COMMENTS_THRESHOLD})")
    print(f"(subreddits with >= {MIN_POSTS_PER_SUBREDDIT} posts only)")
    abs_by_sub.show(TOP_N, truncate=False)

    (abs_by_sub.limit(200)
        .coalesce(1)
        .write.mode("overwrite").option("header", True)
        .csv("s3a://nik-datsbd-s2026/results/q1_virality_absolute_full"))

    # ------------------------------------------------------------------
    # (b) RELATIVE virality rate per subreddit
    # A post is viral if it beat its OWN subreddit's median by REL_MULTIPLIER.
    # Fairer across big vs small subreddits.
    # ------------------------------------------------------------------
    print("\n--- Computing RELATIVE virality rates ---")

    w = Window.partitionBy("subreddit")
    rel_flagged = (
        clean
        .withColumn(
            "sub_median_score",
            F.expr("percentile_approx(score, 0.5)").over(w),
        )
        .withColumn(
            "is_viral_rel",
            (F.col("score") >= REL_MULTIPLIER *
             F.greatest(F.col("sub_median_score"), F.lit(1))).cast("int"),
        )
    )

    rel_by_sub = (
        rel_flagged.groupBy("subreddit")
        .agg(
            F.count("*").alias("total_posts"),
            F.sum("is_viral_rel").alias("viral_posts"),
            F.first("sub_median_score").alias("median_score"),
            F.avg("score").alias("avg_score"),
        )
        .where(F.col("total_posts") >= MIN_POSTS_PER_SUBREDDIT)
        .withColumn(
            "virality_rate_pct",
            F.round(F.col("viral_posts") / F.col("total_posts") * 100, 3),
        )
        .orderBy(F.desc("virality_rate_pct"))
    )

    print(f"\nTop {TOP_N} subreddits by RELATIVE virality rate")
    print(f"(viral = score >= {REL_MULTIPLIER}x subreddit median)")
    rel_by_sub.show(TOP_N, truncate=False)

    (rel_by_sub.limit(200)
        .coalesce(1)
        .write.mode("overwrite").option("header", True)
        .csv("s3a://nik-datsbd-s2026/results/q1_virality_relative_full"))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total cleaned posts: {total_clean:,}")
    print("Absolute results: s3a://nik-datsbd-s2026/results/q1_virality_absolute/")
    print("Relative results: s3a://nik-datsbd-s2026/results/q1_virality_relative/")

    spark.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
