"""
src/eda/q3_engagement_distribution.py

EDA Q3: What is the distribution of engagement across posts?
Does it follow a power law (a few posts get most of the attention)?

We compute:
  (a) Score distribution — percentiles + bucket counts
  (b) Comment count distribution — percentiles + bucket counts
  (c) Gini coefficient approximation — measures inequality
  (d) What share of posts account for 50% / 80% / 95% of total score

Input : 1 month of submissions (DEFAULT_DEV_MONTH).
Output: CSVs written to S3 results path.

Run from master:
    spark-submit \
      --master spark://MASTER_PRIVATE_IP:7077 \
      --deploy-mode client \
      --executor-memory 4G --driver-memory 2G --total-executor-cores 6 \
      src/eda/q3_engagement_distribution.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyspark.sql import functions as F
from common import build_spark, S3_SUBMISSIONS, DEFAULT_DEV_MONTH

S3_RESULTS = "s3a://nik-datsbd-s2026/results"


def main():
    spark = build_spark("EDA_Q3_EngagementDist")

    print("=" * 60)
    print("EDA Q3: Engagement Distribution")
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
        .where(F.col("score") >= 0)
    )

    clean.cache()
    total = clean.count()
    print(f"\nTotal cleaned posts: {total:,}")

    # ------------------------------------------------------------------
    # (a) Score percentiles
    # ------------------------------------------------------------------
    print("\n--- Score Percentiles ---")
    percentiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
    score_percentiles = clean.select(
        F.percentile_approx("score", percentiles).alias("score_pcts"),
        F.percentile_approx("num_comments", percentiles).alias("comments_pcts"),
        F.avg("score").alias("avg_score"),
        F.avg("num_comments").alias("avg_comments"),
        F.max("score").alias("max_score"),
        F.max("num_comments").alias("max_comments"),
        F.sum("score").alias("total_score"),
    ).collect()[0]

    pct_labels = ["p10", "p25", "p50", "p75", "p90", "p95", "p99", "p99.9"]
    print("\nScore percentiles:")
    for label, val in zip(pct_labels, score_percentiles["score_pcts"]):
        print(f"  {label}: {val:,.0f}")
    print(f"  avg: {score_percentiles['avg_score']:,.1f}")
    print(f"  max: {score_percentiles['max_score']:,.0f}")

    print("\nComment count percentiles:")
    for label, val in zip(pct_labels, score_percentiles["comments_pcts"]):
        print(f"  {label}: {val:,.0f}")
    print(f"  avg: {score_percentiles['avg_comments']:,.1f}")
    print(f"  max: {score_percentiles['max_comments']:,.0f}")

    # ------------------------------------------------------------------
    # (b) Score buckets (log scale — good for power law visualization)
    # ------------------------------------------------------------------
    print("\n--- Score Buckets (log scale) ---")
    bucketed = clean.withColumn(
        "score_bucket",
        F.when(F.col("score") == 0, "0")
         .when(F.col("score") <= 1, "1")
         .when(F.col("score") <= 5, "2-5")
         .when(F.col("score") <= 10, "6-10")
         .when(F.col("score") <= 50, "11-50")
         .when(F.col("score") <= 100, "51-100")
         .when(F.col("score") <= 500, "101-500")
         .when(F.col("score") <= 1000, "501-1000")
         .when(F.col("score") <= 5000, "1001-5000")
         .when(F.col("score") <= 10000, "5001-10000")
         .otherwise("10001+")
    )

    score_dist = (
        bucketed.groupBy("score_bucket")
        .agg(
            F.count("*").alias("post_count"),
            F.sum("score").alias("total_score"),
        )
        .withColumn("pct_of_posts",
                    F.round(F.col("post_count") / total * 100, 3))
        .withColumn("pct_of_total_score",
                    F.round(F.col("total_score") /
                            score_percentiles["total_score"] * 100, 3))
    )
    score_dist.show(20, truncate=False)

    (score_dist.coalesce(1)
        .write.mode("overwrite").option("header", True)
        .csv(f"{S3_RESULTS}/q3_score_distribution"))

    # ------------------------------------------------------------------
    # (c) Power law check: what % of posts generate X% of total score?
    # ------------------------------------------------------------------
    print("\n--- Power Law: Top N% of posts → % of total score ---")
    total_score = score_percentiles["total_score"]

    thresholds = [1000, 5000, 10000, 50000]
    rows = []
    for threshold in thresholds:
        above = clean.where(F.col("score") >= threshold)
        count_above = above.count()
        score_above = above.agg(F.sum("score")).collect()[0][0] or 0
        pct_posts = round(count_above / total * 100, 4)
        pct_score = round(score_above / total_score * 100, 2)
        print(f"  Posts with score >= {threshold:>6,}: "
              f"{count_above:>8,} ({pct_posts:.3f}% of posts) "
              f"→ {pct_score:.1f}% of total score")
        rows.append((threshold, count_above, pct_posts, score_above, pct_score))

    power_law_df = spark.createDataFrame(
        rows,
        ["score_threshold", "post_count", "pct_of_posts",
         "total_score_above", "pct_of_total_score"]
    )
    (power_law_df.coalesce(1)
        .write.mode("overwrite").option("header", True)
        .csv(f"{S3_RESULTS}/q3_power_law"))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total cleaned posts: {total:,}")
    print(f"Results: {S3_RESULTS}/q3_score_distribution/")
    print(f"Results: {S3_RESULTS}/q3_power_law/")

    spark.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
