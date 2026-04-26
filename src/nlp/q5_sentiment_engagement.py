"""
src/nlp/q5_sentiment_engagement.py

NLP Q5: How does post sentiment correlate with engagement?
Do negative posts actually receive more comments?

Uses Spark NLP (John Snow Labs) pretrained Vivekn sentiment pipeline
to classify each submission title as positive or negative.
Then computes avg score, avg comments, and viral rate per sentiment class.

Input : 1 month of submissions (DEFAULT_DEV_MONTH), limited to 200K posts
        for speed on the dev run.
Output: CSV written to S3 results path.

Run from master:
    spark-submit \
      --master spark://MASTER_PRIVATE_IP:7077 \
      --deploy-mode client \
      --executor-memory 4G --driver-memory 2G --total-executor-cores 6 \
      --packages com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.3 \
      src/nlp/q5_sentiment_engagement.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyspark.sql import functions as F
import sparknlp
from sparknlp.base import DocumentAssembler, Pipeline
from sparknlp.annotator import (
    Tokenizer,
    Normalizer,
    StopWordsCleaner,
    ViveknSentimentModel,
)
from common import build_spark, S3_SUBMISSIONS, DEFAULT_DEV_MONTH

S3_RESULTS = "s3a://nik-datsbd-s2026/results"
ABS_SCORE_THRESHOLD = 1000
ABS_COMMENTS_THRESHOLD = 100
DEV_SAMPLE_LIMIT = 1_000_000


def main():
    spark = build_spark("NLP_Q5_Sentiment")
    sparknlp.start()

    print("=" * 60)
    print("NLP Q5: Sentiment vs Engagement")
    print("=" * 60)

    path = S3_SUBMISSIONS  # full dataset
    print(f"Reading: {path}")
    df = spark.read.parquet(path)

    # Clean + filter + sample for dev run
    clean = (
        df
        .where(F.col("author") != "[deleted]")
        .where(~F.col("stickied"))
        .where(F.col("distinguished").isNull())
        .where(~F.col("locked"))
        .where(~F.col("quarantine"))
        .where(F.col("title").isNotNull())
        .where(F.length(F.col("title")) > 10)
        .withColumn("is_viral",
            ((F.col("score") >= ABS_SCORE_THRESHOLD) |
             (F.col("num_comments") >= ABS_COMMENTS_THRESHOLD)).cast("int"))
        .select("id", "title", "score", "num_comments",
                "subreddit", "is_viral")
        .limit(DEV_SAMPLE_LIMIT)
    )

    print(f"\nSample size for dev run: {DEV_SAMPLE_LIMIT:,}")

    # ------------------------------------------------------------------
    # Build Spark NLP sentiment pipeline
    # ------------------------------------------------------------------
    document_assembler = (
        DocumentAssembler()
        .setInputCol("title")
        .setOutputCol("document")
    )

    tokenizer = (
        Tokenizer()
        .setInputCols(["document"])
        .setOutputCol("token")
    )

    normalizer = (
        Normalizer()
        .setInputCols(["token"])
        .setOutputCol("normal")
        .setLowercase(True)
    )

    vivekn_sentiment = (
        ViveknSentimentModel.pretrained()
        .setInputCols(["document", "normal"])
        .setOutputCol("sentiment")
    )

    pipeline = Pipeline(stages=[
        document_assembler,
        tokenizer,
        normalizer,
        vivekn_sentiment,
    ])

    print("\nRunning sentiment pipeline...")
    model = pipeline.fit(clean)
    result = model.transform(clean)

    # Extract sentiment label (positive/negative)
    result = result.withColumn(
        "sentiment_label",
        F.col("sentiment").getItem(0).getField("result")
    )

    # ------------------------------------------------------------------
    # Aggregate by sentiment
    # ------------------------------------------------------------------
    print("\n--- Sentiment vs Engagement ---")
    by_sentiment = (
        result.groupBy("sentiment_label")
        .agg(
            F.count("*").alias("post_count"),
            F.avg("score").alias("avg_score"),
            F.avg("num_comments").alias("avg_comments"),
            F.sum("is_viral").alias("viral_posts"),
        )
        .withColumn("viral_rate_pct",
                    F.round(F.col("viral_posts") / F.col("post_count") * 100, 3))
        .withColumn("avg_score", F.round("avg_score", 2))
        .withColumn("avg_comments", F.round("avg_comments", 2))
        .orderBy("sentiment_label")
    )

    by_sentiment.show(truncate=False)

    (by_sentiment.coalesce(1)
        .write.mode("overwrite").option("header", True)
        .csv(f"{S3_RESULTS}/q5_sentiment_engagement_full"))

    # ------------------------------------------------------------------
    # Bonus: top subreddits by sentiment mix
    # ------------------------------------------------------------------
    print("\n--- Sentiment Mix by Subreddit (top 20 by post count) ---")
    by_sub_sentiment = (
        result.groupBy("subreddit", "sentiment_label")
        .agg(F.count("*").alias("post_count"))
        .orderBy(F.desc("post_count"))
        .limit(40)
    )
    by_sub_sentiment.show(40, truncate=False)

    (by_sub_sentiment.coalesce(1)
        .write.mode("overwrite").option("header", True)
        .csv(f"{S3_RESULTS}/q5_sentiment_by_subreddit_full"))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Sample analyzed: {DEV_SAMPLE_LIMIT:,} posts")
    print(f"Results: {S3_RESULTS}/q5_sentiment_engagement/")
    print(f"Results: {S3_RESULTS}/q5_sentiment_by_subreddit/")

    spark.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
