"""
src/nlp/q7_controversy_signal.py

NLP Q7: Is there a "controversy signal" in the language itself?

We measure sentiment variance across comments on each post.
Posts where commenters strongly disagree (high sentiment variance)
are labeled "controversial"; low-variance posts are "consensus".

Approach:
  1. Read comments, run Vivekn sentiment on comment bodies
  2. Map sentiment to numeric: positive=1, negative=0
  3. For each post (link_id), compute:
     - mean sentiment (overall tone)
     - variance of sentiment (disagreement signal)
     - comment count
  4. Join back to submissions to get post metadata
  5. Compare controversial vs consensus posts:
     - avg score, avg num_comments
     - distribution across subreddits

Input : 1 FULL month of COMMENTS (DEFAULT_DEV_MONTH).
Output: CSVs written to S3 results path.

Run from master:
    spark-submit \
      --master spark://MASTER_PRIVATE_IP:7077 \
      --deploy-mode client \
      --executor-memory 4G --driver-memory 2G --total-executor-cores 6 \
      --packages com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.3 \
      src/nlp/q7_controversy_signal.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyspark.sql import functions as F
import sparknlp
from sparknlp.base import DocumentAssembler, Pipeline
from sparknlp.annotator import Tokenizer, Normalizer, ViveknSentimentModel
from common import build_spark, S3_COMMENTS, S3_SUBMISSIONS, DEFAULT_DEV_MONTH

S3_RESULTS = "s3a://aditya-s3-bd/results"
MIN_COMMENTS_PER_POST = 5
CONTROVERSY_THRESHOLD = 0.25


def main():
    spark = build_spark("NLP_Q7_Controversy")
    sparknlp.start()

    print("=" * 60)
    print("NLP Q7: Controversy Signal via Sentiment Variance")
    print(f"Running on FULL month: {DEFAULT_DEV_MONTH}")
    print("=" * 60)

    # Load comments (full month, no limit)
    comments_path = S3_COMMENTS + DEFAULT_DEV_MONTH
    print(f"Reading comments: {comments_path}")
    comments = (
        spark.read.parquet(comments_path)
        .where(F.col("body") != "[deleted]")
        .where(F.col("body") != "[removed]")
        .where(F.col("author") != "[deleted]")
        .where(F.col("body").isNotNull())
        .where(F.length(F.col("body")) > 10)
        .select("id", "link_id", "body", "score")
    )

    total_comments = comments.count()
    print(f"Total comments to process: {total_comments:,}")

    # ------------------------------------------------------------------
    # Spark NLP sentiment pipeline on comment bodies
    # ------------------------------------------------------------------
    document_assembler = (
        DocumentAssembler()
        .setInputCol("body")
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
    vivekn = (
        ViveknSentimentModel.pretrained()
        .setInputCols(["document", "normal"])
        .setOutputCol("sentiment")
    )

    pipeline = Pipeline(stages=[document_assembler, tokenizer,
                                 normalizer, vivekn])

    print("\nRunning sentiment pipeline on comments...")
    model = pipeline.fit(comments)
    result = model.transform(comments)

    # Extract sentiment as numeric (positive=1, negative=0)
    result = (
        result
        .withColumn("sentiment_label",
                    F.col("sentiment").getItem(0).getField("result"))
        .withColumn("sentiment_num",
                    F.when(F.col("sentiment_label") == "positive", 1.0)
                     .otherwise(0.0))
        .withColumn("post_id",
                    F.regexp_replace(F.col("link_id"), "^t3_", ""))
    )

    # ------------------------------------------------------------------
    # Compute sentiment variance per post
    # ------------------------------------------------------------------
    print("\n--- Computing sentiment variance per post ---")
    post_sentiment = (
        result.groupBy("post_id")
        .agg(
            F.count("*").alias("comment_count"),
            F.avg("sentiment_num").alias("mean_sentiment"),
            F.variance("sentiment_num").alias("sentiment_variance"),
            F.avg("score").alias("avg_comment_score"),
        )
        .where(F.col("comment_count") >= MIN_COMMENTS_PER_POST)
        .withColumn("is_controversial",
                    (F.col("sentiment_variance") >=
                     CONTROVERSY_THRESHOLD).cast("int"))
    )

    controversial_count = post_sentiment.where(
        F.col("is_controversial") == 1).count()
    total_posts = post_sentiment.count()
    print(f"\nPosts with >= {MIN_COMMENTS_PER_POST} comments: {total_posts:,}")
    print(f"Controversial posts (variance >= {CONTROVERSY_THRESHOLD}): "
          f"{controversial_count:,} ({controversial_count/total_posts*100:.1f}%)")

    # ------------------------------------------------------------------
    # Join with submissions to get post metadata
    # ------------------------------------------------------------------
    print("\n--- Joining with submissions ---")
    subs_path = S3_SUBMISSIONS + DEFAULT_DEV_MONTH
    submissions = (
        spark.read.parquet(subs_path)
        .select("id", "title", "score", "num_comments",
                "subreddit", "author")
    )

    joined = post_sentiment.join(submissions,
                                  post_sentiment.post_id == submissions.id,
                                  how="inner")

    # ------------------------------------------------------------------
    # Compare controversial vs consensus posts
    # ------------------------------------------------------------------
    print("\n--- Controversial vs Consensus Posts ---")
    comparison = (
        joined.groupBy("is_controversial")
        .agg(
            F.count("*").alias("post_count"),
            F.avg("score").alias("avg_score"),
            F.avg("num_comments").alias("avg_num_comments"),
            F.avg("sentiment_variance").alias("avg_sentiment_variance"),
            F.avg("mean_sentiment").alias("avg_mean_sentiment"),
        )
        .withColumn("avg_score", F.round("avg_score", 2))
        .withColumn("avg_num_comments", F.round("avg_num_comments", 2))
        .withColumn("avg_sentiment_variance",
                    F.round("avg_sentiment_variance", 4))
        .orderBy("is_controversial")
    )
    comparison.show(truncate=False)

    (comparison.coalesce(1)
        .write.mode("overwrite").option("header", True)
        .csv(f"{S3_RESULTS}/q7_controversy_comparison"))

    # Top controversial subreddits
    print("\n--- Most Controversial Subreddits ---")
    controversial_subs = (
        joined.where(F.col("is_controversial") == 1)
        .groupBy("subreddit")
        .agg(
            F.count("*").alias("controversial_posts"),
            F.avg("sentiment_variance").alias("avg_variance"),
            F.avg("score").alias("avg_score"),
        )
        .where(F.col("controversial_posts") >= 5)
        .withColumn("avg_variance", F.round("avg_variance", 4))
        .orderBy(F.desc("controversial_posts"))
        .limit(30)
    )
    controversial_subs.show(truncate=False)

    (controversial_subs.coalesce(1)
        .write.mode("overwrite").option("header", True)
        .csv(f"{S3_RESULTS}/q7_controversial_subreddits"))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Comments analyzed: {total_comments:,}")
    print(f"Posts with enough comments: {total_posts:,}")
    print(f"Controversial posts: {controversial_count:,}")
    print(f"Results: {S3_RESULTS}/q7_controversy_comparison/")
    print(f"Results: {S3_RESULTS}/q7_controversial_subreddits/")

    spark.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()