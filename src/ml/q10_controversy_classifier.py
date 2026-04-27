Question-10:

"""
src/ml/q10_controversy_classifier.py

ML Q10: Can we classify a post as “controversial” vs “consensus” based on content and sentiment features? (Binary classification, using the sentiment-variance label from Q7.)


Uses the Reddit built-in `controversiality` field on comments as label.
A post is "controversial" if >= 20% of its comments are marked
controversial by Reddit.

Binary classification using:
  - Logistic Regression
  - Random Forest Classifier

Features: post metadata + sentiment features from Q7.

Run from master:
    spark-submit \
      --master spark://MASTER_PRIVATE_IP:7077 \
      --deploy-mode client \
      --executor-memory 4G --driver-memory 2G --total-executor-cores 6 \
      src/ml/q10_controversy_classifier.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator
)
from common import build_spark, S3_COMMENTS, S3_SUBMISSIONS, DEFAULT_DEV_MONTH

S3_RESULTS = "s3a://nik-datsbd-s2026/results"
CONTROVERSY_RATE_THRESHOLD = 0.20
MIN_COMMENTS = 5
DEV_COMMENT_LIMIT = 1_000_000


def main():
    spark = build_spark("ML_Q10_ControversyClassifier")

    print("=" * 60)
    print("ML Q10: Controversial vs Consensus Post Classifier")
    print("=" * 60)

    # Build controversy label from comments
    comments_path = S3_COMMENTS + DEFAULT_DEV_MONTH
    print(f"Reading comments: {comments_path}")
    comments = (
        spark.read.parquet(comments_path)
        .where(F.col("author") != "[deleted]")
        .where(F.col("body") != "[removed]")
        .select("link_id", "controversiality", "score")
        .limit(DEV_COMMENT_LIMIT)
    )

    # Aggregate per post: controversiality rate
    post_controversy = (
        comments
        .withColumn("post_id",
                    F.regexp_replace(F.col("link_id"), "^t3_", ""))
        .groupBy("post_id")
        .agg(
            F.count("*").alias("comment_count"),
            F.avg("controversiality").alias("controversy_rate"),
            F.avg("score").alias("avg_comment_score"),
        )
        .where(F.col("comment_count") >= MIN_COMMENTS)
        .withColumn("label",
                    (F.col("controversy_rate") >=
                     CONTROVERSY_RATE_THRESHOLD).cast("double"))
    )

    total = post_controversy.count()
    controversial = post_controversy.where(F.col("label") == 1).count()
    print(f"\nPosts with >= {MIN_COMMENTS} comments: {total:,}")
    print(f"Controversial (>= {CONTROVERSY_RATE_THRESHOLD*100:.0f}% "
          f"controversial comments): {controversial:,} "
          f"({controversial/total*100:.1f}%)")

    # Join with submissions for features
    subs_path = S3_SUBMISSIONS + DEFAULT_DEV_MONTH
    submissions = (
        spark.read.parquet(subs_path)
        .where(F.col("author") != "[deleted]")
        .where(~F.col("stickied"))
        .withColumn("ts", F.to_timestamp(F.col("created_utc")))
        .withColumn("hour_utc", F.hour("ts").cast("double"))
        .withColumn("dayofweek", F.dayofweek("ts").cast("double"))
        .withColumn("title_length", F.length("title").cast("double"))
        .withColumn("score_d", F.col("score").cast("double"))
        .withColumn("num_comments_d", F.col("num_comments").cast("double"))
        .withColumn("over_18", F.col("over_18").cast("double"))
        .select("id", "subreddit", "hour_utc", "dayofweek",
                "title_length", "score_d", "num_comments_d", "over_18")
    )

    data = (
        post_controversy
        .join(submissions,
              post_controversy.post_id == submissions.id,
              how="inner")
        .select("label", "controversy_rate", "avg_comment_score",
                "hour_utc", "dayofweek", "title_length",
                "score_d", "num_comments_d", "over_18", "subreddit")
        .na.fill(0.0)
    )

    indexer = StringIndexer(inputCol="subreddit", outputCol="subreddit_idx",
                            handleInvalid="keep")
    assembler = VectorAssembler(
        inputCols=["hour_utc", "dayofweek", "title_length",
                   "score_d", "num_comments_d", "over_18",
                   "avg_comment_score", "subreddit_idx"],
        outputCol="features"
    )
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features",
                            withMean=False, withStd=True)

    train, test = data.randomSplit([0.8, 0.2], seed=42)
    train.cache()
    test.cache()
    print(f"\nTrain: {train.count():,} | Test: {test.count():,}")

    roc_eval = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    f1_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )

    results = []

    print("\n--- Logistic Regression ---")
    lr_pipe = Pipeline(stages=[
        indexer, assembler, scaler,
        LogisticRegression(featuresCol="scaled_features", labelCol="label",
                           maxIter=20, regParam=0.01)
    ])
    lr_model = lr_pipe.fit(train)
    lr_preds = lr_model.transform(test)
    lr_roc = roc_eval.evaluate(lr_preds)
    lr_f1 = f1_eval.evaluate(lr_preds)
    print(f"  ROC-AUC: {lr_roc:.4f} | F1: {lr_f1:.4f}")
    results.append(("LogisticRegression", lr_roc, lr_f1))

    print("\n--- Random Forest ---")
    rf_pipe = Pipeline(stages=[
        indexer, assembler,
        RandomForestClassifier(maxBins=600000, featuresCol="features", labelCol="label",
                               numTrees=50, maxDepth=6, seed=42)
    ])
    rf_model = rf_pipe.fit(train)
    rf_preds = rf_model.transform(test)
    rf_roc = roc_eval.evaluate(rf_preds)
    rf_f1 = f1_eval.evaluate(rf_preds)
    print(f"  ROC-AUC: {rf_roc:.4f} | F1: {rf_f1:.4f}")
    results.append(("RandomForest", rf_roc, rf_f1))

    results_df = spark.createDataFrame(
        results, ["model", "roc_auc", "f1"]
    )
    print("\n--- Model Comparison ---")
    results_df.show(truncate=False)

    (results_df.coalesce(1)
        .write.mode("overwrite").option("header", True)
        .csv(f"{S3_RESULTS}/q10_controversy_classifier_results"))

    print(f"\nResults: {S3_RESULTS}/q10_controversy_classifier_results/")
    spark.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()