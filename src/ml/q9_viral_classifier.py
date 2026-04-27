Question-9:

"""
src/ml/q9_viral_classifier.py

ML Q9: Can we classify a post as “viral” vs “not viral” using only early features (title, author, timing)? (Binary classification)


Binary classification using:
  - Logistic Regression
  - Random Forest Classifier

Label: is_viral = 1 if score >= 1000 OR num_comments >= 100
Features: same as Q8 regression

Evaluated using Accuracy, F1, ROC-AUC.
Class imbalance handled via class weighting.

Run from master:
    spark-submit \
      --master spark://MASTER_PRIVATE_IP:7077 \
      --deploy-mode client \
      --executor-memory 4G --driver-memory 2G --total-executor-cores 6 \
      src/ml/q9_viral_classifier.py
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
from common import build_spark, S3_SUBMISSIONS, DEFAULT_DEV_MONTH

S3_RESULTS = "s3a://nik-datsbd-s2026/results"
ABS_SCORE_THRESHOLD = 1000
ABS_COMMENTS_THRESHOLD = 100


def main():
    spark = build_spark("ML_Q9_ViralClassifier")

    print("=" * 60)
    print("ML Q9: Viral vs Not-Viral Post Classifier")
    print("=" * 60)

    path = S3_SUBMISSIONS + DEFAULT_DEV_MONTH
    print(f"Reading: {path}")
    df = spark.read.parquet(path)

    data = (
        df
        .where(F.col("author") != "[deleted]")
        .where(~F.col("stickied"))
        .where(F.col("distinguished").isNull())
        .where(~F.col("locked"))
        .where(~F.col("quarantine"))
        .withColumn("ts", F.to_timestamp(F.col("created_utc")))
        .withColumn("hour_utc", F.hour("ts").cast("double"))
        .withColumn("dayofweek", F.dayofweek("ts").cast("double"))
        .withColumn("title_length", F.length("title").cast("double"))
        .withColumn("body_length",
                    F.when(F.col("selftext").isNull(), 0.0)
                     .otherwise(F.length("selftext").cast("double")))
        .withColumn("is_self", F.col("is_self").cast("double"))
        .withColumn("over_18", F.col("over_18").cast("double"))
        .withColumn("label",
                    ((F.col("score") >= ABS_SCORE_THRESHOLD) |
                     (F.col("num_comments") >= ABS_COMMENTS_THRESHOLD))
                    .cast("double"))
        .select("label", "hour_utc", "dayofweek", "title_length",
                "body_length", "is_self", "over_18", "subreddit")
        .na.fill(0.0)
    )

    # Class balance check
    total = data.count()
    viral = data.where(F.col("label") == 1).count()
    print(f"\nTotal: {total:,} | Viral: {viral:,} ({viral/total*100:.1f}%)")

    # Class weights to handle imbalance
    ratio = (total - viral) / viral
    data = data.withColumn(
        "classWeight",
        F.when(F.col("label") == 1, ratio).otherwise(1.0)
    )

    indexer = StringIndexer(inputCol="subreddit", outputCol="subreddit_idx",
                            handleInvalid="keep")
    assembler = VectorAssembler(
        inputCols=["hour_utc", "dayofweek", "title_length",
                   "body_length", "is_self", "over_18", "subreddit_idx"],
        outputCol="features"
    )
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features",
                            withMean=False, withStd=True)

    train, test = data.randomSplit([0.8, 0.2], seed=42)
    train.cache()
    test.cache()
    print(f"Train: {train.count():,} | Test: {test.count():,}")

    roc_eval = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    f1_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )
    acc_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )

    results = []

    # --- Logistic Regression ---
    print("\n--- Logistic Regression ---")
    lr_pipe = Pipeline(stages=[
        indexer, assembler, scaler,
        LogisticRegression(
            featuresCol="scaled_features", labelCol="label",
            weightCol="classWeight", maxIter=20, regParam=0.01
        )
    ])
    lr_model = lr_pipe.fit(train)
    lr_preds = lr_model.transform(test)
    lr_roc = roc_eval.evaluate(lr_preds)
    lr_f1 = f1_eval.evaluate(lr_preds)
    lr_acc = acc_eval.evaluate(lr_preds)
    print(f"  ROC-AUC: {lr_roc:.4f} | F1: {lr_f1:.4f} | Acc: {lr_acc:.4f}")
    results.append(("LogisticRegression", lr_roc, lr_f1, lr_acc))

    # --- Random Forest ---
    print("\n--- Random Forest Classifier ---")
    rf_pipe = Pipeline(stages=[
        indexer, assembler,
        RandomForestClassifier(maxBins=600000, 
            featuresCol="features", labelCol="label",
            numTrees=50, maxDepth=6, seed=42
        )
    ])
    rf_model = rf_pipe.fit(train)
    rf_preds = rf_model.transform(test)
    rf_roc = roc_eval.evaluate(rf_preds)
    rf_f1 = f1_eval.evaluate(rf_preds)
    rf_acc = acc_eval.evaluate(rf_preds)
    print(f"  ROC-AUC: {rf_roc:.4f} | F1: {rf_f1:.4f} | Acc: {rf_acc:.4f}")
    results.append(("RandomForest", rf_roc, rf_f1, rf_acc))

    results_df = spark.createDataFrame(
        results, ["model", "roc_auc", "f1", "accuracy"]
    )
    print("\n--- Model Comparison ---")
    results_df.show(truncate=False)

    (results_df.coalesce(1)
        .write.mode("overwrite").option("header", True)
        .csv(f"{S3_RESULTS}/q9_viral_classifier_results"))

    print(f"\nResults: {S3_RESULTS}/q9_viral_classifier_results/")
    spark.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()