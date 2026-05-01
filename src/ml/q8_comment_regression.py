"""
src/ml/q8_comment_regression.py

ML Q8: Can we predict the number of comments a post will receive?

Uses Spark MLlib regression models:
  - Linear Regression (baseline)
  - Random Forest Regressor
  - Gradient Boosted Trees

Features:
  - score, hour_utc, dayofweek, title_length, body_length,
    is_self, over_18, subreddit (indexed)

Evaluated using RMSE and R².

Run from master:
    spark-submit \
      --master spark://MASTER_PRIVATE_IP:7077 \
      --deploy-mode client \
      --executor-memory 4G --driver-memory 2G --total-executor-cores 6 \
      src/ml/q8_comment_regression.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, VectorAssembler, StandardScaler
)
from pyspark.ml.regression import (
    LinearRegression, RandomForestRegressor, GBTRegressor
)
from pyspark.ml.evaluation import RegressionEvaluator
from common import build_spark, S3_SUBMISSIONS, DEFAULT_DEV_MONTH

S3_RESULTS = "s3a://nik-datsbd-s2026/results"


def main():
    spark = build_spark("ML_Q8_CommentRegression")

    print("=" * 60)
    print("ML Q8: Predict Number of Comments (Regression)")
    print("=" * 60)

    path = S3_SUBMISSIONS  # full dataset
    print(f"Reading: {path}")
    df = spark.read.parquet(path)

    # Feature engineering
    data = (
        df
        .where(F.col("author") != "[deleted]")
        .where(~F.col("stickied"))
        .where(F.col("distinguished").isNull())
        .where(~F.col("locked"))
        .where(~F.col("quarantine"))
        .where(F.col("num_comments") >= 0)
        .where(F.col("score") >= 0)
        .withColumn("ts", F.to_timestamp(F.col("created_utc")))
        .withColumn("hour_utc", F.hour("ts").cast("double"))
        .withColumn("dayofweek", F.dayofweek("ts").cast("double"))
        .withColumn("title_length",
                    F.length(F.col("title")).cast("double"))
        .withColumn("body_length",
                    F.when(F.col("selftext").isNull(), 0.0)
                     .otherwise(F.length(F.col("selftext")).cast("double")))
        .withColumn("is_self", F.col("is_self").cast("double"))
        .withColumn("over_18", F.col("over_18").cast("double"))
        .withColumn("score_d", F.col("score").cast("double"))
        .withColumn("label", F.col("num_comments").cast("double"))
        .select("label", "score_d", "hour_utc", "dayofweek",
                "title_length", "body_length", "is_self",
                "over_18", "subreddit")
        .sample(fraction=0.02, seed=42)  # ~10M sample for ML
        .sample(fraction=0.02, seed=42)  # ~10M sample for ML
        .na.fill(0.0)
    )

    # Index subreddit
    indexer = StringIndexer(
        inputCol="subreddit", outputCol="subreddit_idx",
        handleInvalid="keep"
    )

    assembler = VectorAssembler(
        inputCols=["score_d", "hour_utc", "dayofweek",
                   "title_length", "body_length", "is_self",
                   "over_18"],
        outputCol="features"
    )

    scaler = StandardScaler(
        inputCol="features", outputCol="scaled_features",
        withMean=False, withStd=True
    )

    # Train/test split
    train, test = data.randomSplit([0.8, 0.2], seed=42)
    train.cache()
    test.cache()
    print(f"\nTrain: {train.count():,} | Test: {test.count():,}")

    evaluator_rmse = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse"
    )
    evaluator_r2 = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="r2"
    )

    results = []

    # --- Linear Regression ---
    print("\n--- Linear Regression ---")
    lr_pipeline = Pipeline(stages=[
        indexer, assembler, scaler,
        LinearRegression(
            featuresCol="scaled_features", labelCol="label",
            maxIter=20, regParam=0.1
        )
    ])
    lr_model = lr_pipeline.fit(train)
    lr_preds = lr_model.transform(test)
    lr_rmse = evaluator_rmse.evaluate(lr_preds)
    lr_r2 = evaluator_r2.evaluate(lr_preds)
    print(f"  RMSE: {lr_rmse:.4f} | R²: {lr_r2:.4f}")
    results.append(("LinearRegression", lr_rmse, lr_r2))

    # --- Random Forest ---
    print("\n--- Random Forest Regressor ---")
    rf_pipeline = Pipeline(stages=[
        indexer, assembler,
        RandomForestRegressor(
            featuresCol="features", labelCol="label",
            numTrees=50, maxDepth=6, seed=42
        )
    ])
    rf_model = rf_pipeline.fit(train)
    rf_preds = rf_model.transform(test)
    rf_rmse = evaluator_rmse.evaluate(rf_preds)
    rf_r2 = evaluator_r2.evaluate(rf_preds)
    print(f"  RMSE: {rf_rmse:.4f} | R²: {rf_r2:.4f}")
    results.append(("RandomForest", rf_rmse, rf_r2))

    # --- GBT ---
    print("\n--- Gradient Boosted Trees ---")
    gbt_pipeline = Pipeline(stages=[
        indexer, assembler,
        GBTRegressor(
            featuresCol="features", labelCol="label",
            maxIter=30, maxDepth=5, seed=42
        )
    ])
    gbt_model = gbt_pipeline.fit(train)
    gbt_preds = gbt_model.transform(test)
    gbt_rmse = evaluator_rmse.evaluate(gbt_preds)
    gbt_r2 = evaluator_r2.evaluate(gbt_preds)
    print(f"  RMSE: {gbt_rmse:.4f} | R²: {gbt_r2:.4f}")
    results.append(("GBT", gbt_rmse, gbt_r2))

    # Save results
    results_df = spark.createDataFrame(
        results, ["model", "rmse", "r2"]
    )
    print("\n--- Model Comparison ---")
    results_df.show(truncate=False)

    (results_df.coalesce(1)
        .write.mode("overwrite").option("header", True)
        .csv(f"{S3_RESULTS}/q8_regression_results_full"))

    # Feature importance from RF
    print("\n--- Feature Importance (Random Forest) ---")
    rf_stage = rf_model.stages[-1]
    feature_names = ["score", "hour_utc", "dayofweek",
                     "title_length", "body_length",
                     "is_self", "over_18"]
    importances = list(zip(feature_names,
                           rf_stage.featureImportances.toArray()))
    importances.sort(key=lambda x: -x[1])
    for name, imp in importances:
        print(f"  {name}: {imp:.4f}")

    importance_df = spark.createDataFrame(
        [(n, float(i)) for n, i in importances],
        ["feature", "importance"]
    )
    (importance_df.coalesce(1)
        .write.mode("overwrite").option("header", True)
        .csv(f"{S3_RESULTS}/q8_feature_importance_full"))

    print(f"\nResults: {S3_RESULTS}/q8_regression_results/")
    spark.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
