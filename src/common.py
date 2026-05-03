"""
src/common.py

Shared helpers for all EDA, NLP, and ML scripts in this project.

Usage in any question script:

    from common import build_spark, S3_SUBMISSIONS, S3_COMMENTS

    spark = build_spark("EDA_Q1_Virality")
    df = spark.read.parquet(S3_SUBMISSIONS + "yyyy=2023/mm=06/")
"""
from pyspark.sql import SparkSession


# S3 paths — single source of truth for dataset locations
S3_BUCKET = "s3a://aditya-s3-bd"
S3_SUBMISSIONS = f"{S3_BUCKET}/reddit-data/parquet/submissions/"
S3_COMMENTS = f"{S3_BUCKET}/reddit-data/parquet/comments/"

# Default month for exploratory / single-month development work
DEFAULT_DEV_MONTH = "yyyy=2023/mm=06/"


def build_spark(app_name: str) -> SparkSession:
    """
    Build a SparkSession configured for our cluster + S3 access.
    Uses the IAM instance profile for S3 credentials (no keys in code).
    """
    spark = (
        SparkSession.builder
            .appName(app_name)
            .config(
                "spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.InstanceProfileCredentialsProvider",
            )
            .config(
                "spark.hadoop.fs.s3a.impl",
                "org.apache.hadoop.fs.s3a.S3AFileSystem",
            )
            .config("spark.sql.parquet.enableVectorizedReader", "true")
            .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# Where question scripts write their aggregated results on master
RESULTS_DIR = "/home/ubuntu/project/results"
