"""
src/nlp/q6_viral_vocabulary.py

NLP Q6: What words and phrases distinguish viral posts from average posts?

Uses Spark MLlib TF-IDF to find the most distinctive vocabulary of
high-engagement vs low-engagement post titles.

Approach:
  1. Split posts into VIRAL (score >= 1000 or comments >= 100) vs AVERAGE
  2. Build TF-IDF on titles for each group
  3. Find top 50 words by TF-IDF score in each group
  4. Compare the vocabulary — which words are unique to viral posts?

Input : 1 month of submissions (DEFAULT_DEV_MONTH).
Output: CSVs written to S3 results path.

Run from master:
    spark-submit \
      --master spark://MASTER_PRIVATE_IP:7077 \
      --deploy-mode client \
      --executor-memory 4G --driver-memory 2G --total-executor-cores 6 \
      src/nlp/q6_viral_vocabulary.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyspark.sql import functions as F
from pyspark.ml.feature import (
    Tokenizer, StopWordsRemover, CountVectorizer, IDF
)
from pyspark.ml import Pipeline
from common import build_spark, S3_SUBMISSIONS, DEFAULT_DEV_MONTH

S3_RESULTS = "s3a://nik-datsbd-s2026/results"
ABS_SCORE_THRESHOLD = 1000
ABS_COMMENTS_THRESHOLD = 100
TOP_N_WORDS = 50


def get_top_tfidf_words(df, vocab, top_n=50):
    """Extract top N words by mean TF-IDF score from a DataFrame."""
    from pyspark.ml.linalg import SparseVector
    import numpy as np

    tfidf_rows = df.select("tfidf_features").limit(10000).collect()
    word_scores = {}
    for row in tfidf_rows:
        vec = row["tfidf_features"]
        if vec is not None:
            for idx, val in zip(vec.indices, vec.values):
                word = vocab[idx]
                word_scores[word] = word_scores.get(word, 0) + val

    sorted_words = sorted(word_scores.items(), key=lambda x: -x[1])
    return sorted_words[:top_n]


def main():
    spark = build_spark("NLP_Q6_ViralVocabulary")

    print("=" * 60)
    print("NLP Q6: TF-IDF Vocabulary of Viral vs Average Posts")
    print("=" * 60)

    path = S3_SUBMISSIONS  # full dataset
    print(f"Reading: {path}")
    df = spark.read.parquet(path)

    clean = (
        df
        .where(F.col("author") != "[deleted]")
        .where(~F.col("stickied"))
        .where(F.col("distinguished").isNull())
        .where(~F.col("locked"))
        .where(~F.col("quarantine"))
        .where(F.col("title").isNotNull())
        .where(F.length(F.col("title")) > 5)
        .withColumn("is_viral",
            ((F.col("score") >= ABS_SCORE_THRESHOLD) |
             (F.col("num_comments") >= ABS_COMMENTS_THRESHOLD)).cast("int"))
        .select("id", "title", "score", "num_comments",
                "subreddit", "is_viral")
    )

    total = clean.count()
    viral_count = clean.where(F.col("is_viral") == 1).count()
    avg_count = total - viral_count
    print(f"\nTotal posts: {total:,}")
    print(f"Viral posts: {viral_count:,} ({viral_count/total*100:.1f}%)")
    print(f"Average posts: {avg_count:,} ({avg_count/total*100:.1f}%)")

    # ------------------------------------------------------------------
    # Build TF-IDF pipeline (Spark MLlib — no Spark NLP needed)
    # ------------------------------------------------------------------
    tokenizer = Tokenizer(inputCol="title", outputCol="words_raw")
    remover = StopWordsRemover(
        inputCol="words_raw", outputCol="words",
        stopWords=StopWordsRemover.loadDefaultStopWords("english")
    )
    cv = CountVectorizer(
        inputCol="words", outputCol="tf_features",
        vocabSize=10000, minDF=5.0
    )
    idf = IDF(inputCol="tf_features", outputCol="tfidf_features")

    pipeline = Pipeline(stages=[tokenizer, remover, cv, idf])

    print("\nFitting TF-IDF pipeline on all posts...")
    model = pipeline.fit(clean)
    vocab = model.stages[2].vocabulary
    print(f"Vocabulary size: {len(vocab):,} words")

    # Transform viral and average subsets separately
    viral_df = model.transform(clean.where(F.col("is_viral") == 1))
    avg_df = model.transform(clean.where(F.col("is_viral") == 0))

    # Get top words for each group
    print("\n--- Top 50 Words in VIRAL posts (by TF-IDF) ---")
    viral_words = get_top_tfidf_words(viral_df, vocab, TOP_N_WORDS)
    for word, score in viral_words[:20]:
        print(f"  {word}: {score:,.1f}")

    print("\n--- Top 50 Words in AVERAGE posts (by TF-IDF) ---")
    avg_words = get_top_tfidf_words(avg_df, vocab, TOP_N_WORDS)
    for word, score in avg_words[:20]:
        print(f"  {word}: {score:,.1f}")

    # Save results as DataFrames
    viral_words_df = spark.createDataFrame(
        [(w, float(s), "viral") for w, s in viral_words],
        ["word", "tfidf_score", "group"]
    )
    avg_words_df = spark.createDataFrame(
        [(w, float(s), "average") for w, s in avg_words],
        ["word", "tfidf_score", "group"]
    )
    combined = viral_words_df.union(avg_words_df)
    (combined.coalesce(1)
        .write.mode("overwrite").option("header", True)
        .csv(f"{S3_RESULTS}/q6_viral_vocabulary_full"))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total posts analyzed: {total:,}")
    print(f"Results: {S3_RESULTS}/q6_viral_vocabulary/")

    spark.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
