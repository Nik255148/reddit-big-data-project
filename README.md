# Reddit Virality & Controversy Analysis

**DATS 6450.13 — Big Data Analysis at Scale**
Group 8 Term Project · Spring 2026

---

## Overview

What makes a Reddit post go viral — positivity, negativity, or controversy?
This project analyzes the full Reddit comments and submissions dataset
(June 2023 – July 2024, ~446 GB) to understand how sentiment, content, user
behavior, and timing drive engagement across communities. We use PySpark on
a 4-node EC2 cluster to process billions of rows and build predictive models
for post success and controversy detection.

## Team

- Nikhilesh Narendra Dhavale
- Anushka Rajesh Salvi
- Aditya Karande

## Research Questions

**EDA (Q1–Q4):** Virality rates, temporal patterns, engagement distributions, user history
**NLP (Q5–Q7):** Sentiment vs engagement, vocabulary of viral posts, controversy signal
**ML (Q8–Q10):** Predict comment counts, classify viral posts, classify controversial posts

See `docs/Group_8_Project_Proposal.docx` for the full proposal.

## Infrastructure

- **Cluster:** 1 master + 3 workers on AWS EC2 (t3.large, 100 GB EBS each)
- **Spark:** 3.4.4 on Java 17
- **Python:** 3.10 with PySpark 3.4.4, Spark NLP 5.5.3
- **Storage:** Dataset in private S3 bucket, accessed via IAM instance profile
- **Cluster setup:** see `infrastructure/README.md`

## Repo Structure

- `docs/` — proposal, methodology, writeups
- `infrastructure/` — cluster setup scripts, deployment notes
- `src/common.py` — shared Spark session builder
- `src/eda/` — Q1–Q4 analysis scripts
- `src/nlp/` — Q5–Q7 sentiment & text analysis
- `src/ml/` — Q8–Q10 predictive models
- `notebooks/` — exploratory Jupyter notebooks
- `results/` — CSVs, charts (small aggregates only)
- `scripts/` — one-off helpers

## Running a Job on the Cluster

From the master node:

    cd ~/project/
    spark-submit \
      --master spark://MASTER_PRIVATE_IP:7077 \
      --deploy-mode client \
      --executor-memory 4G \
      --total-executor-cores 6 \
      src/eda/q1_virality_rates.py

Monitor at port 8080 (Master UI) and port 4040 (Application UI) on the master's public IP.

## Dataset

Reddit comments and submissions from June 2023 through July 2024, stored as
Parquet with Hive-style partitioning (yyyy=YYYY/mm=MM/). Approximately 446 GB
compressed, billions of rows when fully read.

Source path (private bucket): `s3a://<bucket>/reddit-data/parquet/`

## License

Academic project — not licensed for redistribution.
