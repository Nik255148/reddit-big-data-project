#!/bin/bash
# scripts/run_all_eda.sh
# Run all EDA questions sequentially on the full dataset.
# Run this from the MASTER NODE after syncing code from management.
#
# Usage (on master):
#   bash ~/project/scripts/run_all_eda.sh

set -e

MASTER_IP="172.31.90.218"
SUBMIT="spark-submit --master spark://$MASTER_IP:7077 --deploy-mode client \
  --executor-memory 4G --driver-memory 2G --total-executor-cores 6"

source ~/spark-cluster/.venv/bin/activate
export PYSPARK_PYTHON=/home/ubuntu/spark-cluster/.venv/bin/python
export PYSPARK_DRIVER_PYTHON=/home/ubuntu/spark-cluster/.venv/bin/python

cd ~/project

echo "=============================="
echo "Running Q2: Temporal Patterns"
echo "=============================="
$SUBMIT src/eda/q2_temporal_patterns.py 2>&1 | tee ~/q2_full.log
echo "Q2 done."

echo "=============================="
echo "Running Q3: Engagement Distribution"
echo "=============================="
$SUBMIT src/eda/q3_engagement_distribution.py 2>&1 | tee ~/q3_full.log
echo "Q3 done."

echo "=============================="
echo "Running Q4: User History"
echo "=============================="
$SUBMIT src/eda/q4_user_history.py 2>&1 | tee ~/q4_full.log
echo "Q4 done."

echo "=============================="
echo "ALL EDA COMPLETE"
echo "=============================="
