#!/bin/bash
# scripts/sync_to_master.sh
#
# Sync the src/ directory from this git repo to the master node's ~/project/.
# Run this from the management instance any time you change code that
# needs to be re-run on the cluster.
#
# Usage:
#   ./scripts/sync_to_master.sh
#
# Requires cluster-config.txt to be populated by setup-spark-cluster.sh
# (lives in ~/dats6450/term-project/spark-cluster-setup/ on this instance).

set -e

CLUSTER_CONFIG="$HOME/dats6450/term-project/spark-cluster-setup/cluster-config.txt"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -f "$CLUSTER_CONFIG" ]; then
  echo "ERROR: cluster-config.txt not found at $CLUSTER_CONFIG" >&2
  echo "Run setup-spark-cluster.sh first, or update the path in this script." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CLUSTER_CONFIG"

echo "Syncing $REPO_ROOT/src/ to ubuntu@$MASTER_PUBLIC_IP:~/project/src/"

# Make sure ~/project exists on master
ssh -i "$HOME/dats6450/term-project/spark-cluster-setup/$KEY_FILE" \
    -o StrictHostKeyChecking=no \
    "ubuntu@$MASTER_PUBLIC_IP" \
    "mkdir -p ~/project/src ~/project/results"

ssh -i "$HOME/dats6450/term-project/spark-cluster-setup/$KEY_FILE" \
    -o StrictHostKeyChecking=no \
    "ubuntu@$MASTER_PUBLIC_IP" \
    "mkdir -p ~/project/src ~/project/results ~/project/scripts"

# Copy the whole src/ tree, preserving structure
scp -i "$HOME/dats6450/term-project/spark-cluster-setup/$KEY_FILE" \
    -o StrictHostKeyChecking=no \
    -r "$REPO_ROOT/src/." \
    "ubuntu@$MASTER_PUBLIC_IP:~/project/src/"

scp -i "$HOME/dats6450/term-project/spark-cluster-setup/$KEY_FILE" \
    -o StrictHostKeyChecking=no \
    -r "$REPO_ROOT/scripts/." \
    "ubuntu@$MASTER_PUBLIC_IP:~/project/scripts/"

echo "Sync complete."
echo ""
echo "On master, run scripts with e.g.:"
echo "  cd ~/project && spark-submit --master spark://\$MASTER_PRIVATE_IP:7077 src/eda/q1_virality_rates.py"
