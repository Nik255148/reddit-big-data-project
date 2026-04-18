#!/bin/bash
# install_worker.sh
# Idempotent worker-node setup for the Spark cluster.
# Installs Java 17, Spark 3.4.4, and AWS JARs. Sets environment variables.
# Run on each worker node from the master via: ssh worker "bash /tmp/install_worker.sh"

set -e

echo "[$(hostname)] Starting worker setup..."

# --- Java 17 (required by Spark 3.4 on Ubuntu 22.04) ---
if ! java -version 2>&1 | grep -q 'openjdk version "17'; then
  echo "[$(hostname)] Installing Java 17..."
  sudo apt-get update -qq
  sudo apt-get install -y -qq openjdk-17-jdk-headless curl wget
fi

# --- Apache Spark 3.4.4 ---
cd ~
if [ ! -d ~/spark ]; then
  echo "[$(hostname)] Downloading Spark 3.4.4..."
  wget -q https://archive.apache.org/dist/spark/spark-3.4.4/spark-3.4.4-bin-hadoop3.tgz
  tar -xzf spark-3.4.4-bin-hadoop3.tgz
  mv spark-3.4.4-bin-hadoop3 spark
  rm spark-3.4.4-bin-hadoop3.tgz
fi

# --- AWS JARs (Hadoop 3.3.4 is bundled with Spark 3.4) ---
cd ~/spark/jars
if [ ! -f hadoop-aws-3.3.4.jar ]; then
  echo "[$(hostname)] Downloading AWS JARs..."
  wget -q https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar
  wget -q https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar
fi

# --- Environment variables (idempotent) ---
if ! grep -q "SPARK_HOME" ~/.bashrc; then
  echo 'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64' >> ~/.bashrc
  echo 'export SPARK_HOME=$HOME/spark' >> ~/.bashrc
  echo 'export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin' >> ~/.bashrc
  echo 'export PS1="[WORKER] \u@\h:\w\$ "' >> ~/.bashrc
fi

echo "[$(hostname)] Worker setup COMPLETE."
