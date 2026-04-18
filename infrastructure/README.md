# Infrastructure: Spark Cluster on AWS EC2

This directory contains the scripts and docs for deploying the Spark cluster used in this project.

## Cluster Topology

- **1 management instance** (external to the cluster) — where you develop, manage the cluster, and push to git
- **1 master node** — runs Spark Master daemon, coordinates workers
- **3 worker nodes** — run Spark Worker daemons, execute tasks

All nodes: `t3.large` (2 vCPU, 8 GB RAM), 100 GB gp3 EBS, Ubuntu 22.04.

| Component | Version |
| --- | --- |
| Apache Spark | 3.4.4 |
| Hadoop | 3.3.4 (bundled with Spark) |
| Java | OpenJDK 17 |
| Python | 3.10 |
| PySpark | 3.4.4 |
| Spark NLP | 5.5.3 |
| hadoop-aws | 3.3.4 |
| aws-java-sdk-bundle | 1.12.262 |

## Prerequisites

Before running any scripts, the management EC2 instance must have:

- AWS CLI configured with credentials that can create/terminate EC2 instances, security groups, and key pairs
- `jq` installed (`sudo apt-get install -y jq`)
- An IAM instance profile named `LabInstanceProfile` that grants S3 read access — this is attached to cluster nodes so PySpark can read from S3 without embedded credentials
- The Reddit dataset pre-copied to a private S3 bucket

## Files in This Directory

### `setup-spark-cluster.sh`

Automated cluster launcher. Creates a security group, SSH key pair, launches 4 EC2 instances, installs Java/Python/Spark/uv on each, sets up passwordless SSH between master and workers, and starts the Spark daemons.

**Usage:**

    ./setup-spark-cluster.sh LAPTOP_PUBLIC_IP

The laptop IP is used to open SSH and the Spark Web UI only to your machine (security group ingress rule).

**Duration:** 10–15 minutes.

**Output files** (all `.gitignore`d — contain IPs and key filenames):

- `cluster-config.txt` — resource IDs, public IPs, key filename
- `cluster-ips.txt` — master and worker private IPs
- `spark-cluster-key-TIMESTAMP.pem` — SSH private key for cluster access

### `install_worker.sh`

Idempotent worker-node setup script. Installs Java 17, Spark 3.4.4, and AWS JARs. Used as a fallback when the automated installer fails partway through (see Known Issues below).

Typical usage from the master node:

    scp install_worker.sh ubuntu@WORKER_IP:/tmp/
    ssh ubuntu@WORKER_IP "bash /tmp/install_worker.sh"

## Known Issues and Fixes

The original lab version of `setup-spark-cluster.sh` had two bugs that broke our first run. The version committed here has both fixed.

### Bug 1: Stale Apache Download URL

The script pointed at `https://downloads.apache.org/spark/spark-3.4.4/...` which returns 404. Apache purges older Spark versions from `downloads` once a newer release ships. The fix: use `https://archive.apache.org/dist/spark/spark-3.4.4/...` instead, which keeps all historical versions. Both occurrences in the script are patched.

### Bug 2: Version Mismatch Between PySpark and Spark

The original script installed PySpark 4.0.0 via pip while downloading Spark 3.5.1 binaries, which is inconsistent. We pinned both to Spark 3.4.4 because Spark NLP 5.x requires Spark 3.x, not 4.x. The `cluster-files/pyproject.toml` dependencies were also updated: `pyspark>=3.4.0,<3.5.0` and `spark-nlp>=5.5.0,<6.0.0`.

### hadoop-aws JAR version

Spark 3.4 bundles Hadoop 3.3.4. The AWS JARs installed must match this version, or you get classpath conflicts. We use `hadoop-aws-3.3.4.jar` (not 3.4.1 as the original script had).

## Deploying the Cluster from Scratch

From the management instance:

1. `cd infrastructure/`
2. Run `./setup-spark-cluster.sh <YOUR_LAPTOP_IP>`
3. Wait 10–15 minutes for the script to finish
4. SSH to master: `source cluster-config.txt && ssh -i $KEY_FILE ubuntu@$MASTER_PUBLIC_IP`
5. Open the Spark Master UI in your browser: `http://<MASTER_PUBLIC_IP>:8080`
6. You should see 3 workers listed as ALIVE

If any worker fails to register, SSH to the worker and run `install_worker.sh` manually, then restart Spark from master: `$SPARK_HOME/sbin/stop-all.sh && $SPARK_HOME/sbin/start-all.sh`.

## Submitting Jobs

From the master node:

    source ~/spark-cluster/.venv/bin/activate
    export PYSPARK_PYTHON=/home/ubuntu/spark-cluster/.venv/bin/python

    spark-submit \
      --master spark://<MASTER_PRIVATE_IP>:7077 \
      --deploy-mode client \
      --executor-memory 4G \
      --driver-memory 2G \
      --total-executor-cores 6 \
      my_script.py

Monitor at `http://<MASTER_PUBLIC_IP>:8080` (Master UI) and `http://<MASTER_PUBLIC_IP>:4040` (Application UI, only while a job runs).

## Cost

Per hour while running:

- 4 × t3.large at $0.0832/hr = ~$0.33/hr
- EBS ~$0.01/hr

About **$0.34/hr total while running**, or **~$8/day if left on 24/7**.

Stopping (not terminating) instances pauses compute cost; EBS still costs ~$0.40/day total while stopped.

## Cleanup

To tear down all cluster resources (instances, security group, key pair):

    ./cleanup-spark-cluster.sh

This script lives in the original lab folder; it is not reproduced here. Terminated instances incur no further charges.
