# HDFS Log Anomaly Detection

Capstone: Deep Learning-Based HDFS Log Anomaly Detection and Incident Intelligence.

## Dataset
HDFS_v1 (LogHub): https://github.com/logpai/loghub/tree/master/HDFS
Download: https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1

Place the ZIP in `data/raw/` and extract its contents into `data/extracted/`.

## Workflow
1. Inspect HDFS_v1 and its labels.
2. Perform Python/Pandas EDA and preprocessing.
3. Use DuckDB and SQL for analytical queries.
4. Train an Isolation Forest baseline.
5. Train and evaluate an LSTM Autoencoder.
6. Expose inference through FastAPI.
7. Visualize prediction outputs in Power BI.
8. Package with Docker and attempt AWS deployment.

## Setup
```bash
conda create -n hdfs_anomaly python=3.11 -y
conda activate hdfs_anomaly
pip install -r requirements.txt
```

Do not commit raw datasets, credentials, `.env` files, or large model artifacts to Git.
