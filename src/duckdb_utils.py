from pathlib import Path
import duckdb

DB_PATH = Path("database/hdfs_anomaly.duckdb")

def connect(db_path: Path = DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))
