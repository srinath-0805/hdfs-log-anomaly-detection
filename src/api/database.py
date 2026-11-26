
from pathlib import Path
from datetime import datetime, timezone
import json
import duckdb
import uuid


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "database" / "hdfs_anomaly.duckdb"


def initialize_database():
    """Create prediction history table if it does not exist."""

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(DB_PATH)) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                prediction_id VARCHAR PRIMARY KEY,
                predicted_at TIMESTAMP,
                event_ids VARCHAR,
                sequence_length INTEGER,
                reconstruction_score DOUBLE,
                threshold DOUBLE,
                prediction VARCHAR
            )
        """)


def save_prediction(event_ids, result):
    """Persist one successful prediction in DuckDB."""

    prediction_id = str(uuid.uuid4())
    predicted_at = datetime.now(timezone.utc).replace(tzinfo=None)

    with duckdb.connect(str(DB_PATH)) as con:
        con.execute("""
            INSERT INTO prediction_history VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            prediction_id,
            predicted_at,
            json.dumps(event_ids),
            len(event_ids),
            float(result["reconstruction_score"]),
            float(result["threshold"]),
            result["prediction"]
        ])

    return {
        "prediction_id": prediction_id,
        "predicted_at": predicted_at.isoformat()
    }


def get_dashboard_summary():
    """Return KPI metrics from DuckDB prediction history."""

    with duckdb.connect(str(DB_PATH), read_only=True) as con:

        result = con.execute("""
            SELECT
                COUNT(*) AS total_predictions,

                COUNT(*) FILTER (
                    WHERE prediction = 'NORMAL'
                ) AS normal_predictions,

                COUNT(*) FILTER (
                    WHERE prediction = 'ANOMALY'
                ) AS anomaly_predictions,

                ROUND(
                    AVG(reconstruction_score), 4
                ) AS average_reconstruction_score,

                MAX(predicted_at) AS latest_prediction_at

            FROM prediction_history
        """).fetchone()

    return {
        "total_predictions": result[0],
        "normal_predictions": result[1],
        "anomaly_predictions": result[2],
        "average_reconstruction_score": result[3],
        "latest_prediction_at": (
            result[4].isoformat() if result[4] else None
        )
    }


def get_prediction_history():
    """Retrieve individual prediction records from DuckDB."""

    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        result = con.execute("""
            SELECT
                prediction_id,
                predicted_at,
                event_ids,
                sequence_length,
                reconstruction_score,
                threshold,
                prediction
            FROM prediction_history
            ORDER BY predicted_at DESC
        """).fetchdf()

    return result.to_dict(orient="records")
