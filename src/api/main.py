from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.inference import predict_sequence
from api.database import (
    initialize_database,
    save_prediction,
    get_dashboard_summary,
    get_prediction_history
)

app = FastAPI(
    title="HDFS Log Anomaly Detection API",
    description="Detect HDFS anomalies using an LSTM Autoencoder",
    version="1.0.0"
)

@app.on_event("startup")
def startup_event():
    initialize_database()

class PredictionRequest(BaseModel):
    event_ids: list[int] = Field(
        ...,
        min_length=1,
        description="Encoded HDFS event IDs"
    )


@app.get("/")
def home():
    return {
        "message": "HDFS Log Anomaly Detection API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "LSTM Autoencoder"
    }

@app.get("/dashboard/summary")
def dashboard_summary():
    try:
        return get_dashboard_summary()

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve dashboard summary"
        )

@app.get("/dashboard/predictions")
def prediction_history():
    try:
        return get_prediction_history()
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve prediction history"
        )

@app.post("/predict")
def predict(request: PredictionRequest):

    try:
        result = predict_sequence(request.event_ids)

        saved_record = save_prediction(
            request.event_ids,
            result
        )

        return {
            "event_ids": request.event_ids,
            "sequence_length": len(request.event_ids),
            **result,
            **saved_record
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )