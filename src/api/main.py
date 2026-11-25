
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.inference import predict_sequence


app = FastAPI(
    title="HDFS Log Anomaly Detection API",
    description="Detect HDFS anomalies using an LSTM Autoencoder",
    version="1.0.0"
)


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


@app.post("/predict")
def predict(request: PredictionRequest):

    try:
        result = predict_sequence(request.event_ids)

        return {
            "event_ids": request.event_ids,
            "sequence_length": len(request.event_ids),
            **result
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )
