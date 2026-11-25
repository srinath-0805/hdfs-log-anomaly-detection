
from pathlib import Path

import torch
import torch.nn.functional as F

from api.model import LSTM_Autoencoder


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "lstm_autoencoder_final.pth"


# Load checkpoint
checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

# Model configuration
model = LSTM_Autoencoder(
    vocab_size=checkpoint["vocab_size"],
    embedding_dim=checkpoint["embedding_dim"],
    hidden_dim=checkpoint["hidden_dim"],
    padding_idx=checkpoint["padding_idx"]
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Inference configuration
PADDING_IDX = checkpoint["padding_idx"]
ANOMALY_THRESHOLD = checkpoint["thresholds"]["97.5th_percentile"]


def predict_sequence(event_ids):
    """
    Calculate reconstruction score and anomaly prediction
    for one HDFS event sequence.
    """

    if not event_ids:
        raise ValueError("Event sequence cannot be empty.")

    if any(
        not isinstance(event_id, int)
        or event_id <= PADDING_IDX
        or event_id >= checkpoint["vocab_size"]
        for event_id in event_ids
    ):
        raise ValueError("Invalid event ID in sequence.")

    input_tensor = torch.tensor(
        [event_ids],
        dtype=torch.long
    )

    lengths = torch.tensor([len(event_ids)])

    with torch.no_grad():
        logits = model(input_tensor, lengths)

        loss = F.cross_entropy(
            logits.transpose(1, 2),
            input_tensor,
            reduction="none",
            ignore_index=PADDING_IDX
        )

        valid_tokens = input_tensor.ne(PADDING_IDX)

        reconstruction_score = (
            (loss * valid_tokens).sum(dim=1)
            / valid_tokens.sum(dim=1)
        ).item()

    prediction = (
        "ANOMALY"
        if reconstruction_score >= ANOMALY_THRESHOLD
        else "NORMAL"
    )

    return {
    "reconstruction_score": float(reconstruction_score),
    "threshold": float(ANOMALY_THRESHOLD),
    "prediction": prediction
    }
