
import torch
import torch.nn as nn


# Original LSTM Encoder
class LSTMEncoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, padding_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

    def forward(self, x, lengths):

        embedded = self.embedding(x)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden, cell) = self.lstm(packed)

        return hidden, cell


# Original LSTM Decoder
class LSTMDecoder(nn.Module):

    def __init__(self, vocab_size, hidden_dim):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.output_layer = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def forward(self, hidden, cell, max_len):

        context = hidden[-1].unsqueeze(1).expand(
            -1, max_len, -1
        )

        output, _ = self.lstm(
            context,
            (hidden, cell)
        )

        logits = self.output_layer(output)

        return logits


# Original LSTM Autoencoder
class LSTM_Autoencoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        padding_idx=0
    ):
        super().__init__()

        self.encoder = LSTMEncoder(
            vocab_size,
            embedding_dim,
            hidden_dim,
            padding_idx
        )

        self.decoder = LSTMDecoder(
            vocab_size,
            hidden_dim
        )

    def forward(self, x, lengths):

        hidden, cell = self.encoder(x, lengths)

        logits = self.decoder(
            hidden,
            cell,
            x.size(1)
        )

        return logits
