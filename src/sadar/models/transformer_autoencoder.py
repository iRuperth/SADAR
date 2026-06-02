from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    # Attention has no built-in sense of order, so we add fixed sinusoidal
    # position signals to the embeddings. This tells the model where each step
    # sits in the window, which matters for a trajectory that unfolds in time.
    def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerAutoencoder(nn.Module):
    # Attention-based reconstruction model. Self-attention lets every timestep
    # look at every other one, so it can capture long-range structure in a
    # trajectory that a strictly sequential LSTM may smear out. A latent
    # bottleneck forces genuine compression: without it the attention stack could
    # just copy the input straight through and every window would reconstruct
    # perfectly, leaving nothing to flag as anomalous.
    def __init__(
        self,
        n_features: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        latent_size: int,
        dropout: float,
        max_len: int,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers, enable_nested_tensor=False
        )

        # Pool the encoded sequence into one vector, squeeze it through the latent
        # bottleneck, then expand it back to feed the decoder.
        self.to_latent = nn.Linear(d_model, latent_size)
        self.from_latent = nn.Linear(latent_size, d_model)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer, num_layers, enable_nested_tensor=False
        )
        self.output = nn.Linear(d_model, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        encoded = self.encoder(self.pos_encoder(self.input_proj(x)))

        # Average over time to summarise the whole window, then compress it.
        z = self.to_latent(encoded.mean(dim=1))

        # Repeat the latent at every step and re-add positions so the decoder has
        # something ordered to rebuild the sequence from.
        seed = self.from_latent(z).unsqueeze(1).repeat(1, length, 1)
        decoded = self.decoder(self.pos_encoder(seed))
        return self.output(decoded)
