from __future__ import annotations

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    # Sequence-to-sequence reconstruction model. The encoder compresses a flight
    # window into a small latent vector and the decoder rebuilds the window from
    # it. Trained on normal traffic only, the network learns the usual approach
    # and departure shapes; windows that break the pattern reconstruct poorly and
    # that reconstruction error becomes the anomaly score.
    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        latent_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # PyTorch only applies LSTM dropout between stacked layers, so asking for
        # it with a single layer does nothing and just prints a warning.
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        # Squeeze the encoder's final hidden state down into the latent bottleneck.
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.latent_dropout = nn.Dropout(dropout)
        # Expand the latent vector back up to the decoder's hidden width.
        self.from_latent = nn.Linear(latent_size, hidden_size)

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        # Project the decoder states back onto the original feature space.
        self.output = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode the window and keep only the last layer's final hidden state as
        # the summary of the whole sequence.
        _, (hidden, _) = self.encoder(x)
        z = self.latent_dropout(self.to_latent(hidden[-1]))

        # The decoder needs an input at every timestep, so repeat the latent
        # vector across the full window length and let the LSTM unroll it.
        seed = self.from_latent(z).unsqueeze(1).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder(seed)
        return self.output(decoded)
