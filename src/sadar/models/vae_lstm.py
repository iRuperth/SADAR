from __future__ import annotations

import torch
from torch import nn


class VAELSTM(nn.Module):
    # Variational LSTM autoencoder. Instead of a single latent vector it encodes
    # each window into a probability distribution (a mean and a variance) over the
    # latent space, samples from it, and decodes. Training balances two goals:
    # rebuild the window well, and keep the latent close to a standard normal (the
    # KL term). This models a distribution of what normal traffic looks like, so
    # the score has a principled probabilistic reading and the latent is smoother
    # than a plain autoencoder's.
    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        latent_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        beta: float = 0.001,
    ) -> None:
        super().__init__()
        # Weight on the KL term. Kept small so it regularises the latent without
        # overpowering the reconstruction and collapsing the posterior.
        self.beta = beta
        # PyTorch only applies LSTM dropout between stacked layers, so it is a
        # no-op (and warns) with a single layer.
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        # Two heads turn the encoder summary into the parameters of the latent
        # Gaussian: its mean and its log-variance.
        self.to_mu = nn.Linear(hidden_size, latent_size)
        self.to_logvar = nn.Linear(hidden_size, latent_size)
        self.from_latent = nn.Linear(latent_size, hidden_size)

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, n_features)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, (hidden, _) = self.encoder(x)
        last = hidden[-1]
        return self.to_mu(last), self.to_logvar(last)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Sample z = mu + sigma * eps with eps ~ N(0, I). Writing it this way keeps
        # the randomness outside the network so gradients still flow to mu and
        # logvar (the reparameterisation trick).
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, length: int) -> torch.Tensor:
        seed = self.from_latent(z).unsqueeze(1).repeat(1, length, 1)
        decoded, _ = self.decoder(seed)
        return self.output(decoded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For scoring we decode from the latent mean (no sampling) so the
        # reconstruction error is deterministic and comparable across runs.
        mu, _ = self.encode(x)
        return self.decode(mu, x.size(1))

    def elbo_loss(self, x: torch.Tensor) -> torch.Tensor:
        # Training objective: reconstruction error of a sampled latent plus the
        # KL divergence that pulls the latent distribution towards N(0, I).
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, x.size(1))
        reconstruction_term = nn.functional.mse_loss(reconstruction, x)
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return reconstruction_term + self.beta * kl
