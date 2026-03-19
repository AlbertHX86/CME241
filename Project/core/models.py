from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet


class MLPBackbone(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: Tuple[int, int] = (128, 128)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
        )
        self.output_dim = hidden_sizes[1]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class TemporalCNNBackbone(nn.Module):
    def __init__(
        self,
        window_size: int,
        n_assets: int,
        n_features: int,
        action_dim: int,
        hidden_size: int = 128,
        temporal_pool_size: int = 4,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.n_assets = int(n_assets)
        self.n_features = int(n_features)
        self.action_dim = int(action_dim)
        self.series_channels = self.n_assets * self.n_features
        self.temporal_pool_size = int(temporal_pool_size)

        self.conv = nn.Sequential(
            nn.Conv1d(self.series_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(self.temporal_pool_size),
        )
        fc_in = 64 * self.temporal_pool_size + self.action_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_in, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.output_dim = hidden_size

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.size(0)
        series_len = self.window_size * self.n_assets * self.n_features
        series = obs[:, :series_len].view(batch_size, self.window_size, self.series_channels).transpose(1, 2)
        prev_action = obs[:, series_len:series_len + self.action_dim]
        h_conv = self.conv(series)
        h_flat = h_conv.reshape(batch_size, -1)
        h = torch.cat([h_flat, prev_action], dim=-1)
        return self.fc(h)


class DirichletActor(nn.Module):
    def __init__(self, backbone: nn.Module, action_dim: int):
        super().__init__()
        self.backbone = backbone
        self.alpha_head = nn.Linear(self.backbone.output_dim, action_dim)

    def concentration(self, obs: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.alpha_head(self.backbone(obs))) + 1.0

    def distribution(self, obs: torch.Tensor) -> Dirichlet:
        return Dirichlet(self.concentration(obs))


class ValueCritic(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.value_head = nn.Linear(self.backbone.output_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.backbone(obs)).squeeze(-1)
