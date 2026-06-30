"""The imitation policy: a small MLP mapping observation -> action.

Deliberately minimal. The point of the project is to study *data distribution*
(BC vs DAgger), so we hold the model fixed and simple — same network for both
methods. If BC fails and DAgger succeeds with this identical net, the
difference is provably about the data, not the architecture.

Hopper: obs_dim=11, act_dim=3, actions in [-1, 1]. We tanh the output to
respect the action bounds.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        layers, last = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, act_dim), nn.Tanh()]  # bound actions to [-1, 1]
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)

    @torch.no_grad()
    def predict(self, obs, deterministic=True):
        """Mirror the Expert.predict interface so eval/DAgger can call either
        the expert or the learner through the same `.predict(obs)` signature.
        Takes raw obs (np), returns action (np)."""
        single = np.asarray(obs).ndim == 1
        x = torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32)
        a = self.forward(x).cpu().numpy()
        return a[0] if single else a