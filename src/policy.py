"""Shared MLP policy architecture used by both BC and DAgger.

Keeping a single ``MLPPolicy`` class here ensures BC and DAgger are always
compared on identical network capacity.  The class is a thin ``nn.Module``
wrapper that adds a NumPy-friendly ``predict`` method so it can be used as a
drop-in callable wherever ``obs -> action`` is expected.
"""

import torch
import torch.nn as nn
import numpy as np


class MLPPolicy(nn.Module):
    """Fully-connected policy network mapping observations to actions.

    Architecture: Linear → ReLU stacked ``len(hidden_sizes)`` times, followed
    by a linear output layer.  No activation on the output, so it can represent
    both continuous actions (regression) and raw logits (if wrapped with a loss
    that applies softmax).

    Args:
        obs_dim: Dimensionality of the observation vector.
        act_dim: Dimensionality of the action output.
        hidden_sizes: Width of each hidden layer. Default ``(64, 64)`` matches
            the SB3 MlpPolicy default so comparisons are fair.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: tuple = (64, 64)):
        super().__init__()
        layers = []
        in_size = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.ReLU()]
            in_size = h
        layers.append(nn.Linear(in_size, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard PyTorch forward pass (batched tensors in, tensor out).

        Args:
            x: ``(batch, obs_dim)`` float tensor.

        Returns:
            ``(batch, act_dim)`` float tensor of raw action values.
        """
        return self.net(x)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Inference helper: accept a single NumPy observation, return action.

        Adds and removes the batch dimension internally so callers don't have
        to manage tensor shapes.

        Args:
            obs: ``(obs_dim,)`` float array.

        Returns:
            ``(act_dim,)`` float NumPy array.
        """
        with torch.no_grad():
            x = torch.FloatTensor(obs).unsqueeze(0)
            return self.net(x).squeeze(0).numpy()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Alias for ``predict`` so the policy works as a plain callable.

        This lets ``MLPPolicy`` instances be passed anywhere an
        ``obs -> action`` callable is expected (e.g. ``rollout_learner``).
        """
        return self.predict(obs)
