"""Behavior Cloning (BC) training loop.

BC treats imitation learning as supervised regression: given a fixed dataset of
(observation, action) pairs collected from an expert, it minimises MSE between
the policy's predicted actions and the expert's recorded actions.

Limitation: BC is vulnerable to compounding errors — small deviations from the
training distribution accumulate over a trajectory, pushing the agent into
states the expert never visited.  The ``04_bc_data_ablation`` script
demonstrates this collapse as the number of demos decreases.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from policy import MLPPolicy


def train_bc(
    demos: dict,
    obs_dim: int,
    act_dim: int,
    n_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> MLPPolicy:
    """Train an MLP policy via supervised imitation on a fixed demo dataset.

    Args:
        demos: Dict with ``"observations"`` ``(N, obs_dim)`` and ``"actions"``
            ``(N, act_dim)`` NumPy arrays, as returned by ``collect_demos`` or
            ``DaggerBuffer.to_arrays()``.
        obs_dim: Observation vector length — used to construct the network.
        act_dim: Action vector length — used to construct the network.
        n_epochs: Number of full passes over the dataset.
        batch_size: Mini-batch size for gradient updates.
        lr: Adam learning rate.

    Returns:
        The trained ``MLPPolicy`` (in eval-mode-compatible state; gradients
        still enabled — call ``policy.eval()`` before deployment if needed).
    """
    obs = torch.FloatTensor(demos["observations"])
    acts = torch.FloatTensor(demos["actions"])

    dataset = TensorDataset(obs, acts)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    policy = MLPPolicy(obs_dim, act_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(n_epochs):
        total_loss = 0.0
        for batch_obs, batch_acts in loader:
            pred = policy(batch_obs)
            loss = loss_fn(pred, batch_acts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}  loss={total_loss / len(loader):.4f}")

    return policy
