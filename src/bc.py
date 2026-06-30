"""Behavior cloning: plain supervised regression from obs to expert action.

There is nothing imitation-specific in the training math here — it's MSE
regression. The *only* thing that makes it "imitation learning" is where the
data came from (expert rollouts) and where it gets tested (the env). That's the
whole subtlety BC ignores and DAgger addresses.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from src.policy import MLPPolicy
from src.data import DemoDataset


def train_bc(dataset: DemoDataset, obs_dim, act_dim, *,
             epochs=100, batch_size=256, lr=1e-3, hidden=(256, 256),
             device="cpu", seed=0, verbose=True):
    """Fit an MLPPolicy to (obs -> action) by MSE. Returns the trained policy.

    'epochs' here = passes' worth of gradient steps: we do
    len(dataset)/batch_size steps per epoch, sampling random minibatches.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy = MLPPolicy(obs_dim, act_dim, hidden=hidden).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n = len(dataset)
    steps_per_epoch = max(1, n // batch_size)
    history = []
    for ep in range(epochs):
        ep_loss = 0.0
        for _ in range(steps_per_epoch):
            obs_b, act_b = dataset.sample(batch_size)
            obs_t = torch.as_tensor(obs_b, dtype=torch.float32, device=device)
            act_t = torch.as_tensor(act_b, dtype=torch.float32, device=device)
            pred = policy(obs_t)
            loss = loss_fn(pred, act_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        ep_loss /= steps_per_epoch
        history.append(ep_loss)
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}  mse={ep_loss:.5f}")
    return policy, history