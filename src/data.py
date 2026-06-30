"""Collect expert demonstrations and hold them as a supervised dataset.

A "demo" here = one full expert trajectory (episode), stored as its own
(observations, actions) arrays. We keep trajectories *separate* rather than
mashing everything into one flat array, because step 05 needs to train BC on
the first-N demos (1, 4, 16, ...) to show return collapse. Storing per-episode
lets us slice by demo count cheaply, with no re-rolling of the expert.

The same Dataset class also backs DAgger (step 05): DAgger keeps appending
freshly-labeled (state, expert_action) pairs to a growing buffer, which is
exactly a flat (obs, act) store. So one class serves both.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym


def collect_demonstrations(expert, env_id="Hopper-v5", n_episodes=25, seed=2000):
    """Roll the expert out for n_episodes and record (obs, action) per step.

    Returns a list of trajectories; each trajectory is a dict with:
        "obs":     (T_i, obs_dim) float32
        "act":     (T_i, act_dim) float32
        "return":  scalar episode return (for sanity-checking demo quality)
    Trajectories vary in length T_i (Hopper episodes end when it falls/truncates).
    """
    env = gym.make(env_id)
    trajectories = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        obs_buf, act_buf = [], []
        done, ep_return = False, 0.0
        while not done:
            action = expert.predict(obs, deterministic=True)
            obs_buf.append(np.asarray(obs, dtype=np.float32))
            act_buf.append(np.asarray(action, dtype=np.float32))
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated
        trajectories.append({
            "obs": np.array(obs_buf, dtype=np.float32),
            "act": np.array(act_buf, dtype=np.float32),
            "return": np.float32(ep_return),
        })
    env.close()
    return trajectories


def save_demonstrations(trajectories, path):
    """Save the list of trajectories to a single .npz.

    np.savez can't store a ragged list directly, so we concatenate all steps
    into flat (obs, act) arrays plus a `lengths` array recording each episode's
    length. That lets us reconstruct exact per-episode boundaries on load.
    """
    obs = np.concatenate([t["obs"] for t in trajectories], axis=0)
    act = np.concatenate([t["act"] for t in trajectories], axis=0)
    lengths = np.array([len(t["obs"]) for t in trajectories], dtype=np.int64)
    returns = np.array([t["return"] for t in trajectories], dtype=np.float32)
    np.savez(path, obs=obs, act=act, lengths=lengths, returns=returns)


def load_demonstrations(path):
    """Inverse of save: returns the list-of-trajectories form."""
    d = np.load(path)
    obs, act, lengths = d["obs"], d["act"], d["lengths"]
    returns = d["returns"]
    trajectories, start = [], 0
    for i, L in enumerate(lengths):
        end = start + int(L)
        trajectories.append({
            "obs": obs[start:end],
            "act": act[start:end],
            "return": returns[i],
        })
        start = end
    return trajectories


class DemoDataset:
    """A flat (obs, act) store for supervised imitation learning.

    Build it from the first-N demos (BC ablation) via `from_trajectories`,
    or start it empty and `.add(obs, act)` to it (DAgger aggregation).
    `sample(batch_size)` yields random minibatches for the training loop.
    """

    def __init__(self, obs=None, act=None):
        self.obs = obs if obs is not None else np.empty((0, 0), np.float32)
        self.act = act if act is not None else np.empty((0, 0), np.float32)

    @classmethod
    def from_trajectories(cls, trajectories, n_demos=None):
        """Flatten the first `n_demos` trajectories into one (obs, act) store.
        n_demos=None uses all of them. This is the BC-ablation knob."""
        chosen = trajectories if n_demos is None else trajectories[:n_demos]
        obs = np.concatenate([t["obs"] for t in chosen], axis=0)
        act = np.concatenate([t["act"] for t in chosen], axis=0)
        return cls(obs.astype(np.float32), act.astype(np.float32))

    def add(self, obs, act):
        """Append new (obs, act) pairs. Used by DAgger to grow the buffer."""
        obs = np.asarray(obs, dtype=np.float32)
        act = np.asarray(act, dtype=np.float32)
        if self.obs.size == 0:
            self.obs, self.act = obs, act
        else:
            self.obs = np.concatenate([self.obs, obs], axis=0)
            self.act = np.concatenate([self.act, act], axis=0)

    def __len__(self):
        return len(self.obs)

    def sample(self, batch_size):
        idx = np.random.randint(0, len(self.obs), size=batch_size)
        return self.obs[idx], self.act[idx]