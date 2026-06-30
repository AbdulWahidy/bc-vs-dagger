"""DAgger: Dataset Aggregation.

Reuses everything — MLPPolicy, train_bc, Expert.predict, DemoDataset.add. The
only new idea is WHO drives during data collection:

  repeat:
    1. train policy on the current aggregated dataset        [train_bc]
    2. roll the *student* out; record the states it visits   [student drives]
    3. ask the expert for the right action at each state      [Expert.predict]
    4. add (student_state, expert_action) to the dataset      [DemoDataset.add]

We track cumulative expert-label count so step 06 can compare label-efficiency
against BC on the same axes.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym

from src.bc import train_bc
from src.data import DemoDataset


def collect_student_states(policy, env_id, n_episodes, seed):
    """Roll the CURRENT student out; record every observation it visits.
    We keep the student's STATES and discard the student's actions — those are
    the mistakes. The expert supplies the correct actions afterward."""
    env = gym.make(env_id)
    states, returns = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            states.append(np.asarray(obs, dtype=np.float32))
            action = policy.predict(obs, deterministic=True)  # student drives
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
    env.close()
    return np.array(states, dtype=np.float32), returns


def evaluate_policy(policy, env_id, n_episodes, seed):
    env = gym.make(env_id)
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            action = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
    env.close()
    return np.array(returns)


def run_dagger(seed_dataset, expert, obs_dim, act_dim, *, env_id="Hopper-v5",
               n_iterations=8, rollout_episodes=1, epochs=100, train_seed=0,
               eval_episodes=20, eval_seed=3000, collect_seed=5000, verbose=True):
    """Run DAgger from a seed dataset. Returns (final_policy, records).

    Each iteration: retrain from scratch on the aggregated data (matches how BC
    ablation trained — apples to apples), evaluate, then collect+label+aggregate.
    records: per-iteration dicts with cumulative n_labels, return mean/std.
    """
    dataset = DemoDataset(seed_dataset.obs.copy(), seed_dataset.act.copy())
    records = []

    for it in range(n_iterations + 1):
        policy, _ = train_bc(dataset, obs_dim, act_dim,
                             epochs=epochs, seed=train_seed, verbose=False)
        r = evaluate_policy(policy, env_id, eval_episodes, eval_seed)
        rec = dict(iter=it, n_labels=len(dataset),
                   return_mean=float(r.mean()), return_std=float(r.std()))
        if verbose:
            print(f"  DAgger iter {it:2d}  labels={len(dataset):6d}  "
                  f"return={r.mean():7.1f} ± {r.std():5.1f}")

        if it < n_iterations:
            states, roll_rets = collect_student_states(
                policy, env_id, rollout_episodes, collect_seed + it * 100)
            expert_actions = expert.predict(states, deterministic=True)
            dataset.add(states, expert_actions)
            rec["new_labels"] = int(len(states))
        records.append(rec)

    return policy, records