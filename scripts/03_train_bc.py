"""Train BC on the FULL demo set and evaluate it.

This is the 'BC works when given plenty of data' baseline. With ~25 demos the
state coverage is wide enough that BC should reach a solid fraction of expert
return. Step 04 then starves it to expose the failure.
"""
import argparse
import os
import sys

import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data import load_demonstrations, DemoDataset  # noqa: E402
from src.bc import train_bc  # noqa: E402


def evaluate_policy(policy, env_id, n_episodes=20, seed=3000):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="Hopper-v5")
    p.add_argument("--demos", default="data/demos.npz")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    trajs = load_demonstrations(args.demos)
    dataset = DemoDataset.from_trajectories(trajs)  # all demos
    obs_dim = dataset.obs.shape[1]
    act_dim = dataset.act.shape[1]
    print(f"Training BC on {len(trajs)} demos "
          f"({len(dataset)} pairs, obs_dim={obs_dim}, act_dim={act_dim})")

    policy, _ = train_bc(dataset, obs_dim, act_dim,
                         epochs=args.epochs, seed=args.seed)

    r = evaluate_policy(policy, args.env_id, n_episodes=args.eval_episodes)
    print(f"\nBC (full data) return: mean={r.mean():.1f} ± {r.std():.1f} "
          f"(min={r.min():.1f}, max={r.max():.1f})")
    print("Compare to expert ~2700-2800. Next: step 04, the data ablation.")


if __name__ == "__main__":
    main()