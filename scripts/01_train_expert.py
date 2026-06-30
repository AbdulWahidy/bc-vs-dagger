"""Train and validate the PPO expert for Hopper-v5.

Validation runs the *encapsulated* Expert (raw-obs interface) on a fresh env,
which is exactly what BC and DAgger will use later — so a good number here
means the whole expert pipeline (incl. normalization) is wired correctly.
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
import gymnasium as gym

# make `src` importable no matter what directory you run this from
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.experts import train_expert, Expert, _paths  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(expert, env_id, n_episodes=10, seed=1000):
    env = gym.make(env_id)
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            action = expert.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
    env.close()
    return np.array(returns)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="Hopper-v5")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", default="experts")
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--skip-train", action="store_true",
                   help="reuse an already-trained expert; only evaluate")
    args = p.parse_args()

    set_seed(args.seed)
    model_path, stats_path = _paths(args.env_id, args.save_dir)

    if not args.skip_train:
        model_path, stats_path = train_expert(
            env_id=args.env_id, total_timesteps=args.timesteps,
            seed=args.seed, save_dir=args.save_dir,
        )

    expert = Expert.load(model_path, stats_path)
    r = evaluate(expert, args.env_id, n_episodes=args.eval_episodes)
    print(f"\nExpert over {len(r)} episodes: mean={r.mean():.1f} ± {r.std():.1f}"
          f"   (min={r.min():.1f}, max={r.max():.1f})")

    if r.mean() < 2000:
        print("WARNING: weak expert (<2000). Try another --seed or "
              "--timesteps 1500000. A muddy expert muddies the whole study.")
    else:
        print("Expert looks good. Proceed to demo collection (step 02).")


if __name__ == "__main__":
    main()