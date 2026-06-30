"""Show the failure: sweep #demos → return collapse."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from data import load_demos
from bc import train_bc
from evaluate import evaluate_policy
from utils import seed_everything, log_to_csv

ENV_ID = "LunarLander-v2"
DEMO_PATH = "data/demos.npz"
RESULTS_CSV = "results/bc_ablation.csv"
DEMO_COUNTS = [5, 10, 20, 50, 100]
N_EPOCHS = 50
SEED = 42

if __name__ == "__main__":
    seed_everything(SEED)
    demos = load_demos(DEMO_PATH)

    obs_dim = demos["observations"].shape[1]
    act_dim = demos["actions"].shape[1] if demos["actions"].ndim > 1 else 1

    os.makedirs("results", exist_ok=True)

    for n in DEMO_COUNTS:
        subset = {
            "observations": demos["observations"][:n * 100],
            "actions": demos["actions"][:n * 100],
        }
        policy = train_bc(subset, obs_dim, act_dim, n_epochs=N_EPOCHS)
        mean, std = evaluate_policy(policy, ENV_ID, n_episodes=20, seed=SEED)
        print(f"n_demos={n:4d}  return={mean:.1f} ± {std:.1f}")
        log_to_csv(RESULTS_CSV, {"n_demos": n, "mean_return": mean, "std_return": std})
