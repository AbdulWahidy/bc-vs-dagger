"""Train a BC policy on the collected demonstrations."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from data import load_demos
from bc import train_bc
from evaluate import evaluate_policy
from utils import seed_everything, log_to_csv

ENV_ID = "LunarLander-v2"
DEMO_PATH = "data/demos.npz"
RESULTS_CSV = "results/bc_results.csv"
N_EPOCHS = 100
SEED = 42

if __name__ == "__main__":
    seed_everything(SEED)
    demos = load_demos(DEMO_PATH)

    obs_dim = demos["observations"].shape[1]
    act_dim = demos["actions"].shape[1] if demos["actions"].ndim > 1 else 1

    print(f"Training BC  obs_dim={obs_dim}  act_dim={act_dim}  "
          f"transitions={len(demos['observations'])}")

    policy = train_bc(demos, obs_dim, act_dim, n_epochs=N_EPOCHS)

    mean, std = evaluate_policy(policy, ENV_ID, n_episodes=20, seed=SEED)
    print(f"BC return: {mean:.1f} ± {std:.1f}")

    os.makedirs("results", exist_ok=True)
    log_to_csv(RESULTS_CSV, {"method": "BC", "n_demos": len(demos["observations"]),
                              "mean_return": mean, "std_return": std})
    torch.save(policy.state_dict(), "results/bc_policy.pt")
