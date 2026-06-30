"""Train a DAgger policy."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import gymnasium as gym
import torch
from experts import load_expert, query_expert
from dagger import train_dagger
from evaluate import evaluate_policy
from utils import seed_everything, log_to_csv

ENV_ID = "LunarLander-v2"
EXPERT_PATH = "experts/ppo_lunarlander"
RESULTS_CSV = "results/dagger_results.csv"
N_ITERATIONS = 10
ROLLOUT_EPISODES = 10
N_EPOCHS = 20
SEED = 42

if __name__ == "__main__":
    seed_everything(SEED)

    env = gym.make(ENV_ID)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0] if hasattr(env.action_space, "shape") else env.action_space.n
    env.close()

    model = load_expert(EXPERT_PATH, ENV_ID)
    expert_fn = lambda obs: query_expert(model, obs)

    print(f"Training DAgger for {N_ITERATIONS} iterations...")
    policy = train_dagger(
        expert_fn,
        ENV_ID,
        obs_dim=obs_dim,
        act_dim=act_dim,
        n_iterations=N_ITERATIONS,
        rollout_episodes=ROLLOUT_EPISODES,
        n_epochs=N_EPOCHS,
    )

    mean, std = evaluate_policy(policy, ENV_ID, n_episodes=20, seed=SEED)
    print(f"DAgger return: {mean:.1f} ± {std:.1f}")

    os.makedirs("results", exist_ok=True)
    log_to_csv(RESULTS_CSV, {"method": "DAgger", "n_iterations": N_ITERATIONS,
                              "mean_return": mean, "std_return": std})
    torch.save(policy.state_dict(), "results/dagger_policy.pt")
