"""BC data-starvation ablation: the failure made visible.

Retrain the IDENTICAL BC setup on the first-N demos for N in {1,2,4,8,16,25}
and evaluate each. Everything is held fixed except the number of demos, so the
return curve isolates ONE variable: state-space coverage. The collapse at low N
is compounding error / distribution shift — the student drifts into states the
narrow demo set never covered.

We also seed-average each N (a few BC training seeds) so the curve reflects the
method, not one lucky/unlucky initialization.
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
    p.add_argument("--demo-counts", type=int, nargs="+",
                   default=[1, 2, 4, 8, 16, 25])
    p.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--out", default="results/bc_ablation.npz")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    trajs = load_demonstrations(args.demos)
    max_demos = len(trajs)
    obs_dim = trajs[0]["obs"].shape[1]
    act_dim = trajs[0]["act"].shape[1]

    counts, means, stds = [], [], []
    for n_demos in args.demo_counts:
        if n_demos > max_demos:
            print(f"(skipping N={n_demos}: only {max_demos} demos available)")
            continue
        dataset = DemoDataset.from_trajectories(trajs, n_demos=n_demos)
        seed_means = []
        for s in args.train_seeds:
            policy, _ = train_bc(dataset, obs_dim, act_dim,
                                 epochs=args.epochs, seed=s, verbose=False)
            r = evaluate_policy(policy, args.env_id,
                                n_episodes=args.eval_episodes)
            seed_means.append(r.mean())
        seed_means = np.array(seed_means)
        counts.append(n_demos)
        means.append(seed_means.mean())
        stds.append(seed_means.std())
        print(f"N={n_demos:2d} demos ({len(dataset):5d} pairs)  "
              f"return = {seed_means.mean():7.1f} ± {seed_means.std():5.1f}  "
              f"(across {len(args.train_seeds)} seeds)")

    np.savez(args.out, counts=np.array(counts),
             means=np.array(means), stds=np.array(stds))
    print(f"\nSaved ablation -> {args.out}")
    print("Expect: low return at few demos, rising toward ~2800 at 25.")
    print("That rising curve IS distribution shift. Next: step 05, DAgger.")


if __name__ == "__main__":
    main()