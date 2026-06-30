"""Step 05: DAgger from a starved 1-demo seed.

BC at 1 demo was stuck at ~1015 ± 684 (collapsed AND unstable). Seed DAgger
with that SAME 1 demo; let it collect expert corrections on the student's own
drifted states. Watch return climb to expert level while we count cumulative
expert labels — the basis for the label-efficiency comparison in step 06.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.experts import Expert, _paths
from src.data import load_demonstrations, DemoDataset
from src.dagger import run_dagger


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="Hopper-v5")
    p.add_argument("--demos", default="data/demos.npz")
    p.add_argument("--expert-dir", default="experts")
    p.add_argument("--seed-demos", type=int, default=1)
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--rollout-episodes", type=int, default=1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--out", default="results/dagger.npz")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    trajs = load_demonstrations(args.demos)
    seed_dataset = DemoDataset.from_trajectories(trajs, n_demos=args.seed_demos)
    obs_dim, act_dim = seed_dataset.obs.shape[1], seed_dataset.act.shape[1]

    model_path, stats_path = _paths(args.env_id, args.expert_dir)
    expert = Expert.load(model_path, stats_path)

    print(f"Seeding DAgger with {args.seed_demos} demo(s) "
          f"({len(seed_dataset)} pairs). Running {args.iterations} iterations.\n")
    policy, records = run_dagger(
        seed_dataset, expert, obs_dim, act_dim,
        env_id=args.env_id, n_iterations=args.iterations,
        rollout_episodes=args.rollout_episodes, epochs=args.epochs,
        train_seed=args.train_seed,
    )

    labels = np.array([r["n_labels"] for r in records])
    means = np.array([r["return_mean"] for r in records])
    stds = np.array([r["return_std"] for r in records])
    np.savez(args.out, labels=labels, means=means, stds=stds)
    print(f"\nSaved DAgger curve -> {args.out}")
    print(f"Start: {means[0]:.1f} ± {stds[0]:.1f} at {labels[0]} labels")
    print(f"End:   {means[-1]:.1f} ± {stds[-1]:.1f} at {labels[-1]} labels")
    print("Step 06 plots this against the BC ablation.")


if __name__ == "__main__":
    main()