"""DAgger from a starved 1-demo seed, averaged over
several independent runs so the curve matches the BC ablation's methodology.

Each run is a full, independent DAgger process with its own train + collect
seeds (eval seed fixed for fairness). We average per iteration: label count and
return across runs, with std = run-to-run spread of the method. Same idea as
the 3-seed averaging step 04 used for BC, so the two curves are now apples-to-
apples.
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
    p.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", default="results/dagger.npz")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    trajs = load_demonstrations(args.demos)
    seed_dataset = DemoDataset.from_trajectories(trajs, n_demos=args.seed_demos)
    obs_dim, act_dim = seed_dataset.obs.shape[1], seed_dataset.act.shape[1]

    model_path, stats_path = _paths(args.env_id, args.expert_dir)
    expert = Expert.load(model_path, stats_path)

    print(f"Seeding DAgger with {args.seed_demos} demo(s) "
          f"({len(seed_dataset)} pairs).")
    print(f"Averaging over {len(args.train_seeds)} independent runs "
          f"x {args.iterations} iterations.\n")

    all_labels, all_means = [], []   # each row = one run, over iterations
    for run_i, s in enumerate(args.train_seeds):
        print(f"--- run {run_i + 1}/{len(args.train_seeds)} (train_seed={s}) ---")
        _, records = run_dagger(
            seed_dataset, expert, obs_dim, act_dim,
            env_id=args.env_id, n_iterations=args.iterations,
            rollout_episodes=args.rollout_episodes, epochs=args.epochs,
            train_seed=s, collect_seed=5000 + run_i * 10000,
            eval_seed=3000, verbose=True,
        )
        all_labels.append([r["n_labels"] for r in records])
        all_means.append([r["return_mean"] for r in records])
        print()

    all_labels = np.array(all_labels)   # (runs, iters)
    all_means = np.array(all_means)     # (runs, iters)
    labels = all_labels.mean(axis=0)    # mean label count per iteration
    means = all_means.mean(axis=0)      # mean return per iteration
    stds = all_means.std(axis=0)        # run-to-run spread of the method

    np.savez(args.out, labels=labels, means=means, stds=stds)

    print("Averaged DAgger curve:")
    for i in range(len(labels)):
        print(f"  iter {i:2d}  labels~{labels[i]:6.0f}  "
              f"return={means[i]:7.1f} ± {stds[i]:5.1f}")
    print(f"\nSaved -> {args.out}")
    print(f"Start: {means[0]:.1f} at ~{labels[0]:.0f} labels")
    print(f"End:   {means[-1]:.1f} at ~{labels[-1]:.0f} labels")


if __name__ == "__main__":
    main() 