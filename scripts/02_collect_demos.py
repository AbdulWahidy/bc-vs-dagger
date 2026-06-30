"""Collect a pile of expert demonstrations and save them to data/.

We collect a generous number of episodes ONCE here. Step 05 then slices the
first-N of them (1, 4, 16, ...) to run the data-starvation ablation, with no
need to ever re-roll the expert.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.experts import Expert, _paths  # noqa: E402
from src.data import (  # noqa: E402
    collect_demonstrations, save_demonstrations, load_demonstrations,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="Hopper-v5")
    p.add_argument("--expert-dir", default="experts")
    p.add_argument("--out", default="data/demos.npz")
    p.add_argument("--episodes", type=int, default=25,
                   help="how many expert trajectories to record")
    p.add_argument("--seed", type=int, default=2000)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model_path, stats_path = _paths(args.env_id, args.expert_dir)
    expert = Expert.load(model_path, stats_path)

    print(f"Rolling out expert for {args.episodes} episodes...")
    trajs = collect_demonstrations(
        expert, env_id=args.env_id, n_episodes=args.episodes, seed=args.seed,
    )
    save_demonstrations(trajs, args.out)

    # sanity report — the demos should look like your evaluated expert
    rets = np.array([t["return"] for t in trajs])
    lens = np.array([len(t["obs"]) for t in trajs])
    total_steps = int(lens.sum())
    print(f"\nSaved {len(trajs)} demos -> {args.out}")
    print(f"Demo returns: mean={rets.mean():.1f} ± {rets.std():.1f} "
          f"(min={rets.min():.1f}, max={rets.max():.1f})")
    print(f"Total (obs, act) pairs: {total_steps}  "
          f"(avg episode length {lens.mean():.0f})")

    # prove the save/load round-trip is exact before we build on it
    reloaded = load_demonstrations(args.out)
    assert len(reloaded) == len(trajs)
    assert np.allclose(reloaded[0]["obs"], trajs[0]["obs"])
    print("Round-trip load verified. Proceed to BC (step 03).")


if __name__ == "__main__":
    main()