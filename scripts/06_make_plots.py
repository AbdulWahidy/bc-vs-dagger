""" The two figures.

Fig 1 (bc_ablation.png): BC return vs number of demos — the distribution-shift
collapse (low + unstable at few demos, expert-level by ~4).

Fig 2 (label_efficiency.png): return vs cumulative EXPERT LABELS, BC vs DAgger
on one axis — the money plot. DAgger climbs from a starved 1-demo seed by
spending labels on the student's own drifted states.
"""
import argparse, os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data import load_demonstrations


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bc", default="results/bc_ablation.npz")
    p.add_argument("--dagger", default="results/dagger.npz")
    p.add_argument("--demos", default="data/demos.npz")
    p.add_argument("--outdir", default="results")
    args = p.parse_args()

    bc = np.load(args.bc)
    dag = np.load(args.dagger)
    bc_counts, bc_means, bc_stds = bc["counts"], bc["means"], bc["stds"]
    dag_labels, dag_means, dag_stds = dag["labels"], dag["means"], dag["stds"]

    trajs = load_demonstrations(args.demos)
    lengths = np.array([len(t["obs"]) for t in trajs])
    expert_return = float(np.mean([t["return"] for t in trajs]))
    cum = np.cumsum(lengths)
    bc_labels = np.array([cum[c - 1] for c in bc_counts])  # first-c demos

    # ---- Figure 1: BC ablation ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(bc_counts, bc_means, yerr=bc_stds, marker="o",
                capsize=4, color="#c0392b", label="Behavior cloning")
    ax.axhline(expert_return, ls="--", color="gray", label="Expert")
    ax.set_xscale("log", base=2)
    ax.set_xticks(bc_counts)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Number of expert demonstrations")
    ax.set_ylabel("Episode return")
    ax.set_title("BC collapses and destabilizes when starved of demos")
    ax.legend(); ax.grid(True, alpha=0.3)
    f1 = os.path.join(args.outdir, "bc_ablation.png")
    fig.tight_layout(); fig.savefig(f1, dpi=150); plt.close(fig)

    # ---- Figure 2: label efficiency ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(bc_labels, bc_means, yerr=bc_stds, marker="o",
                capsize=4, color="#c0392b", label="Behavior cloning")
    ax.errorbar(dag_labels, dag_means, yerr=dag_stds, marker="s",
                capsize=4, color="#2980b9", label="DAgger (1-demo seed)")
    ax.axhline(expert_return, ls="--", color="gray", label="Expert")
    ax.set_xlabel("Cumulative expert labels (obs–action pairs)")
    ax.set_ylabel("Episode return")
    ax.set_title("Label efficiency: BC vs DAgger")
    ax.legend(); ax.grid(True, alpha=0.3)
    f2 = os.path.join(args.outdir, "label_efficiency.png")
    fig.tight_layout(); fig.savefig(f2, dpi=150); plt.close(fig)

    print(f"Saved {f1}")
    print(f"Saved {f2}")
    print(f"(expert reference = {expert_return:.0f})")


if __name__ == "__main__":
    main()