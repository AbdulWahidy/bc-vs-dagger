"""Generate return curves: BC vs DAgger vs expert."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np
from utils import plot_return_curves

BC_ABLATION_CSV = "results/bc_ablation.csv"
DAGGER_CSV = "results/dagger_results.csv"
EXPERT_RETURN = 250.0  # approximate; replace with actual value from 01_train_expert.py
PLOT_PATH = "results/bc_vs_dagger.png"

if __name__ == "__main__":
    results = {}

    if os.path.exists(BC_ABLATION_CSV):
        df = pd.read_csv(BC_ABLATION_CSV)
        results["BC"] = (
            df["n_demos"].tolist(),
            df["mean_return"].tolist(),
            df["std_return"].tolist(),
        )

    if os.path.exists(DAGGER_CSV):
        df = pd.read_csv(DAGGER_CSV)
        n = len(results.get("BC", ([],))[0]) or 5
        xs = list(range(1, n + 1))
        means = [df["mean_return"].iloc[0]] * n
        stds = [df["std_return"].iloc[0]] * n
        results["DAgger"] = (xs, means, stds)

    n = len(results.get("BC", ([1, 2, 3, 4, 5],))[0])
    results["Expert"] = (
        list(range(1, n + 1)),
        [EXPERT_RETURN] * n,
        [0.0] * n,
    )

    os.makedirs("results", exist_ok=True)
    plot_return_curves(results, save_path=PLOT_PATH)
    print(f"Plot saved to {PLOT_PATH}")
