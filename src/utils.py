"""Shared utilities: deterministic seeding, CSV logging, and plotting.

All side-effect-free helpers live here so the main training modules stay
focused on algorithmic logic.
"""

import random
import os
import csv

import numpy as np
import matplotlib.pyplot as plt
import torch


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed: Integer seed applied to all RNG sources.  Use the same value
            across all scripts to make runs comparable.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def log_to_csv(path: str, row: dict) -> None:
    """Append a single row to a CSV file, writing the header on first creation.

    Creates any missing parent directories automatically.

    Args:
        path: Destination file path (e.g. ``"results/bc_results.csv"``).
        row: Dict whose keys become column names.  Key order is preserved on
            Python 3.7+; all keys must be present on every call to the same
            file so that columns stay consistent.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def plot_return_curves(results: dict, save_path: str | None = None) -> None:
    """Plot mean return curves with shaded ±1 std bands for multiple methods.

    Args:
        results: Mapping from method label to a 3-tuple
            ``(x_values, mean_returns, std_returns)`` where all three are
            equal-length sequences.  Example::

                {
                    "BC":     ([5, 10, 20, 50], [100, 150, 180, 200], [10, 8, 6, 5]),
                    "DAgger": ([5, 10, 20, 50], [200, 210, 215, 220], [5, 4, 4, 3]),
                    "Expert": ([5, 10, 20, 50], [250, 250, 250, 250], [0, 0, 0, 0]),
                }

        save_path: If provided, the figure is saved to this path (parent
            directories are created automatically) at 150 dpi.  If ``None``,
            ``plt.show()`` is called instead.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (xs, means, stds) in results.items():
        means = np.array(means)
        stds = np.array(stds)
        ax.plot(xs, means, label=label)
        ax.fill_between(xs, means - stds, means + stds, alpha=0.2)

    ax.set_xlabel("Number of demonstrations / DAgger iterations")
    ax.set_ylabel("Mean return")
    ax.set_title("BC vs DAgger vs Expert")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)
