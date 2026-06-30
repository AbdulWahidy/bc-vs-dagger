"""Headless policy evaluation: run rollouts and return mean/std return.

Kept separate from training code so the same evaluation protocol is used
for the expert, BC policy, and DAgger policy — ensuring fair comparison.
Each episode uses a different seed offset so results are not locked to a
single initial state.
"""

import numpy as np
import gymnasium as gym


def evaluate_policy(
    policy,
    env_id: str,
    n_episodes: int = 20,
    seed: int = 0,
) -> tuple[float, float]:
    """Evaluate a policy over multiple episodes and return summary statistics.

    Args:
        policy: Callable ``obs -> action``.  Compatible with ``MLPPolicy``,
            a SB3 model wrapped in a lambda, or any other callable.
        env_id: Gymnasium environment ID to evaluate in.
        n_episodes: Number of evaluation episodes.  More episodes reduce
            variance in the reported statistics.
        seed: Base random seed.  Episode ``i`` uses ``seed + i`` so each
            episode starts from a distinct but reproducible initial state.

    Returns:
        ``(mean_return, std_return)`` computed across all episodes.
    """
    env = gym.make(env_id)
    returns = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0
        while not done:
            action = policy(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        returns.append(total_reward)

    env.close()
    return float(np.mean(returns)), float(np.std(returns))
