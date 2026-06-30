"""DAgger (Dataset Aggregation) training loop.

DAgger fixes BC's compounding-error problem by iteratively expanding the
training dataset with states the *learner* actually visits, labelled by the
expert.  After each rollout the buffer grows and the policy is retrained from
scratch on the full aggregate — so the policy eventually covers the states it
will encounter at test time.

Reference: Ross, Gordon & Bagnell (2011) — "A Reduction of Imitation Learning
and Structured Prediction to No-Regret Online Learning."
"""

import numpy as np

from data import DaggerBuffer, rollout_learner
from bc import train_bc
from policy import MLPPolicy


def train_dagger(
    expert,
    env_id: str,
    obs_dim: int,
    act_dim: int,
    n_iterations: int = 10,
    rollout_episodes: int = 10,
    n_epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> MLPPolicy:
    """Run the DAgger outer loop and return the final imitation policy.

    Each iteration:
      1. Roll out the *current* learner to collect visited observations.
      2. Query the *expert* on every visited observation to get labels.
      3. Add the (obs, expert_action) pairs to the aggregation buffer.
      4. Retrain the policy from scratch on the full buffer via BC.

    Args:
        expert: Callable ``obs -> action`` — the oracle that provides labels.
            Typically a lambda wrapping ``query_expert``.
        env_id: Gymnasium environment ID used for learner rollouts.
        obs_dim: Observation vector length — passed to the policy constructor.
        act_dim: Action vector length — passed to the policy constructor.
        n_iterations: Number of DAgger iterations (dataset grows each round).
        rollout_episodes: Episodes rolled out *per iteration* by the learner.
        n_epochs: BC training epochs run after each aggregation step.
        batch_size: Mini-batch size for BC inner loop.
        lr: Adam learning rate for BC inner loop.

    Returns:
        The final ``MLPPolicy`` trained on the fully aggregated dataset.
    """
    buffer = DaggerBuffer()
    # Start with a randomly initialised policy; it improves each iteration.
    policy = MLPPolicy(obs_dim, act_dim)

    for iteration in range(n_iterations):
        print(f"\n--- DAgger iteration {iteration + 1}/{n_iterations} ---")

        # Step 1: collect states the current learner visits.
        episode_obs_list = rollout_learner(policy, env_id, n_episodes=rollout_episodes)

        # Step 2: relabel every visited state with the expert's action.
        iter_obs = np.concatenate(episode_obs_list, axis=0)
        iter_acts = np.array([expert(o) for o in iter_obs])

        # Step 3: aggregate into the growing buffer.
        buffer.add(iter_obs, iter_acts)
        print(f"Buffer size: {len(buffer)} transitions")

        # Step 4: retrain from scratch on the full aggregate.
        agg = buffer.to_arrays()
        policy = train_bc(agg, obs_dim, act_dim, n_epochs=n_epochs,
                          batch_size=batch_size, lr=lr)

    return policy
