"""PPO expert: train it, save it, and load it as a queryable policy.

The Expert class is the reusable piece. It exposes one method,
`predict(raw_obs) -> action`, called by BOTH demo collection (step 02) and
DAgger (step 05). DAgger needs the expert to be a live, callable function,
not a frozen dataset, so it can be queried at the novel states the learner
visits.

Wrinkle hidden in here: the PPO expert is trained with observation
normalization (VecNormalize), so its network expects *normalized* obs. We save
the normalization stats next to the model and apply them inside `predict`, so
the rest of the project works in the env's raw observation space.
"""
from __future__ import annotations
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Tuned PPO hyperparameters for Hopper, adapted from the SB3 RL Zoo.
# Reliably yields a usable expert (~2500-3500 return) in ~1M steps.
HOPPER_PPO_HYPERPARAMS = dict(
    policy="MlpPolicy",
    n_steps=512,
    batch_size=32,
    n_epochs=5,
    gamma=0.999,
    gae_lambda=0.99,
    learning_rate=9.80828e-05,
    ent_coef=0.00229519,
    clip_range=0.2,
    max_grad_norm=0.7,
    vf_coef=0.835671,
    device="cpu",  # small MLP: CPU beats MPS here and dodges op-support quirks
)


def _paths(env_id, save_dir):
    model_path = os.path.join(save_dir, f"{env_id}_ppo.zip")
    stats_path = os.path.join(save_dir, f"{env_id}_obs_stats.npz")
    return model_path, stats_path


def train_expert(env_id="Hopper-v5", total_timesteps=1_000_000, seed=0,
                 save_dir="experts", verbose=1):
    """Train a PPO expert with obs+reward normalization; save the model and
    the final observation-normalization statistics."""
    os.makedirs(save_dir, exist_ok=True)

    venv = DummyVecEnv([lambda: gym.make(env_id)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(env=venv, seed=seed, verbose=verbose, **HOPPER_PPO_HYPERPARAMS)
    model.learn(total_timesteps=total_timesteps, log_interval=10)

    model_path, stats_path = _paths(env_id, save_dir)
    model.save(model_path)
    np.savez(
        stats_path,
        mean=venv.obs_rms.mean,
        var=venv.obs_rms.var,
        clip_obs=np.float32(venv.clip_obs),
        epsilon=np.float32(venv.epsilon),
    )
    venv.close()
    print(f"Saved expert    -> {model_path}")
    print(f"Saved obs stats -> {stats_path}")
    return model_path, stats_path


class Expert:
    """A loadable, queryable PPO expert. Takes RAW observations, returns actions."""

    def __init__(self, model, obs_mean, obs_var, clip_obs=10.0, epsilon=1e-8):
        self.model = model
        self.obs_mean = obs_mean.astype(np.float32)
        self.obs_var = obs_var.astype(np.float32)
        self.clip_obs = float(clip_obs)
        self.epsilon = float(epsilon)

    @classmethod
    def load(cls, model_path, stats_path, device="cpu"):
        model = PPO.load(model_path, device=device)
        s = np.load(stats_path)
        return cls(model, s["mean"], s["var"], s["clip_obs"], s["epsilon"])

    def _normalize(self, obs):
        norm = (obs - self.obs_mean) / np.sqrt(self.obs_var + self.epsilon)
        return np.clip(norm, -self.clip_obs, self.clip_obs)

    def predict(self, obs, deterministic=True):
        """obs: shape (obs_dim,) or (N, obs_dim). Returns matching action(s)."""
        obs = np.asarray(obs, dtype=np.float32)
        single = obs.ndim == 1
        batch = obs[None, :] if single else obs
        norm = self._normalize(batch).astype(np.float32)
        actions, _ = self.model.predict(norm, deterministic=deterministic)
        return actions[0] if single else actions