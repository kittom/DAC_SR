import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from ppo_theory_env import TheoryBenchmarkContinuousWrapper
import gymnasium as gym

class RandomNWrapper(gym.Wrapper):
    """
    Gym wrapper to randomize n for each episode.
    """
    def __init__(self, env_class, n_min=10, n_max=100, **kwargs):
        self.env_class = env_class
        self.n_min = n_min
        self.n_max = n_max
        self.kwargs = kwargs
        super().__init__(self.env_class(n=self.n_min, **self.kwargs))

    def reset(self, **kwargs):
        n = np.random.randint(self.n_min, self.n_max + 1)
        self.env = self.env_class(n=n, **self.kwargs)
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

if __name__ == "__main__":
    out_dir = "ppo_theory_output"
    os.makedirs(out_dir, exist_ok=True)
    # Wrap the environment to randomize n each episode
    def make_env():
        return RandomNWrapper(TheoryBenchmarkContinuousWrapper, n_min=10, n_max=100)
    env = make_vec_env(make_env, n_envs=4)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=out_dir,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        device="auto"
    )
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=out_dir, name_prefix="ppo_model")
    model.learn(total_timesteps=100_000, callback=checkpoint_callback)
    model.save(os.path.join(out_dir, "ppo_theory_final"))
    print(f"Training complete. Model saved to {out_dir}/ppo_theory_final") 