import numpy as np
import gymnasium as gym
from dacbench.benchmarks import TheoryBenchmark

class TheoryBenchmarkContinuousWrapper(gym.Env):
    """
    Wraps DACBench TheoryBenchmark to accept a continuous action in [0, 1],
    which is mapped to bitflips in [1, n].
    Observation: [n, current_fitness]
    Action: Box([0.0], [1.0])
    """
    def __init__(self, n=None, seed=None):
        self.benchmark = TheoryBenchmark()
        self.env = self.benchmark.get_environment()
        self.n = n if n is not None else self.env.n
        self.observation_space = gym.spaces.Box(
            low=np.array([1, 0]), high=np.array([self.n, self.n]), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([0.0]), high=np.array([1.0]), dtype=np.float32)
        self.seed = seed
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed = seed
        # Set n for this episode if needed
        if hasattr(self.env, 'set_n'):
            self.env.set_n(self.n)
        else:
            self.env.n = self.n
        obs, _ = self.env.reset()
        self.current_fitness = obs[1] if len(obs) > 1 else 0
        self.state = np.array([self.n, self.current_fitness], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        # action is a float in [0, 1], map to bitflips in [1, n]
        bitflips = int(np.clip(np.round(1 + action[0] * (self.n - 1)), 1, self.n))
        # DACBench expects an action index; we need to find the closest available bitflip in the portfolio
        if hasattr(self.env, 'action_choices'):
            # Find the closest available bitflip in the portfolio
            available = self.env.action_choices[self.env.inst_id]
            idx = int(np.argmin(np.abs(np.array(available) - bitflips)))
            env_action = idx
        else:
            env_action = bitflips - 1
        obs, reward, terminated, truncated, info = self.env.step(env_action)
        self.current_fitness = obs[1] if len(obs) > 1 else 0
        self.state = np.array([self.n, self.current_fitness], dtype=np.float32)
        done = terminated or truncated
        return self.state, reward, False, done, info

    def render(self, mode="human"):
        self.env.render(mode=mode)

    def close(self):
        self.env.close() 