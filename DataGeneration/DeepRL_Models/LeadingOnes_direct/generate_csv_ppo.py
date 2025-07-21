import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from ppo_theory_env import TheoryBenchmarkContinuousWrapper

# Output path (relative to project root)
OUTPUT_PATH = "../../../DataSets/DeepRL/LeadingOnesModel.csv"

# n values to evaluate
N_VALUES = [10, 20, 30, 40, 50, 100, 200, 500]

# Path to trained PPO model
MODEL_PATH = "ppo_theory_output/ppo_theory_final"


def generate_leadingones_model_csv(model_path=MODEL_PATH, n_values=N_VALUES, output_path=OUTPUT_PATH):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model = PPO.load(model_path)
    data = []
    for n in n_values:
        env = TheoryBenchmarkContinuousWrapper(n=n)
        for current_state in range(n+1):
            # Set up the environment to the desired state
            base_env = env
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            base_env.current_fitness = current_state
            base_env.bitstring = np.array([1]*current_state + [0]*(n-current_state))
            state = np.array([n, current_state], dtype=np.float32)
            # Get action from model (no exploration)
            action, _ = model.predict(state, deterministic=True)
            bitflips = int(np.clip(np.round(1 + action[0] * (n - 1)), 1, n))
            data.append([n, current_state, bitflips])
    df = pd.DataFrame(data, columns=["n", "current_state", "bitflips"])
    df.to_csv(output_path, index=False, header=False)
    print(f"CSV saved to {output_path} with shape {df.shape}")

if __name__ == "__main__":
    generate_leadingones_model_csv() 