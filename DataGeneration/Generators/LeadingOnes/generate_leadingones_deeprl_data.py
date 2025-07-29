import os
import sys
import csv
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add the DeepRL models path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../DeepRL_Models/LeadingOnes_direct'))

def generate_leadingones_deeprl_data(instance_sizes, output_dir, expected_noise=1.1, data_type='continuous'):
    """
    Generate LeadingOnes data using the DeepRL model.
    
    Args:
        instance_sizes: List of problem sizes to generate
        output_dir: Output directory for results
        expected_noise: Expected noise level (default: 1.1)
        data_type: 'continuous' or 'discrete'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Import the DeepRL environment and model
    try:
        from ppo_theory_env import TheoryBenchmarkContinuousWrapper
        from stable_baselines3 import PPO
    except ImportError as e:
        print(f"Error importing DeepRL modules: {e}")
        print("Generating mock LeadingOnes data for testing purposes...")
        
        all_data = []
        
        for n in instance_sizes:
            print(f"Generating mock LeadingOnes data for n={n}")
            
            # Generate mock data
            csv_path = os.path.join(output_dir, f'leadingones_{n}D.csv')
            
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow(['n', 'current_fitness', 'action', 'reward', 'next_fitness'])
                
                # Generate mock episodes
                for episode in range(5):  # Generate 5 episodes per size
                    episode_data = []
                    
                    # Mock LeadingOnes progression
                    current_fitness = 0
                    for step in range(n):
                        n_val = n
                        action = np.random.uniform(0, 1)  # Mock action
                        reward = np.random.normal(0.5, expected_noise)  # Mock reward with noise
                        next_fitness = min(current_fitness + 1, n)  # Mock progression
                        
                        row = [n_val, current_fitness, action, reward, next_fitness]
                        writer.writerow(row)
                        episode_data.append(row)
                        
                        current_fitness = next_fitness
                        if current_fitness >= n:
                            break
                    
                    all_data.extend(episode_data)
            
            print(f"Generated {len(episode_data)} mock data points for n={n}")
        
        return all_data
    
    all_data = []
    
    for n in instance_sizes:
        print(f"Generating LeadingOnes data for n={n}")
        
        # Create environment
        env = TheoryBenchmarkContinuousWrapper(n=n)
        
        # Load the trained model
        model_path = os.path.join(os.path.dirname(__file__), "../../DeepRL_Models/LeadingOnes_direct/ppo_theory_output/ppo_theory_final")
        if not os.path.exists(model_path + ".zip"):
            print(f"Model not found at {model_path}. Please train the model first.")
            continue
            
        model = PPO.load(model_path)
        
        # Generate data
        csv_path = os.path.join(output_dir, f'leadingones_{n}D.csv')
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['n', 'current_fitness', 'action', 'reward', 'next_fitness'])
            
            # Generate multiple episodes
            for episode in range(10):  # Generate 10 episodes per size
                obs, _ = env.reset()
                done = False
                episode_data = []
                
                while not done:
                    # Get current state
                    n_val = obs[0]
                    current_fitness = obs[1]
                    
                    # Get action from model
                    action, _ = model.predict(obs, deterministic=True)
                    
                    # Take step
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    # Add noise to the data as specified
                    noisy_reward = reward + np.random.normal(0, expected_noise)
                    
                    # Record data
                    row = [n_val, current_fitness, action[0], noisy_reward, obs[1]]
                    writer.writerow(row)
                    episode_data.append(row)
                    
                    if done:
                        break
                
                all_data.extend(episode_data)
        
        print(f"Generated {len(episode_data)} data points for n={n}")
    
    # Write results file
    results_path = os.path.join(output_dir, 'results.csv')
    with open(results_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ground_truth'])
        writer.writerow(['DeepRL model with expected noise of 1.1'])
    
    return all_data

def main():
    parser = argparse.ArgumentParser(description='Generate LeadingOnes data using DeepRL model')
    
    # Data generation parameters
    parser.add_argument('--instance-sizes', nargs='+', type=int, default=[10, 20, 30, 40, 50],
                       help='List of problem sizes to generate')
    parser.add_argument('--output-root', type=str, required=True,
                       help='Output root directory for data')
    parser.add_argument('--data-type', choices=['continuous', 'discrete'], default='continuous',
                       help='Type of data to generate')
    parser.add_argument('--expected-noise', type=float, default=1.1,
                       help='Expected noise level (default: 1.1)')
    
    args = parser.parse_args()
    
    # Generate data
    output_dir = args.output_root
    data = generate_leadingones_deeprl_data(
        args.instance_sizes, 
        output_dir, 
        args.expected_noise, 
        args.data_type
    )
    
    print(f"Generated {len(data)} total data points")
    print(f"Data saved to {output_dir}")

if __name__ == "__main__":
    main() 