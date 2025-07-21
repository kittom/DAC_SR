#!/usr/bin/env python3
"""
Training script for DeepRL model on LeadingOnes problem
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
import gymnasium as gym
from datetime import datetime

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ddqn import DQN
from leadingones_eval import LeadingOnesEval

# Use the proper DACBench LeadingOnes environment
from dacbench.benchmarks import TheoryBenchmark

def train_deeprl_model(n=50, k=3, episodes=1000, output_dir="./output"):
    """
    Train the DeepRL model on LeadingOnes problem using DACBench
    """
    print(f"Training DeepRL model for LeadingOnes with n={n}, k={k}")
    
    # Create DACBench environment
    benchmark = TheoryBenchmark()
    env = benchmark.get_environment()
    eval_env = benchmark.get_environment()
    
    # Initialize DQN agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    os.makedirs(output_dir, exist_ok=True)
    
    agent = DQN(
        state_dim=state_dim,
        action_dim=action_dim,
        env=env,
        eval_env=eval_env,
        out_dir=output_dir,
        gamma=0.95,
        seed=42
    )
    
    # Train the agent
    print("Starting training...")
    best_model_path = agent.train(
        episodes=episodes,
        max_env_time_steps=100,
        epsilon=0.8,  # Higher initial exploration
        epsilon_decay=True,
        epsilon_decay_end_point=0.7,  # Decay for longer
        epsilon_decay_end_value=0.1,  # Keep some exploration
        eval_every_n_steps=100,
        n_eval_episodes_per_instance=5,
        save_agent_at_every_eval=True,
        max_train_time_steps=episodes * 10,
        begin_learning_after=50,  # Start learning earlier
        batch_size=64,  # Larger batch size
        log_level=1,
        use_formula=True  # Use formula for faster evaluation
    )
    
    print(f"Training completed. Best model saved to: {best_model_path}")
    return best_model_path

def generate_csv_from_model(model_path, n_values, output_path):
    """
    Generate CSV file similar to GTLeadingOnes.csv using the trained model
    """
    print(f"Generating CSV from trained model: {model_path}")
    
    # Load the trained model
    benchmark = TheoryBenchmark()
    env = benchmark.get_environment()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQN(state_dim, action_dim, env)
    agent.load(model_path)
    
    # Generate data for different n values
    data = []
    
    for n in n_values:
        print(f"Generating data for n={n}")
        # Create a new environment for this n value
        benchmark = TheoryBenchmark()
        env = benchmark.get_environment()
        
        # Set the environment to use the specific n value
        # Note: This is a simplified approach - in practice, you'd need to configure the benchmark properly
        for current_state in range(n + 1):
            # Create state (this would need to be adapted to the actual DACBench state format)
            # For now, using a simplified state representation
            state = np.array([n, current_state], dtype=np.float32)
            
            # Get action from trained model
            action = agent.act(state, epsilon=0)  # No exploration
            
            # Calculate the expected bitflip value based on the action
            # This is a simplified mapping - in reality, this would be more complex
            if action == 0:
                bitflip = n / (current_state + 1) if current_state < n else 1
            elif action == 1:
                bitflip = n / (current_state + 2) if current_state < n - 1 else 1
            else:
                bitflip = n / (current_state + 3) if current_state < n - 2 else 1
            
            # Round to integer for discrete data
            bitflip = round(bitflip)
            
            data.append([n, current_state, bitflip])
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data, columns=['n', 'k', 'leading_ones'])
    df.to_csv(output_path, index=False, header=False)
    
    print(f"CSV file generated: {output_path}")
    print(f"Data shape: {df.shape}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Train DeepRL model for LeadingOnes')
    parser.add_argument('--n', type=int, default=50, help='Problem size for training')
    parser.add_argument('--k', type=int, default=3, help='Portfolio size')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--generate-csv', action='store_true', help='Generate CSV after training')
    parser.add_argument('--n-values', nargs='+', type=int, default=[10, 20, 30, 40, 50, 100, 200, 500], 
                       help='N values for CSV generation')
    
    args = parser.parse_args()
    
    # Train the model
    best_model_path = train_deeprl_model(
        n=args.n,
        k=args.k,
        episodes=args.episodes,
        output_dir=args.output_dir
    )
    
    # Generate CSV if requested
    if args.generate_csv and best_model_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(args.output_dir, f"DeepRL_LeadingOnes_{timestamp}.csv")
        
        generate_csv_from_model(
            model_path=best_model_path,
            n_values=args.n_values,
            output_path=csv_path
        )

if __name__ == "__main__":
    main() 