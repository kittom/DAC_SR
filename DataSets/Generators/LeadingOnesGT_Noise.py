import pandas as pd
import numpy as np
import os
import sys

def generate_leading_ones_ground_truth_with_noise(instance_sizes=None, noise_level=0.1, noise_type='gaussian'):
    """
    Generate ground truth data for LeadingOnes problem with added noise.
    
    Creates a CSV file with columns:
    - InstanceSize: The size of the problem instance
    - CurrentState: All possible states from 0 to InstanceSize
    - Bitflip: Calculated as (instance_size/ (current_state + 1)) + noise
    - TheoreticalBitflip: The clean theoretical value without noise
    
    Args:
        instance_sizes (list, optional): List of instance sizes to generate data for.
                                        If None, uses default sizes [10, 20, 30, 40, 50, 100]
        noise_level (float): Standard deviation of the noise (default: 0.1)
        noise_type (str): Type of noise - 'gaussian' or 'uniform' (default: 'gaussian')
    """
    
    # Define the instance sizes to cycle through
    if instance_sizes is None:
        instance_sizes = [10, 20, 30, 40, 50, 100]
    
    # Lists to store the data
    instance_size_data = []
    current_state_data = []
    bitflip_data = []
    theoretical_bitflip_data = []
    
    # Generate data for each instance size
    for instance_size in instance_sizes:
        # For each instance size, generate all possible current states
        for current_state in range(instance_size + 1):
            # Calculate theoretical bitflip value
            theoretical_bitflip = (instance_size / (current_state + 1))
            
            # Add noise based on the specified type
            if noise_type == 'gaussian':
                noise = np.random.normal(0, noise_level)
            elif noise_type == 'uniform':
                noise = np.random.uniform(-noise_level, noise_level)
            else:
                raise ValueError("noise_type must be 'gaussian' or 'uniform'")
            
            # Calculate noisy bitflip value
            noisy_bitflip = theoretical_bitflip + noise
            
            # Ensure bitflip values are positive (noise might make them negative)
            noisy_bitflip = max(0.001, noisy_bitflip)
            
            # Append data to lists
            instance_size_data.append(instance_size)
            current_state_data.append(current_state)
            bitflip_data.append(noisy_bitflip)
            theoretical_bitflip_data.append(theoretical_bitflip)
    
    # Create DataFrame (without theoretical bitflip column)
    df = pd.DataFrame({
        'InstanceSize': instance_size_data,
        'CurrentState': current_state_data,
        'Bitflip': bitflip_data
    })
    
    # Create Ground_Truth_Noise directory if it doesn't exist
    noise_dir = '../Ground_Truth_Noise'
    os.makedirs(noise_dir, exist_ok=True)
    
    # Save to CSV file without headers, with noise level in filename
    output_filename = f'GTLeadingOnes_Noise_{noise_type}_{noise_level}.csv'
    output_path = os.path.join(noise_dir, output_filename)
    df.to_csv(output_path, index=False, header=False)
    
    print(f"Ground truth data with noise generated successfully!")
    print(f"Noise type: {noise_type}, Noise level: {noise_level}")
    print(f"Output saved to: {output_path}")
    print(f"Total rows: {len(df)}")
    print(f"Instance sizes: {instance_sizes}")
    print(f"Sample data (first 10 rows):")
    print(df.head(10))
    
    # Print noise statistics (calculate from the original theoretical values)
    noise_values = []
    for i, row in df.iterrows():
        theoretical = row['InstanceSize'] / (row['CurrentState'] + 1)
        noise_values.append(row['Bitflip'] - theoretical)
    
    noise_values = np.array(noise_values)
    print(f"\nNoise Statistics:")
    print(f"Mean noise: {noise_values.mean():.6f}")
    print(f"Std noise: {noise_values.std():.6f}")
    print(f"Min noise: {noise_values.min():.6f}")
    print(f"Max noise: {noise_values.max():.6f}")
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate LeadingOnes ground truth data with noise')
    parser.add_argument('instance_sizes', nargs='*', type=int, 
                       help='Instance sizes to generate data for (default: 10 20 30 40 50 100)')
    parser.add_argument('--noise-level', type=float, default=0.1,
                       help='Noise level (standard deviation for gaussian, range for uniform) (default: 0.1)')
    parser.add_argument('--noise-type', choices=['gaussian', 'uniform'], default='gaussian',
                       help='Type of noise to add (default: gaussian)')
    
    args = parser.parse_args()
    
    # Use provided instance sizes or default
    if args.instance_sizes:
        instance_sizes = args.instance_sizes
        print(f"Generating data for instance sizes: {instance_sizes}")
    else:
        instance_sizes = None
        print("Using default instance sizes: [10, 20, 30, 40, 50, 100]")
    
    # Generate the ground truth data with noise
    df = generate_leading_ones_ground_truth_with_noise(
        instance_sizes=instance_sizes,
        noise_level=args.noise_level,
        noise_type=args.noise_type
    ) 