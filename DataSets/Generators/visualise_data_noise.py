import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def visualize_leading_ones_data_with_noise(instance_size=None, data_file=None, noise_type=None, noise_level=None):
    """
    Visualize the LeadingOnes ground truth data with noise.
    
    Args:
        instance_size (int, optional): If provided, shows 2D plot for that specific instance size.
                                      If None, shows 3D plot of all data.
        data_file (str): Name of the CSV file to load (if None, will try to find the file)
        noise_type (str): Type of noise used (gaussian/uniform)
        noise_level (float): Level of noise used
    """
    
    # If data_file is not provided, try to find the noise file
    if data_file is None:
        # Look for noise files in the directory
        noise_dir = '../Ground_Truth_Noise'
        if os.path.exists(noise_dir):
            noise_files = [f for f in os.listdir(noise_dir) if f.startswith('GTLeadingOnes_Noise_') and f.endswith('.csv')]
            if noise_files:
                data_file = noise_files[0]  # Use the first noise file found
                print(f"Using noise file: {data_file}")
            else:
                print("Error: No noise data files found in ../Ground_Truth_Noise/")
                return
        else:
            print("Error: ../Ground_Truth_Noise/ directory not found")
            return
    
    # Load the data first to determine valid instance sizes
    data_path = f'../Ground_Truth_Noise/{data_file}'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    # Extract noise info from filename if not provided
    if noise_type is None or noise_level is None:
        # Parse filename: GTLeadingOnes_Noise_{type}_{level}.csv
        filename_parts = data_file.replace('.csv', '').split('_')
        if len(filename_parts) >= 4:
            noise_type = filename_parts[2]
            noise_level = filename_parts[3]
    
    # Read CSV without headers and assign column names
    # For noisy data, we expect: InstanceSize, CurrentState, Bitflip
    df = pd.read_csv(data_path, header=None, 
                     names=['InstanceSize', 'CurrentState', 'Bitflip'])
    
    # Get unique instance sizes from the data
    valid_instance_sizes = sorted(df['InstanceSize'].unique())
    
    # Check if instance_size is provided and valid
    if instance_size is not None:
        if instance_size not in valid_instance_sizes:
            print(f"Error: Instance size {instance_size} not found in valid sizes: {valid_instance_sizes}")
            return
        print(f"Visualizing noisy data for instance size: {instance_size}")
    else:
        print("Visualizing all noisy data in 3D")
    
    if instance_size is not None:
        # Filter data for specific instance size
        df_filtered = df[df['InstanceSize'] == instance_size]
        
        if df_filtered.empty:
            print(f"No data found for instance size {instance_size}")
            return
        
        # Create 2D plot
        plt.figure(figsize=(12, 8))
        
        # Plot the noisy data points
        plt.scatter(df_filtered['CurrentState'], df_filtered['Bitflip'], 
                   c='blue', alpha=0.6, s=30, label='Noisy Data Points')
        
        # Plot the theoretical clean line
        current_states = np.arange(0, instance_size + 1)
        theoretical_bitflips = instance_size / (current_states + 1)
        plt.plot(current_states, theoretical_bitflips, 'r-', linewidth=3, 
                label='Theoretical: n/(s+1)')
        
        plt.xlabel('Current State')
        plt.ylabel('Bitflip Value')
        title = f'LeadingOnes with {noise_type.capitalize()} Noise (σ={noise_level}): Bitflip vs Current State (Instance Size = {instance_size})'
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Create Visualisations directory if it doesn't exist
        viz_dir = '../Ground_Truth_Noise/Visualisations'
        os.makedirs(viz_dir, exist_ok=True)
        
        # Save the plot with noise info in filename
        output_filename = f'LeadingOnes_Noise_{noise_type}_{noise_level}_{instance_size}_2D.png'
        output_path = os.path.join(viz_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"2D plot saved to: {output_path}")
        
    else:
        # Create 3D plot for all data
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create scatter plot with different colors for each instance size
        colors = plt.cm.viridis(np.linspace(0, 1, len(valid_instance_sizes)))
        
        for i, size in enumerate(valid_instance_sizes):
            df_filtered = df[df['InstanceSize'] == size]
            
            # Plot noisy data points
            ax.scatter(df_filtered['InstanceSize'], 
                      df_filtered['CurrentState'], 
                      df_filtered['Bitflip'], 
                      c=[colors[i]], 
                      label=f'Size {size} (Noisy)',
                      s=15, alpha=0.6)
            
            # Plot theoretical surface lines
            current_states = np.arange(0, size + 1)
            theoretical_bitflips = size / (current_states + 1)
            ax.plot([size] * len(current_states), 
                   current_states, 
                   theoretical_bitflips, 
                   c=colors[i], 
                   linewidth=3, 
                   alpha=0.8,
                   linestyle='--')
        
        ax.set_xlabel('Instance Size')
        ax.set_ylabel('Current State')
        ax.set_zlabel('Bitflip Value')
        title = f'LeadingOnes with {noise_type.capitalize()} Noise (σ={noise_level}): 3D Visualization of All Data'
        ax.set_title(title)
        ax.legend()
        
        # Create Visualisations directory if it doesn't exist
        viz_dir = '../Ground_Truth_Noise/Visualisations'
        os.makedirs(viz_dir, exist_ok=True)
        
        # Save the plot with noise info in filename
        output_filename = f'LeadingOnes_Noise_{noise_type}_{noise_level}_3D.png'
        output_path = os.path.join(viz_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"3D plot saved to: {output_path}")
    
    plt.show()

def main():
    """Main function to handle command line arguments and run visualization."""
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == 'all':
            # Show 3D plot of all data
            data_file = sys.argv[2] if len(sys.argv) > 2 else None
            noise_type = sys.argv[3] if len(sys.argv) > 3 else None
            noise_level = sys.argv[4] if len(sys.argv) > 4 else None
            visualize_leading_ones_data_with_noise(None, data_file, noise_type, noise_level)
        else:
            try:
                # Try to parse as instance size
                instance_size = int(sys.argv[1])
                data_file = sys.argv[2] if len(sys.argv) > 2 else None
                noise_type = sys.argv[3] if len(sys.argv) > 3 else None
                noise_level = sys.argv[4] if len(sys.argv) > 4 else None
                visualize_leading_ones_data_with_noise(instance_size, data_file, noise_type, noise_level)
            except ValueError:
                # Try to get valid instance sizes from data for better error message
                try:
                    noise_dir = '../Ground_Truth_Noise'
                    if os.path.exists(noise_dir):
                        noise_files = [f for f in os.listdir(noise_dir) if f.startswith('GTLeadingOnes_Noise_') and f.endswith('.csv')]
                        if noise_files:
                            data_path = os.path.join(noise_dir, noise_files[0])
                            df = pd.read_csv(data_path, header=None, 
                                           names=['InstanceSize', 'CurrentState', 'Bitflip'])
                            valid_sizes = sorted(df['InstanceSize'].unique())
                            print("Usage: python visualise_data_noise.py [instance_size|all] [data_file] [noise_type] [noise_level]")
                            print(f"  instance_size: One of {valid_sizes}")
                            print("  all: Show 3D plot of all data")
                            print("  data_file: Name of the CSV file (optional)")
                            print("  noise_type: Type of noise (optional)")
                            print("  noise_level: Level of noise (optional)")
                            print("  (no argument): Show 3D plot of all data")
                        else:
                            print("Usage: python visualise_data_noise.py [instance_size|all] [data_file] [noise_type] [noise_level]")
                            print("  instance_size: Integer value from the data file")
                            print("  all: Show 3D plot of all data")
                            print("  data_file: Name of the CSV file (optional)")
                            print("  noise_type: Type of noise (optional)")
                            print("  noise_level: Level of noise (optional)")
                    else:
                        print("Usage: python visualise_data_noise.py [instance_size|all] [data_file] [noise_type] [noise_level]")
                        print("  instance_size: Integer value from the data file")
                        print("  all: Show 3D plot of all data")
                        print("  data_file: Name of the CSV file (optional)")
                        print("  noise_type: Type of noise (optional)")
                        print("  noise_level: Level of noise (optional)")
                except:
                    print("Usage: python visualise_data_noise.py [instance_size|all] [data_file] [noise_type] [noise_level]")
                    print("  instance_size: Integer value from the data file")
                    print("  all: Show 3D plot of all data")
                    print("  data_file: Name of the CSV file (optional)")
                    print("  noise_type: Type of noise (optional)")
                    print("  noise_level: Level of noise (optional)")
    else:
        # No arguments provided, show 3D plot of all data
        visualize_leading_ones_data_with_noise()

if __name__ == "__main__":
    main() 