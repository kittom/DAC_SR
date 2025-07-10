import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

def visualize_leading_ones_data_with_noise(instance_size=None, data_file=None, noise_type=None, noise_level=None, data_type='continuous'):
    """
    Visualize the LeadingOnes ground truth data with noise.
    
    Args:
        instance_size (int, optional): If provided, shows 2D plot for that specific instance size.
                                      If None, shows 3D plot of all data.
        data_file (str): Name of the CSV file to load (if None, will try to find the file)
        noise_type (str): Type of noise used (gaussian/uniform)
        noise_level (float): Level of noise used
        data_type (str): Type of data to visualize - 'continuous' or 'discrete'
    """
    
    # If data_file is not provided, try to find the noise file
    if data_file is None:
        # Look for noise files in the directory
        noise_dir = os.path.join(PROJECT_ROOT, f'DataSets/Ground_Truth_Noise/LeadingOnes/{data_type}')
        if os.path.exists(noise_dir):
            noise_files = [f for f in os.listdir(noise_dir) if f.startswith('GTLeadingOnes_Noise_') and f.endswith('.csv')]
            if noise_files:
                data_file = noise_files[0]  # Use the first noise file found
                print(f"Using noise file: {data_file}")
            else:
                print(f"Error: No noise data files found in {noise_dir}/")
                return
        else:
            print(f"Error: {noise_dir}/ directory not found")
            return
    
    # Load the data first to determine valid instance sizes
    data_path = os.path.join(PROJECT_ROOT, f'DataSets/Ground_Truth_Noise/LeadingOnes/{data_type}/{data_file}')
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
    
    viz_dir = os.path.join(PROJECT_ROOT, f'DataSets/Ground_Truth_Noise/LeadingOnes/{data_type}/Visualisations')
    os.makedirs(viz_dir, exist_ok=True)

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
        title = f'LeadingOnes ({data_type.capitalize()}) with {noise_type.capitalize()} Noise (σ={noise_level}): Bitflip vs Current State (Instance Size = {instance_size})'
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
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
        title = f'LeadingOnes ({data_type.capitalize()}) with {noise_type.capitalize()} Noise (σ={noise_level}): 3D Visualization of All Data'
        ax.set_title(title)
        ax.legend()
        
        # Save the plot with noise info in filename
        output_filename = f'LeadingOnes_Noise_{noise_type}_{noise_level}_3D.png'
        output_path = os.path.join(viz_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"3D plot saved to: {output_path}")
    
    plt.close()

def main():
    """Main function to handle command line arguments and run visualization."""
    
    # Default values
    instance_size = None
    data_file = None
    noise_type = None
    noise_level = None
    data_type = 'continuous'
    
    # Parse command line arguments
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--data-type':
            if i + 1 < len(sys.argv):
                data_type = sys.argv[i + 1]
                if data_type not in ['continuous', 'discrete']:
                    print("Error: data_type must be 'continuous' or 'discrete'")
                    sys.exit(1)
                i += 2
            else:
                print("Error: --data-type requires a value")
                sys.exit(1)
        elif sys.argv[i] == 'all':
            # Show 3D plot of all data
            visualize_leading_ones_data_with_noise(None, data_file, noise_type, noise_level, data_type)
            return
        else:
            # Check if it's a number (instance size)
            try:
                instance_size = int(sys.argv[i])
                i += 1
                # Get additional parameters if available
                if i < len(sys.argv):
                    data_file = sys.argv[i]
                    i += 1
                if i < len(sys.argv):
                    noise_type = sys.argv[i]
                    i += 1
                if i < len(sys.argv):
                    noise_level = sys.argv[i]
                    i += 1
            except ValueError:
                print(f"Warning: Ignoring non-numeric argument: {sys.argv[i]}")
                i += 1
    
    if instance_size is not None:
        visualize_leading_ones_data_with_noise(instance_size, data_file, noise_type, noise_level, data_type)
    else:
        # No arguments provided, show 3D plot of all data
        visualize_leading_ones_data_with_noise(None, data_file, noise_type, noise_level, data_type)

if __name__ == "__main__":
    main() 