import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

def visualize_leading_ones_data(instance_size=None, data_type='continuous'):
    """
    Visualize the LeadingOnes ground truth data.
    
    Args:
        instance_size (int, optional): If provided, shows 2D plot for that specific instance size.
                                      If None, shows 3D plot of all data.
        data_type (str): Type of data to visualize - 'continuous' or 'discrete'
    """
    
    # Load the data first to determine valid instance sizes
    data_path = os.path.join(PROJECT_ROOT, f'DataSets/Ground_Truth/LeadingOnes/{data_type}/GTLeadingOnes.csv')
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    # Read CSV without headers and assign column names
    df = pd.read_csv(data_path, header=None, names=['InstanceSize', 'CurrentState', 'Bitflip'])
    
    # Get unique instance sizes from the data
    valid_instance_sizes = sorted(df['InstanceSize'].unique())
    
    # Check if instance_size is provided and valid
    if instance_size is not None:
        if instance_size not in valid_instance_sizes:
            print(f"Error: Instance size {instance_size} not found in valid sizes: {valid_instance_sizes}")
            return
        print(f"Visualizing data for instance size: {instance_size}")
    else:
        print("Visualizing all data in 3D")
    
    # Create Visualisations directory if it doesn't exist
    viz_dir = os.path.join(PROJECT_ROOT, f'DataSets/Ground_Truth/LeadingOnes/{data_type}/Visualisations')
    os.makedirs(viz_dir, exist_ok=True)

    if instance_size is not None:
        # Filter data for specific instance size
        df_filtered = df[df['InstanceSize'] == instance_size]
        
        if df_filtered.empty:
            print(f"No data found for instance size {instance_size}")
            return
        
        # Create 2D plot
        plt.figure(figsize=(10, 6))
        
        # Plot the data points
        plt.plot(df_filtered['CurrentState'], df_filtered['Bitflip'], 'b-o', linewidth=2, markersize=4, label='Data Points')
        
        # Generate theoretical line using the equation: bitflip = instance_size / (current_state + 1)
        current_states = np.arange(0, instance_size + 1)
        theoretical_bitflips = instance_size / (current_states + 1)
        plt.plot(current_states, theoretical_bitflips, 'r--', linewidth=2, label='Theoretical: n/(s+1)')
        
        plt.xlabel('Current State')
        plt.ylabel('Bitflip Value')
        title = f'LeadingOnes ({data_type.capitalize()}): Bitflip vs Current State (Instance Size = {instance_size})'
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(viz_dir, f'LeadingOnes_{instance_size}_2D.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"2D plot saved to: {output_path}")
        
    else:
        # Create 3D plot for all data
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create scatter plot with different colors for each instance size
        colors = plt.cm.viridis(np.linspace(0, 1, len(valid_instance_sizes)))
        
        for i, size in enumerate(valid_instance_sizes):
            df_filtered = df[df['InstanceSize'] == size]
            ax.scatter(df_filtered['InstanceSize'], 
                      df_filtered['CurrentState'], 
                      df_filtered['Bitflip'], 
                      c=[colors[i]], 
                      label=f'Size {size}',
                      s=20)
        
        # Add theoretical surface lines for each instance size
        for i, size in enumerate(valid_instance_sizes):
            current_states = np.arange(0, size + 1)
            theoretical_bitflips = size / (current_states + 1)
            ax.plot([size] * len(current_states), 
                   current_states, 
                   theoretical_bitflips, 
                   c=colors[i], 
                   linewidth=2, 
                   alpha=0.7)
        
        ax.set_xlabel('Instance Size')
        ax.set_ylabel('Current State')
        ax.set_zlabel('Bitflip Value')
        title = f'LeadingOnes ({data_type.capitalize()}): 3D Visualization of All Data'
        ax.set_title(title)
        ax.legend()
        
        # Save the plot
        output_path = os.path.join(viz_dir, 'LeadingOnes_3D.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"3D plot saved to: {output_path}")
    
    plt.close()

def main():
    """Main function to handle command line arguments and run visualization."""
    
    # Default values
    instance_size = None
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
            visualize_leading_ones_data(None, data_type)
            return
        else:
            # Check if it's a number (instance size)
            try:
                instance_size = int(sys.argv[i])
                i += 1
            except ValueError:
                print(f"Warning: Ignoring non-numeric argument: {sys.argv[i]}")
                i += 1
    
    if instance_size is not None:
        visualize_leading_ones_data(instance_size, data_type)
    else:
        # No arguments provided, show 3D plot of all data
        visualize_leading_ones_data(None, data_type)

if __name__ == "__main__":
    main() 