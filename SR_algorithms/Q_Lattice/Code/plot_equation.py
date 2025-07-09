import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D

# Define the sympy equation from the QLattice output
def create_equation():
    # Define symbols
    InstanceSize, CurrentState = sp.symbols('InstanceSize CurrentState')
    
    # The equation from the QLattice output
    equation = 0.0147 - 7.15*(-0.00942*InstanceSize - 0.000508)/(0.0665*CurrentState + 0.0659)
    
    return equation, InstanceSize, CurrentState

def plot_equation_3d():
    # Create the equation
    equation, InstanceSize, CurrentState = create_equation()
    
    # Convert to numpy function for plotting
    f = sp.lambdify((InstanceSize, CurrentState), equation, 'numpy')
    
    # Create meshgrid for plotting
    x = np.linspace(10, 100, 50)  # InstanceSize range
    y = np.linspace(0, 100, 50)   # CurrentState range
    X, Y = np.meshgrid(x, y)
    
    # Calculate Z values
    Z = f(X, Y)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Set labels
    ax.set_xlabel('InstanceSize')
    ax.set_ylabel('CurrentState')
    ax.set_zlabel('Bitflip (Predicted)')
    ax.set_title('QLattice Symbolic Regression Model\nBitflip = f(InstanceSize, CurrentState)')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('Plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return equation

def plot_equation_2d():
    # Create the equation
    equation, InstanceSize, CurrentState = create_equation()
    
    # Convert to numpy function for plotting
    f = sp.lambdify((InstanceSize, CurrentState), equation, 'numpy')
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Fixed InstanceSize, varying CurrentState
    instance_sizes = [10, 30, 50, 100]
    current_states = np.linspace(0, 100, 100)
    
    for size in instance_sizes:
        bitflips = f(size, current_states)
        ax1.plot(current_states, bitflips, label=f'InstanceSize={size}')
    
    ax1.set_xlabel('CurrentState')
    ax1.set_ylabel('Bitflip (Predicted)')
    ax1.set_title('Bitflip vs CurrentState (Fixed InstanceSize)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Fixed CurrentState, varying InstanceSize
    current_states_fixed = [0, 25, 50, 75]
    instance_sizes_range = np.linspace(10, 100, 100)
    
    for state in current_states_fixed:
        bitflips = f(instance_sizes_range, state)
        ax2.plot(instance_sizes_range, bitflips, label=f'CurrentState={state}')
    
    ax2.set_xlabel('InstanceSize')
    ax2.set_ylabel('Bitflip (Predicted)')
    ax2.set_title('Bitflip vs InstanceSize (Fixed CurrentState)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Heatmap
    x = np.linspace(10, 100, 50)
    y = np.linspace(0, 100, 50)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    
    im = ax3.contourf(X, Y, Z, levels=20, cmap='viridis')
    ax3.set_xlabel('InstanceSize')
    ax3.set_ylabel('CurrentState')
    ax3.set_title('Bitflip Heatmap')
    fig.colorbar(im, ax=ax3)
    
    # Plot 4: Actual vs Predicted (if we have actual data)
    try:
        # Load actual data for comparison
        import pandas as pd
        df = pd.read_csv('../../../DataSets/Ground_Truth/GTLeadingOnes.csv')
        
        # Calculate predictions
        predictions = f(df['InstanceSize'], df['CurrentState'])
        
        ax4.scatter(df['Bitflip'], predictions, alpha=0.6)
        ax4.plot([df['Bitflip'].min(), df['Bitflip'].max()], 
                [df['Bitflip'].min(), df['Bitflip'].max()], 'r--', label='Perfect Prediction')
        ax4.set_xlabel('Actual Bitflip')
        ax4.set_ylabel('Predicted Bitflip')
        ax4.set_title('Actual vs Predicted Bitflip')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
    except Exception as e:
        ax4.text(0.5, 0.5, f'Could not load actual data\nfor comparison: {str(e)}', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Actual vs Predicted (Data not available)')
    
    plt.tight_layout()
    plt.savefig('Plot_2D.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Creating 3D plot...")
    equation = plot_equation_3d()
    print(f"Equation: {equation}")
    
    print("\nCreating 2D plots...")
    plot_equation_2d()
    
    print("Plots saved as 'Plot.png' and 'Plot_2D.png'") 