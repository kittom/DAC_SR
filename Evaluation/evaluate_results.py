#!/usr/bin/env python3
"""
Evaluation script for symbolic regression results.
Reads results.csv files, standardizes variable names, and creates visualizations.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sympy as sp
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

class SymbolicRegressionEvaluator:
    def __init__(self, results_file_path):
        """
        Initialize the evaluator with a results file path.
        
        Args:
            results_file_path (str): Path to the results.csv or results_rounding.csv file
        """
        self.results_file_path = Path(results_file_path)
        self.results_dir = self.results_file_path.parent
        self.equations = {}
        self.original_data = None
        self.is_rounding = 'rounding' in str(self.results_file_path)
        
        # Create output directory
        if self.is_rounding:
            self.output_dir = self.results_dir / 'rounded_results'
        else:
            self.output_dir = self.results_dir / 'results'
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Initialized evaluator for: {self.results_file_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Is rounding version: {self.is_rounding}")
    
    def load_results(self):
        """Load equations from the results file."""
        print(f"\nLoading results from: {self.results_file_path}")
        
        try:
            # Read the results file
            df = pd.read_csv(self.results_file_path)
            
            # Get the first row (equations)
            if len(df) > 0:
                for column in df.columns:
                    equation = df.iloc[0][column]
                    if pd.notna(equation) and equation.strip():
                        self.equations[column] = equation.strip()
                        print(f"  {column}: {equation}")
            
            print(f"Loaded {len(self.equations)} equations")
            
        except Exception as e:
            print(f"Error loading results: {e}")
            return False
        
        return True
    
    def load_original_data(self):
        """Load the original dataset to understand variable ranges."""
        print(f"\nLoading original data...")
        
        # Look for the original data file
        possible_names = ['GTLeadingOnes.csv', 'data.csv', 'dataset.csv']
        data_file = None
        
        for name in possible_names:
            potential_file = self.results_dir / name
            if potential_file.exists():
                data_file = potential_file
                break
        
        if data_file is None:
            print("Warning: Could not find original data file")
            return False
        
        try:
            # Load the data (assuming no header, comma-separated)
            self.original_data = pd.read_csv(data_file, header=None)
            
            # Determine number of features (all columns except the last)
            n_features = len(self.original_data.columns) - 1
            print(f"Original data shape: {self.original_data.shape}")
            print(f"Number of features: {n_features}")
            print(f"Feature columns: 0 to {n_features-1}")
            print(f"Target column: {n_features}")
            
            # Show data ranges
            for i in range(n_features):
                col_data = self.original_data.iloc[:, i]
                print(f"  Feature {i}: min={col_data.min()}, max={col_data.max()}, mean={col_data.mean():.2f}")
            
            return True
            
        except Exception as e:
            print(f"Error loading original data: {e}")
            return False
    
    def standardize_variable_names(self, equation_str):
        """
        Standardize variable names in equations.
        Convert various formats (x0, x1, x_0, x_1, etc.) to standard format.
        """
        if not equation_str or pd.isna(equation_str):
            return equation_str
        
        # Remove quotes if present
        equation_str = str(equation_str).strip('"\'')
        
        # Step 1: Handle KAN format (x_1, x_2, etc.) first
        # Find all x_1, x_2 patterns and map them to x0, x1, etc.
        x_underscore_pattern = r'x_(\d+)'
        x_underscore_matches = re.findall(x_underscore_pattern, equation_str)
        
        if x_underscore_matches:
            # Create mapping for x_ variables
            for i, match in enumerate(sorted(set(x_underscore_matches))):
                old_var = f'x_{match}'
                new_var = f'x{i}'
                equation_str = re.sub(r'\b' + old_var + r'\b', new_var, equation_str)
        
        # Step 2: Handle k, n variables
        if 'k' in equation_str or 'n' in equation_str:
            # Find existing x variables to determine next index
            existing_x_vars = re.findall(r'x(\d+)', equation_str)
            next_index = len(set(existing_x_vars)) if existing_x_vars else 0
            
            if 'k' in equation_str:
                equation_str = re.sub(r'\bk\b', f'x{next_index}', equation_str)
                next_index += 1
            
            if 'n' in equation_str:
                equation_str = re.sub(r'\bn\b', f'x{next_index}', equation_str)
        
        # Step 3: Ensure consistent x0, x1, x2 ordering
        # Find all x variables and reorder them
        all_x_vars = re.findall(r'x(\d+)', equation_str)
        unique_x_vars = sorted(list(set(all_x_vars)))
        
        if len(unique_x_vars) > 1:
            # Create final mapping to ensure x0, x1, x2, etc. order
            final_mapping = {}
            for i, var_num in enumerate(unique_x_vars):
                old_var = f'x{var_num}'
                new_var = f'x{i}'
                if old_var != new_var:
                    final_mapping[old_var] = new_var
            
            # Apply final mapping
            for old_var, new_var in final_mapping.items():
                equation_str = re.sub(r'\b' + old_var + r'\b', new_var, equation_str)
        
        return equation_str
    
    def get_equation_complexity(self, equation_str):
        """Calculate a simple complexity score for the equation."""
        if not equation_str:
            return 0
        
        # Count operators and functions
        operators = ['+', '-', '*', '/', '**', '^']
        functions = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'round']
        
        complexity = 0
        
        # Count operators
        for op in operators:
            complexity += equation_str.count(op)
        
        # Count functions
        for func in functions:
            complexity += equation_str.count(func)
        
        # Add base complexity for length
        complexity += len(equation_str) * 0.1
        
        return complexity
    
    def create_line_plots(self):
        """Create line plots for equations with less than 3 dimensions."""
        print(f"\nCreating line plots...")
        
        if not self.original_data is not None:
            print("Error: Original data not loaded")
            return False
        
        n_features = len(self.original_data.columns) - 1
        
        if n_features > 2:
            print(f"Warning: {n_features} features detected. Line plots work best with ≤2 features.")
            print("Creating 2D projections instead...")
        
        # Create plots for each equation
        for algorithm, equation in self.equations.items():
            print(f"\nProcessing {algorithm}: {equation}")
            
            try:
                # Standardize variable names
                standardized_eq = self.standardize_variable_names(equation)
                print(f"  Standardized: {standardized_eq}")
                
                # Create the plot
                self._create_single_plot(algorithm, standardized_eq, n_features)
                
            except Exception as e:
                print(f"  Error processing {algorithm}: {e}")
                continue
        
        return True
    
    def _create_single_plot(self, algorithm, equation, n_features):
        """Create a single line plot for an equation."""
        
        # Set up the plot
        plt.figure(figsize=(12, 8))
        
        if n_features == 1:
            # 1D plot: y vs x0
            self._plot_1d(algorithm, equation)
        elif n_features == 2:
            # 2D plot: y vs x0, x1 (surface or contour)
            self._plot_2d(algorithm, equation)
        else:
            # Multi-dimensional: create 2D projections
            self._plot_2d_projection(algorithm, equation, n_features)
        
        # Save the plot
        plot_filename = f"{algorithm}_plot.png"
        plot_path = self.output_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved plot: {plot_path}")
    
    def _plot_1d(self, algorithm, equation):
        """Create a 1D line plot."""
        try:
            # Parse the equation
            x0 = sp.Symbol('x0')
            expr = sp.sympify(equation)
            
            # Create x values
            x_data = self.original_data.iloc[:, 0]
            x_range = np.linspace(x_data.min(), x_data.max(), 100)
            
            # Evaluate the equation
            y_values = []
            for x_val in x_range:
                try:
                    y_val = expr.subs(x0, x_val)
                    if y_val.is_real:
                        y_values.append(float(y_val))
                    else:
                        y_values.append(np.nan)
                except:
                    y_values.append(np.nan)
            
            # Plot
            plt.plot(x_range, y_values, label=algorithm, linewidth=2)
            plt.xlabel('x0')
            plt.ylabel('y')
            plt.title(f'{algorithm} Equation: {equation}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        except Exception as e:
            plt.text(0.5, 0.5, f'Error plotting {algorithm}:\n{str(e)}', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'{algorithm} - Plot Error')
    
    def _plot_2d(self, algorithm, equation):
        """Create a 2D plot showing y as the output of the equation."""
        try:
            # Parse the equation
            x0, x1 = sp.Symbol('x0'), sp.Symbol('x1')
            expr = sp.sympify(equation)
            
            # Get data ranges
            x0_data = self.original_data.iloc[:, 0]
            x1_data = self.original_data.iloc[:, 1]
            
            x0_range = np.linspace(x0_data.min(), x0_data.max(), 50)
            x1_range = np.linspace(x1_data.min(), x1_data.max(), 50)
            
            X0, X1 = np.meshgrid(x0_range, x1_range)
            Z = np.zeros_like(X0)
            
            # Evaluate the equation
            for i in range(X0.shape[0]):
                for j in range(X0.shape[1]):
                    try:
                        val = expr.subs([(x0, X0[i, j]), (x1, X1[i, j])])
                        if val.is_real:
                            Z[i, j] = float(val)
                        else:
                            Z[i, j] = np.nan
                    except:
                        Z[i, j] = np.nan
            
            # Create subplots: 3D surface and line plots
            fig = plt.figure(figsize=(16, 10))
            
            # 3D Surface plot
            ax1 = fig.add_subplot(2, 2, 1, projection='3d')
            surf = ax1.plot_surface(X0, X1, Z, cmap='viridis', alpha=0.8)
            ax1.set_xlabel('x0')
            ax1.set_ylabel('x1')
            ax1.set_zlabel('y (equation output)')
            ax1.set_title(f'{algorithm} - 3D Surface')
            fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
            
            # Line plot: y vs x0 with x1 fixed at mean
            ax2 = fig.add_subplot(2, 2, 2)
            x1_fixed = x1_data.mean()
            y_values_x0 = []
            for x0_val in x0_range:
                try:
                    y_val = expr.subs([(x0, x0_val), (x1, x1_fixed)])
                    if y_val.is_real:
                        y_values_x0.append(float(y_val))
                    else:
                        y_values_x0.append(np.nan)
                except:
                    y_values_x0.append(np.nan)
            
            ax2.plot(x0_range, y_values_x0, 'b-', linewidth=2, label=f'x1 = {x1_fixed:.2f}')
            ax2.set_xlabel('x0')
            ax2.set_ylabel('y')
            ax2.set_title(f'{algorithm} - y vs x0 (x1 fixed)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Line plot: y vs x1 with x0 fixed at mean
            ax3 = fig.add_subplot(2, 2, 3)
            x0_fixed = x0_data.mean()
            y_values_x1 = []
            for x1_val in x1_range:
                try:
                    y_val = expr.subs([(x0, x0_fixed), (x1, x1_val)])
                    if y_val.is_real:
                        y_values_x1.append(float(y_val))
                    else:
                        y_values_x1.append(np.nan)
                except:
                    y_values_x1.append(np.nan)
            
            ax3.plot(x1_range, y_values_x1, 'r-', linewidth=2, label=f'x0 = {x0_fixed:.2f}')
            ax3.set_xlabel('x1')
            ax3.set_ylabel('y')
            ax3.set_title(f'{algorithm} - y vs x1 (x0 fixed)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Contour plot for reference
            ax4 = fig.add_subplot(2, 2, 4)
            contour = ax4.contourf(X0, X1, Z, levels=20, cmap='viridis')
            ax4.set_xlabel('x0')
            ax4.set_ylabel('x1')
            ax4.set_title(f'{algorithm} - Contour (y values)')
            fig.colorbar(contour, ax=ax4)
            
            # Add equation text
            fig.suptitle(f'{algorithm} Equation: {equation}', fontsize=12, y=0.98)
            plt.tight_layout()
            
        except Exception as e:
            plt.figure(figsize=(12, 8))
            plt.text(0.5, 0.5, f'Error plotting {algorithm}:\n{str(e)}', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'{algorithm} - Plot Error')
    
    def _plot_2d_projection(self, algorithm, equation, n_features):
        """Create a 2D projection for multi-dimensional data."""
        try:
            # For now, just show the equation text
            plt.text(0.5, 0.5, f'{algorithm}\n{equation}\n\n{n_features}D data - 2D projection not implemented yet', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
            plt.title(f'{algorithm} - {n_features}D Equation')
            plt.axis('off')
            
        except Exception as e:
            plt.text(0.5, 0.5, f'Error plotting {algorithm}:\n{str(e)}', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'{algorithm} - Plot Error')
    
    def run_evaluation(self):
        """Run the complete evaluation pipeline."""
        print("=== Symbolic Regression Results Evaluation ===\n")
        
        # Step 1: Load results
        if not self.load_results():
            return False
        
        # Step 2: Load original data
        if not self.load_original_data():
            return False
        
        # Step 3: Create plots
        if not self.create_line_plots():
            return False
        
        print(f"\n=== Evaluation Complete ===")
        print(f"Results saved in: {self.output_dir}")
        
        return True

def main():
    """Main function to run the evaluation."""
    if len(sys.argv) != 2:
        print("Usage: python evaluate_results.py <path_to_results_file>")
        print("Example: python evaluate_results.py DataSets/Ground_Truth/LeadingOnes/continuous/results.csv")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    if not os.path.exists(results_file):
        print(f"Error: Results file '{results_file}' not found!")
        sys.exit(1)
    
    # Create and run evaluator
    evaluator = SymbolicRegressionEvaluator(results_file)
    success = evaluator.run_evaluation()
    
    if success:
        print("\n✅ Evaluation completed successfully!")
    else:
        print("\n❌ Evaluation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 