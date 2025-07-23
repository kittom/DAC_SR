import feyn
import pandas as pd
import sys
import os
from feyn.tools import split

def run_qlattice_analysis(csv_file_path):
    """
    Run Q-Lattice analysis on the provided CSV file and return the discovered equation.
    
    Args:
        csv_file_path (str): Path to the CSV file containing the dataset
        
    Returns:
        str: The discovered equation as a string
    """
    print(f"Loading dataset from: {csv_file_path}")
    
    # Load the dataset into a pandas dataframe
    df = pd.read_csv(csv_file_path, header=None)
    
    # Set standard column names: x_1, x_2, ..., y
    num_features = len(df.columns) - 1
    feature_columns = [f'x_{i+1}' for i in range(num_features)]
    output_column = 'y'
    
    df.columns = feature_columns + [output_column]
    
    print(f"Dataset shape: {df.shape}")
    print(f"Feature columns: {feature_columns}")
    print(f"Output column: {output_column}")
    
    # Train/test split
    train, test = split(df, ratio=[0.8, 0.2])
    
    print("Instantiating QLattice...")
    # Instantiate a QLattice
    # Note: Q-Lattice uses feyn library which has built-in support for:
    # - Basic operations: +, -, *, /, **
    # - Trigonometric: sin, cos, tan, arcsin, arccos, arctan
    # - Exponential/Logarithmic: exp, log, log10
    # - Other: sqrt, abs, sign, floor, ceil, round
    # The library automatically selects appropriate functions based on data patterns
    ql = feyn.QLattice()
    
    print("Running Q-Lattice auto_run...")
    # Q-Lattice will automatically discover mathematical relationships
    # including square roots, trigonometric functions, and other complex operations
    models = ql.auto_run(
        data=train,
        output_name=output_column
    )
    
    # Select the best Model
    best = models[0]
    
    print("Best model found. Extracting symbolic expression...")
    # Get the symbolic expression
    sympy_model = best.sympify(signif=3)
    equation = str(sympy_model.as_expr())
    
    print(f"Discovered equation: {equation}")
    return equation

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <csv_file_path>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    try:
        equation = run_qlattice_analysis(csv_file)
        
        # Get the directory of the input CSV file
        csv_dir = os.path.dirname(csv_file)
        results_file = os.path.join(csv_dir, "results.csv")
        
        # Update or create results.csv
        if os.path.exists(results_file):
            # Read existing results
            results_df = pd.read_csv(results_file)
            
            # Add or update 'qlattice' column
            if 'qlattice' not in results_df.columns:
                results_df['qlattice'] = ''
            
            # Update the first row with the equation
            if len(results_df) > 0:
                results_df.loc[0, 'qlattice'] = equation
            else:
                # If file is empty, add a row
                new_row = pd.DataFrame({'qlattice': [equation]})
                results_df = pd.concat([results_df, new_row], ignore_index=True)
        else:
            # Create new results file
            results_df = pd.DataFrame({'qlattice': [equation]})
        
        # Save the updated results
        results_df.to_csv(results_file, index=False)
        print(f"Results saved to: {results_file}")
        print(f"Equation: {equation}")
        
    except Exception as e:
        print(f"Error during Q-Lattice analysis: {e}")
        sys.exit(1)