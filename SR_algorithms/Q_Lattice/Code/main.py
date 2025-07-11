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
    
    # Set column names based on the LeadingOnes dataset structure
    if len(df.columns) == 3:
        df.columns = ['n', 'k', 'leading_ones']
        output_name = 'leading_ones'
    else:
        # For other datasets, use generic column names
        df.columns = [f'col_{i}' for i in range(len(df.columns))]
        output_name = df.columns[-1]  # Use the last column as output
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Output column: {output_name}")
    
    # Train/test split
    train, test = split(df, ratio=[0.8, 0.2])
    
    print("Instantiating QLattice...")
    # Instantiate a QLattice
    ql = feyn.QLattice()
    
    print("Running Q-Lattice auto_run...")
    models = ql.auto_run(
        data=train,
        output_name=output_name
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