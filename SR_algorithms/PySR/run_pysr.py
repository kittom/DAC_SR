import sys
import os
import pandas as pd
import numpy as np
import glob

try:
    from pysr import PySRRegressor
except ImportError:
    print("PySR is not installed in this environment. Please install it with 'pip install pysr'.")
    sys.exit(1)

def extract_best_equation_from_hof(hof_path):
    try:
        hof_df = pd.read_csv(hof_path)
        if 'Loss' in hof_df.columns and 'Equation' in hof_df.columns:
            best_row = hof_df.loc[hof_df['Loss'].idxmin()]
            return str(best_row['Equation'])
        else:
            print(f"hall_of_fame.csv at {hof_path} missing required columns.")
            return "ERROR: hall_of_fame.csv missing columns"
    except Exception as e:
        print(f"Error reading hall_of_fame.csv: {e}")
        return "ERROR: Could not read hall_of_fame.csv"

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_pysr.py <csv_file_path>")
        sys.exit(1)

    csv_file = sys.argv[1]
    if not os.path.isfile(csv_file):
        print(f"Error: CSV file '{csv_file}' not found!")
        sys.exit(1)

    # Load the dataset
    try:
        data = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    # Assume the last column is the target, the rest are features
    if data.shape[1] < 2:
        print("Error: CSV file must have at least one feature column and one target column.")
        sys.exit(1)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Run PySR symbolic regression
    print("Running PySR symbolic regression...")
    try:
        model = PySRRegressor(
            niterations=40,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "exp", "log"],
            progress=True,
            model_selection="best",
            maxsize=20,
            procs=0,  # Use all available cores
            output_directory="outputs"
        )
        model.fit(X, y)
    except Exception as e:
        print(f"Error running PySR: {e}")
        sys.exit(1)

    # Find the latest hall_of_fame.csv in the outputs directory
    outputs_dir = os.path.join(os.getcwd(), "outputs")
    hof_path = None
    if os.path.isdir(outputs_dir):
        # Find all hall_of_fame.csv files in subdirectories
        hof_files = glob.glob(os.path.join(outputs_dir, "*/hall_of_fame.csv"))
        if hof_files:
            # Use the most recently modified one
            hof_path = max(hof_files, key=os.path.getmtime)
            print(f"Found hall_of_fame.csv at: {hof_path}")
    
    if hof_path:
        equation = extract_best_equation_from_hof(hof_path)
        print(f"Best symbolic equation from hall_of_fame.csv: {equation}")
    else:
        print("No hall_of_fame.csv found. Falling back to model.equations_ (may be empty)...")
        try:
            equations_df = model.equations_
            if equations_df is not None and len(equations_df) > 0:
                best_row = equations_df.iloc[[0]]
                equation = best_row.iloc[0]['equation']
                print(f"Best symbolic equation found: {equation}")
            else:
                print("No equations found by PySR.")
                equation = "ERROR: No equation found"
        except Exception as e:
            print(f"Error extracting equation: {e}")
            equation = "ERROR: Could not extract equation"

    # Update or create results.csv in the same directory as the input CSV
    csv_dir = os.path.dirname(csv_file)
    results_file = os.path.join(csv_dir, "results.csv")
    try:
        if os.path.exists(results_file):
            results_df = pd.read_csv(results_file)
            if 'pysr' not in results_df.columns:
                results_df['pysr'] = ''
            # Always update the first row
            if len(results_df) > 0:
                results_df.at[0, 'pysr'] = equation
            else:
                # If the file is empty but has columns, create a single row
                empty_row = {col: '' for col in results_df.columns}
                empty_row['pysr'] = equation
                results_df = pd.DataFrame([empty_row])
        else:
            # If the file doesn't exist, create a single-row DataFrame with only pysr
            results_df = pd.DataFrame({'pysr': [equation]})
        results_df.to_csv(results_file, index=False)
        print(f"Results saved to: {results_file}")
    except Exception as e:
        print(f"Error updating results file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 