import aifeynman
import pandas as pd
import numpy as np
import os
import sys
import glob

def run_aifeynman_analysis(csv_file_path):
    """
    Run AI-Feynman analysis on the provided CSV file and return the discovered equation.
    
    Args:
        csv_file_path (str): Path to the CSV file containing the dataset
        
    Returns:
        str: The discovered equation as a string
    """
    print(f"Processing CSV file: {csv_file_path}")
    
    # Get the directory and filename
    pathdir = "/home/mk422/Documents/DAC_SR/SR_algorithms/Ai_feynman/"
    
    # Convert CSV to AI-Feynman format (space-separated, no headers)
    df = pd.read_csv(csv_file_path)
    
    # Create a temporary filename for AI-Feynman
    base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    filename = f"{base_name}.txt"
    
    # Save data in AI-Feynman format
    data_path = os.path.join(pathdir, filename)
    np.savetxt(data_path, df.values, fmt='%.6f')
    
    print(f"Data saved to: {data_path}")
    print(f"Data shape: {df.shape}")
    
    # AI-Feynman parameters
    BF_try_time = 60
    BF_ops_file_type = "14ops.txt"
    polyfit_deg = 3
    NN_epochs = 500
    vars_name = ['x1', 'x2', 'y']  # Adjust based on your data
    test_percentage = 20
    
    print("Running AI-Feynman...")
    # Run AI-Feynman
    aifeynman.run_aifeynman(pathdir, filename, BF_try_time=BF_try_time, 
                           BF_ops_file_type=BF_ops_file_type, polyfit_deg=polyfit_deg, 
                           NN_epochs=NN_epochs, vars_name=vars_name, 
                           test_percentage=test_percentage)
    
    # Look for the solution file
    solution_file = os.path.join(pathdir, "results", f"solution_{base_name}")
    
    if os.path.exists(solution_file):
        print(f"Solution file found: {solution_file}")
        
        # Read the solution file
        try:
            # The solution file has columns: [test_error, log_error, log_error_all, equation, ...]
            solutions = np.loadtxt(solution_file, dtype=str, delimiter=' ')
            
            if len(solutions.shape) == 1:
                # Single solution
                solutions = solutions.reshape(1, -1)
            
            # Find the best solution (lowest test error or log error)
            best_idx = 0
            best_error = float('inf')
            
            for i, solution in enumerate(solutions):
                try:
                    # Try to get test error first, then log error
                    if len(solution) >= 4:
                        test_error = float(solution[0])
                        if test_error < best_error:
                            best_error = test_error
                            best_idx = i
                    elif len(solution) >= 2:
                        log_error = float(solution[1])
                        if log_error < best_error:
                            best_error = log_error
                            best_idx = i
                except:
                    continue
            
            # Extract the equation (usually the last column)
            best_solution = solutions[best_idx]
            equation = best_solution[-1]  # Last column contains the equation
            
            print(f"Best equation found: {equation}")
            print(f"Error: {best_error}")
            
            return equation
            
        except Exception as e:
            print(f"Error reading solution file: {e}")
            return "ERROR: Could not read solution file"
    else:
        print(f"Solution file not found: {solution_file}")
        return "ERROR: No solution file generated"

def update_results_csv(csv_file_path, equation):
    """
    Update the results.csv file with the discovered equation.
    
    Args:
        csv_file_path (str): Path to the input CSV file
        equation (str): The discovered equation
    """
    # Get the directory of the input CSV file
    csv_dir = os.path.dirname(csv_file_path)
    results_file = os.path.join(csv_dir, "results.csv")
    
    # Update or create results.csv
    if os.path.exists(results_file):
        # Read existing results
        results_df = pd.read_csv(results_file)
        
        # Add or update 'ai_feynman' column
        if 'ai_feynman' not in results_df.columns:
            results_df['ai_feynman'] = ''
        
        # Update the first row with the equation
        if len(results_df) > 0:
            results_df.loc[0, 'ai_feynman'] = equation
        else:
            # If file is empty, add a row
            new_row = pd.DataFrame({'ai_feynman': [equation]})
            results_df = pd.concat([results_df, new_row], ignore_index=True)
    else:
        # Create new results file
        results_df = pd.DataFrame({'ai_feynman': [equation]})
    
    # Save the updated results
    results_df.to_csv(results_file, index=False)
    print(f"Results updated in: {results_file}")
    return results_file

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <csv_file_path>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    try:
        # Run AI-Feynman analysis
        equation = run_aifeynman_analysis(csv_file)
        
        # Update results.csv
        results_file = update_results_csv(csv_file, equation)
        
        print(f"EQUATION: {equation}")
        print(f"Results saved to: {results_file}")
        
    except Exception as e:
        print(f"Error during AI-Feynman analysis: {e}")
        sys.exit(1)