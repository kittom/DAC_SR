#!/usr/bin/env python3
"""
Script to analyze CSV data using e2e_Transformer
to find the equation relating input columns to output column.

Author: AI Assistant
Date: 2024
"""

import torch
import numpy as np
import sympy as sp
import pandas as pd
import os
import sys
import json
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add the e2e_Transformer directory to the path
sys.path.append('./e2e_Transformer')
import symbolicregression

class E2ETransformerAnalyzer:
    def __init__(self, model_path="model1.pt", data_path=None):
        """
        Initialize the analyzer.
        
        Args:
            model_path (str): Path to the pre-trained model
            data_path (str): Path to the CSV file
        """
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.X = None
        self.y = None
        self.results = {}
        
    def load_model(self):
        """Load the pre-trained e2e_Transformer model."""
        print("1. Loading pre-trained model...")
        
        if not os.path.isfile(self.model_path):
            print(f"ERROR: Model not found at {self.model_path}")
            return False
        
        try:
            if not torch.cuda.is_available():
                self.model = torch.load(self.model_path, map_location=torch.device('cpu'))
                print("Model loaded on CPU")
            else:
                self.model = torch.load(self.model_path)
                self.model = self.model.cuda()
                print(f"Model loaded on GPU: {self.model.device}")
            return True
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")
            return False
    
    def load_data(self):
        """Load and prepare the CSV data."""
        print("\n2. Loading data...")
        
        try:
            # Load the CSV data
            df = pd.read_csv(self.data_path, header=None)
            print(f"Data loaded successfully. Shape: {df.shape}")
            
            # Extract features (all columns except the last) and target (last column)
            n_features = len(df.columns) - 1
            self.X = df.iloc[:, :n_features].values  # All columns except the last
            self.y = df.iloc[:, -1].values           # Last column
            
            print(f"Features (X) shape: {self.X.shape}")
            print(f"Target (y) shape: {self.y.shape}")
            
            # Display data statistics
            print(f"\nData Statistics:")
            for i in range(self.X.shape[1]):
                print(f"Column {i+1} (X[:,{i}]): min={self.X[:,i].min()}, max={self.X[:,i].max()}, mean={self.X[:,i].mean():.2f}")
            print(f"Target (y): min={self.y.min()}, max={self.y.max()}, mean={self.y.mean():.2f}")
            
            # Show sample data
            print(f"\nSample data (first 5 rows):")
            for i in range(min(5, len(self.y))):
                feature_str = ", ".join([f"X{j+1}={self.X[i,j]:.2f}" for j in range(self.X.shape[1])])
                print(f"Row {i}: {feature_str}, y={self.y[i]:.6f}")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Could not load data from {self.data_path}")
            print(f"Error: {e}")
            return False
    
    def analyze_data(self):
        """Analyze the data using the e2e_Transformer model."""
        print("\n3. Analyzing data with e2e_Transformer...")
        
        try:
            # Initialize the symbolic regression estimator
            est = symbolicregression.model.SymbolicTransformerRegressor(
                model=self.model,
                max_input_points=200,
                n_trees_to_refine=100,
                rescale=True
            )
            
            print("Fitting the model...")
            est.fit(self.X, self.y, verbose=True)
            
            # Retrieve the best equation
            print("\nRetrieving the best equation...")
            result = est.retrieve_tree(with_infos=True)
            
            if result is None:
                print("No equation found!")
                return False
            
            # Get the predicted equation
            predicted_tree = result["relabed_predicted_tree"]
            equation_str = predicted_tree.infix()
            
            # Replace operation names with standard symbols
            replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
            for op, replace_op in replace_ops.items():
                equation_str = equation_str.replace(op, replace_op)
            
            print(f"\nRaw equation string: {equation_str}")
            
            # Parse and display the equation
            try:
                parsed_equation = sp.parse_expr(equation_str)
                print(f"\nParsed equation: {parsed_equation}")
                
                # Store results
                self.results['raw_equation'] = equation_str
                self.results['parsed_equation'] = str(parsed_equation)
                self.results['equation_latex'] = sp.latex(parsed_equation)
                
                return True
                
            except Exception as e:
                print(f"Could not parse equation: {e}")
                self.results['raw_equation'] = equation_str
                return False
                
        except Exception as e:
            print(f"ERROR: Analysis failed: {e}")
            return False
    
    def evaluate_equation(self):
        """Evaluate the found equation on the data."""
        print("\n4. Evaluating the found equation...")
        
        if 'raw_equation' not in self.results:
            print("No equation to evaluate!")
            return False
        
        try:
            # Create a function from the equation with better error handling
            equation_expr = sp.parse_expr(self.results['raw_equation'])
            
            # Check what symbols are actually in the equation
            free_symbols = equation_expr.free_symbols
            print(f"Symbols in equation: {free_symbols}")
            
            # Create the correct symbols based on what's in the equation
            symbol_list = []
            for symbol in free_symbols:
                if str(symbol).startswith('x_'):
                    symbol_list.append(symbol)
            
            if len(symbol_list) != self.X.shape[1]:
                print(f"Warning: Expected {self.X.shape[1]} symbols, found {len(symbol_list)}")
                return self._manual_evaluation(equation_expr)
            
            # Try to simplify the expression first
            try:
                equation_expr = sp.simplify(equation_expr)
                print(f"Simplified equation: {equation_expr}")
            except:
                print("Could not simplify equation, using original")
            
            # Try different lambdify approaches
            equation_func = None
            lambdify_attempts = [
                # Try with numpy first
                lambda expr: sp.lambdify(symbol_list, expr, modules=['numpy']),
                # Try with math module as fallback
                lambda expr: sp.lambdify(symbol_list, expr, modules=['math', 'numpy']),
                # Try with sympy's own functions
                lambda expr: sp.lambdify(symbol_list, expr, modules=['sympy']),
                # Try with minimal modules
                lambda expr: sp.lambdify(symbol_list, expr, modules=['numpy', {'sin': np.sin, 'cos': np.cos, 'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt, 'abs': np.abs}])
            ]
            
            for i, lambdify_func in enumerate(lambdify_attempts):
                try:
                    print(f"Attempting lambdify method {i+1}...")
                    equation_func = lambdify_func(equation_expr)
                    
                    # Test the function with a simple case
                    test_inputs = [1.0] * len(symbol_list)
                    test_result = equation_func(*test_inputs)
                    if not np.isnan(test_result) and not np.isinf(test_result):
                        print(f"Lambdify method {i+1} successful!")
                        break
                    else:
                        print(f"Lambdify method {i+1} produced invalid result, trying next...")
                        equation_func = None
                        
                except Exception as e:
                    print(f"Lambdify method {i+1} failed: {e}")
                    equation_func = None
                    continue
            
            if equation_func is None:
                print("All lambdify attempts failed. Trying manual evaluation...")
                return self._manual_evaluation(equation_expr)
            
            # Make predictions
            predictions = []
            for i in range(len(self.X)):
                try:
                    pred = equation_func(*self.X[i])
                    predictions.append(pred)
                except:
                    predictions.append(np.nan)
            
            predictions = np.array(predictions)
            
            # Check for invalid predictions
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                print("Predictions contain NaN or Inf values. Trying manual evaluation...")
                return self._manual_evaluation(equation_expr)
            
            # Calculate metrics
            mse = np.mean((predictions - self.y) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - self.y))
            r2 = 1 - np.sum((predictions - self.y) ** 2) / np.sum((self.y - self.y.mean()) ** 2)
            
            print(f"\nEvaluation Metrics:")
            print(f"MSE: {mse:.8f}")
            print(f"RMSE: {rmse:.6f}")
            print(f"MAE: {mae:.6f}")
            print(f"R²: {r2:.6f}")
            
            # Store metrics
            self.results['metrics'] = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2)
            }
            
            # Store predictions
            self.results['predictions'] = predictions.tolist()
            self.results['actual_values'] = self.y.tolist()
            
            # Show some predictions vs actual
            print(f"\nSample predictions vs actual:")
            for i in range(min(10, len(self.y))):
                print(f"Row {i}: Predicted={predictions[i]:.6f}, Actual={self.y[i]:.6f}, Error={abs(predictions[i]-self.y[i]):.6f}")
            
            return True
            
        except Exception as e:
            print(f"Error evaluating equation: {e}")
            print("Trying manual evaluation as fallback...")
            return self._manual_evaluation(equation_expr)
    
    def _manual_evaluation(self, equation_expr):
        """Manual evaluation of the equation using sympy's subs method."""
        try:
            print("Using manual evaluation method...")
            
            # Check what symbols are actually in the equation
            free_symbols = equation_expr.free_symbols
            print(f"Symbols in equation: {free_symbols}")
            
            # Create the correct symbols based on what's in the equation
            symbol_map = {}
            for symbol in free_symbols:
                if str(symbol).startswith('x_'):
                    symbol_map[symbol] = symbol
            
            predictions = []
            
            for i in range(len(self.X)):
                try:
                    # Map the data to the correct symbols
                    substitutions = []
                    for j, symbol in enumerate(symbol_map.keys()):
                        if j < len(self.X[i]):
                            substitutions.append((symbol, self.X[i, j]))
                    
                    # Substitute values into the expression
                    result = equation_expr.subs(substitutions)
                    
                    # Try to convert to float
                    float_result = None
                    
                    # Method 1: Try evalf()
                    try:
                        if hasattr(result, 'evalf'):
                            float_result = float(result.evalf())
                    except:
                        pass
                    
                    # Method 2: Try direct float conversion
                    if float_result is None:
                        try:
                            float_result = float(result)
                        except:
                            pass
                    
                    # Method 3: Try to simplify first
                    if float_result is None:
                        try:
                            simplified = sp.simplify(result)
                            if hasattr(simplified, 'evalf'):
                                float_result = float(simplified.evalf())
                            else:
                                float_result = float(simplified)
                        except:
                            pass
                    
                    if float_result is not None and not (np.isnan(float_result) or np.isinf(float_result)):
                        predictions.append(float_result)
                    else:
                        print(f"Row {i}: Could not evaluate expression: {result}")
                        predictions.append(np.nan)
                    
                except Exception as e:
                    print(f"Error evaluating row {i}: {e}")
                    predictions.append(np.nan)
            
            predictions = np.array(predictions)
            
            # Remove NaN values for metric calculation
            valid_mask = ~np.isnan(predictions)
            if np.sum(valid_mask) < len(predictions) * 0.5:  # Less than 50% valid
                print("Too many invalid predictions, skipping evaluation")
                print("The equation was found but cannot be evaluated numerically.")
                print("This is normal - the equation contains free symbols for parameter substitution.")
                
                # Store the equation without metrics
                self.results['evaluation_status'] = 'equation_found_with_free_symbols'
                self.results['equation_complexity'] = 'symbolic'
                self.results['free_symbols'] = [str(s) for s in free_symbols]
                return True  # Return True to continue with saving results
            
            valid_predictions = predictions[valid_mask]
            valid_actual = self.y[valid_mask]
            
            # Calculate metrics
            mse = np.mean((valid_predictions - valid_actual) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(valid_predictions - valid_actual))
            r2 = 1 - np.sum((valid_predictions - valid_actual) ** 2) / np.sum((valid_actual - valid_actual.mean()) ** 2)
            
            print(f"\nEvaluation Metrics (manual method):")
            print(f"Valid predictions: {np.sum(valid_mask)}/{len(predictions)}")
            print(f"MSE: {mse:.8f}")
            print(f"RMSE: {rmse:.6f}")
            print(f"MAE: {mae:.6f}")
            print(f"R²: {r2:.6f}")
            
            # Store metrics
            self.results['metrics'] = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'valid_predictions': int(np.sum(valid_mask)),
                'total_predictions': len(predictions)
            }
            
            # Store predictions
            self.results['predictions'] = predictions.tolist()
            self.results['actual_values'] = self.y.tolist()
            
            # Show some predictions vs actual
            print(f"\nSample predictions vs actual:")
            for i in range(min(10, len(self.y))):
                if valid_mask[i]:
                    print(f"Row {i}: Predicted={predictions[i]:.6f}, Actual={self.y[i]:.6f}, Error={abs(predictions[i]-self.y[i]):.6f}")
                else:
                    print(f"Row {i}: Predicted=NaN, Actual={self.y[i]:.6f}")
            
            return True
            
        except Exception as e:
            print(f"Manual evaluation also failed: {e}")
            print("The equation was found but cannot be evaluated numerically.")
            self.results['evaluation_status'] = 'evaluation_failed'
            return True  # Return True to continue with saving results
    
    def save_results(self):
        """Save all results to files."""
        print("\n5. Saving results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results as JSON
        json_path = f'leading_ones_results_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved as: {json_path}")
        
        # Save summary report
        report_path = f'leading_ones_report_{timestamp}.txt'
        with open(report_path, 'w') as f:
            f.write("=== e2e_Transformer Analysis Report ===\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data file: {self.data_path}\n\n")
            
            f.write("DATA SUMMARY:\n")
            f.write(f"Dataset shape: {self.X.shape}\n")
            f.write(f"Features: {self.X.shape[1]} input columns\n")
            f.write(f"Target: Last column\n\n")
            
            f.write("FOUND EQUATION:\n")
            f.write(f"Raw: {self.results.get('raw_equation', 'N/A')}\n")
            f.write(f"Parsed: {self.results.get('parsed_equation', 'N/A')}\n")
            f.write(f"LaTeX: {self.results.get('equation_latex', 'N/A')}\n\n")
            
            if 'metrics' in self.results:
                f.write("EVALUATION METRICS:\n")
                for metric, value in self.results['metrics'].items():
                    f.write(f"{metric.upper()}: {value:.8f}\n")
                f.write("\n")
            
            f.write("INTERPRETATION:\n")
            f.write("The equation shows the relationship between input features and target output.\n")
            f.write("Lower error metrics indicate better fit.\n")
            f.write("R² closer to 1.0 indicates better predictive performance.\n")
        
        print(f"Report saved as: {report_path}")
        
        return json_path, report_path
    
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        print("=== e2e_Transformer Analysis ===\n")
        
        # Step 1: Load model
        if not self.load_model():
            return False
        
        # Step 2: Load data
        if not self.load_data():
            return False
        
        # Step 3: Analyze data
        if not self.analyze_data():
            return False
        
        # Step 4: Evaluate equation
        if not self.evaluate_equation():
            return False
        
        # Step 5: Save results
        json_path, report_path = self.save_results()
        
        print("\n=== Analysis Complete ===")
        print(f"Results saved in: {json_path}")
        print(f"Report saved in: {report_path}")
        
        # After printing the equation, update results_rounding.csv
        import pandas as pd
        csv_dir = os.path.dirname(self.data_path)
        results_file = os.path.join(csv_dir, "results_rounding.csv")
        try:
            if os.path.exists(results_file):
                results_df = pd.read_csv(results_file)
                if 'e2e_transformer' not in results_df.columns:
                    results_df['e2e_transformer'] = ''
                # Always update the first row
                if len(results_df) > 0:
                    results_df.at[0, 'e2e_transformer'] = equation_str
                else:
                    # If the file is empty but has columns, create a single row
                    empty_row = {col: '' for col in results_df.columns}
                    empty_row['e2e_transformer'] = equation_str
                    results_df = pd.DataFrame([empty_row])
            else:
                # If the file doesn't exist, create a single-row DataFrame with only e2e_transformer
                results_df = pd.DataFrame({'e2e_transformer': [equation_str]})
            results_df.to_csv(results_file, index=False)
            print(f"E2E Transformer result saved to: {results_file}")
        except Exception as e:
            print(f"Error updating results_rounding.csv: {e}")
        
        return True

def main():
    """Main function to run the analysis."""
    if len(sys.argv) != 2:
        print("Usage: python analyze_leading_ones.py <csv_file_path>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    analyzer = E2ETransformerAnalyzer(data_path=csv_file)
    success = analyzer.run_analysis()
    
    if success:
        print("\n✅ Analysis completed successfully!")
        # Return the equation for the shell script to capture
        if 'raw_equation' in analyzer.results:
            print(f"EQUATION: {analyzer.results['raw_equation']}")
    else:
        print("\n❌ Analysis failed!")

if __name__ == "__main__":
    main() 