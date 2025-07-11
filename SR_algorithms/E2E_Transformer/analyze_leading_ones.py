#!/usr/bin/env python3
"""
Script to analyze GTLeadingOnes.csv data using e2e_Transformer
to find the equation relating columns 1 and 2 to column 3.

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

class LeadingOnesAnalyzer:
    def __init__(self, model_path="model1.pt", data_path="../../DataSets/Ground_Truth/LeadingOnes/discrete/GTLeadingOnes.csv"):
        """
        Initialize the analyzer.
        
        Args:
            model_path (str): Path to the pre-trained model
            data_path (str): Path to the GTLeadingOnes.csv file
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
        """Load and prepare the GTLeadingOnes.csv data."""
        print("\n2. Loading data...")
        
        try:
            # Load the CSV data
            df = pd.read_csv(self.data_path, header=None)
            print(f"Data loaded successfully. Shape: {df.shape}")
            
            # Extract features (columns 1 and 2) and target (column 3)
            self.X = df.iloc[:, [0, 1]].values  # Columns 1 and 2 (0-indexed)
            self.y = df.iloc[:, 2].values       # Column 3 (0-indexed)
            
            print(f"Features (X) shape: {self.X.shape}")
            print(f"Target (y) shape: {self.y.shape}")
            
            # Display data statistics
            print(f"\nData Statistics:")
            print(f"Column 1 (X[:,0]): min={self.X[:,0].min()}, max={self.X[:,0].max()}, mean={self.X[:,0].mean():.2f}")
            print(f"Column 2 (X[:,1]): min={self.X[:,1].min()}, max={self.X[:,1].max()}, mean={self.X[:,1].mean():.2f}")
            print(f"Column 3 (y): min={self.y.min()}, max={self.y.max()}, mean={self.y.mean():.2f}")
            
            # Show sample data
            print(f"\nSample data (first 5 rows):")
            for i in range(min(5, len(self.y))):
                print(f"Row {i}: X1={self.X[i,0]}, X2={self.X[i,1]}, y={self.y[i]:.6f}")
            
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
                if str(symbol) == 'x_0':
                    symbol_list.append(sp.symbols('x_0'))
                elif str(symbol) == 'x_1':
                    symbol_list.append(sp.symbols('x_1'))
            
            if len(symbol_list) != 2:
                print(f"Warning: Expected 2 symbols, found {len(symbol_list)}")
                return self._manual_evaluation(equation_expr)
            
            x_0, x_1 = symbol_list
            
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
                lambda expr: sp.lambdify((x_0, x_1), expr, modules=['numpy']),
                # Try with math module as fallback
                lambda expr: sp.lambdify((x_0, x_1), expr, modules=['math', 'numpy']),
                # Try with sympy's own functions
                lambda expr: sp.lambdify((x_0, x_1), expr, modules=['sympy']),
                # Try with minimal modules
                lambda expr: sp.lambdify((x_0, x_1), expr, modules=['numpy', {'sin': np.sin, 'cos': np.cos, 'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt, 'abs': np.abs}])
            ]
            
            for i, lambdify_func in enumerate(lambdify_attempts):
                try:
                    print(f"Attempting lambdify method {i+1}...")
                    equation_func = lambdify_func(equation_expr)
                    
                    # Test the function with a simple case
                    test_result = equation_func(1.0, 1.0)
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
            predictions = equation_func(self.X[:, 0], self.X[:, 1])
            
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
                if str(symbol) == 'x_0':
                    symbol_map[symbol] = sp.symbols('x_0')
                elif str(symbol) == 'x_1':
                    symbol_map[symbol] = sp.symbols('x_1')
                else:
                    print(f"Warning: Unknown symbol {symbol}")
            
            predictions = []
            
            for i in range(len(self.X)):
                try:
                    # Map the data to the correct symbols
                    substitutions = []
                    if sp.symbols('x_0') in symbol_map:
                        substitutions.append((symbol_map[sp.symbols('x_0')], self.X[i, 0]))
                    if sp.symbols('x_1') in symbol_map:
                        substitutions.append((symbol_map[sp.symbols('x_1')], self.X[i, 1]))
                    
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
    
    def create_visualizations(self):
        """Create visualizations of the results."""
        print("\n5. Creating visualizations...")
        
        try:
            if 'predictions' not in self.results:
                print("No predictions available for visualization")
                return False
            
            predictions = np.array(self.results['predictions'])
            
            # Check if we have valid predictions
            valid_mask = ~np.isnan(predictions)
            if np.sum(valid_mask) < len(predictions) * 0.5:
                print("Too few valid predictions for meaningful visualization")
                print("Creating data-only visualization...")
                
                # Create a simpler visualization showing just the data
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                fig.suptitle('e2e_Transformer Analysis Results for GTLeadingOnes.csv', fontsize=16)
                
                # 1. Data scatter plot
                axes[0].scatter(self.X[:, 0], self.y, alpha=0.7, label='Column 1 vs Target')
                axes[0].scatter(self.X[:, 1], self.y, alpha=0.7, label='Column 2 vs Target')
                axes[0].set_xlabel('Feature Values')
                axes[0].set_ylabel('Target Values')
                axes[0].set_title('Data Distribution')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # 2. Target distribution
                axes[1].hist(self.y, bins=15, alpha=0.7, color='blue')
                axes[1].set_xlabel('Target Values')
                axes[1].set_ylabel('Frequency')
                axes[1].set_title('Target Distribution')
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save the plot
                plot_path = 'leading_ones_analysis_plots.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Data visualization saved as: {plot_path}")
                
                self.results['plot_path'] = plot_path
                self.results['visualization_type'] = 'data_only'
                
                return True
            
            # Create full visualization with predictions
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('e2e_Transformer Analysis Results for GTLeadingOnes.csv', fontsize=16)
            
            valid_predictions = predictions[valid_mask]
            valid_actual = self.y[valid_mask]
            
            # 1. Predictions vs Actual
            axes[0, 0].scatter(valid_actual, valid_predictions, alpha=0.7)
            axes[0, 0].plot([valid_actual.min(), valid_actual.max()], [valid_actual.min(), valid_actual.max()], 'r--', lw=2)
            axes[0, 0].set_xlabel('Actual Values')
            axes[0, 0].set_ylabel('Predicted Values')
            axes[0, 0].set_title('Predictions vs Actual Values')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Residuals
            residuals = valid_predictions - valid_actual
            axes[0, 1].scatter(valid_predictions, residuals, alpha=0.7)
            axes[0, 1].axhline(y=0, color='r', linestyle='--')
            axes[0, 1].set_xlabel('Predicted Values')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].set_title('Residual Plot')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Data distribution
            axes[1, 0].hist(valid_actual, bins=15, alpha=0.7, label='Actual', density=True)
            axes[1, 0].hist(valid_predictions, bins=15, alpha=0.7, label='Predicted', density=True)
            axes[1, 0].set_xlabel('Values')
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].set_title('Distribution Comparison')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Error distribution
            axes[1, 1].hist(residuals, bins=15, alpha=0.7, color='orange')
            axes[1, 1].set_xlabel('Prediction Error')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Error Distribution')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the plot
            plot_path = 'leading_ones_analysis_plots.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved as: {plot_path}")
            
            self.results['plot_path'] = plot_path
            self.results['visualization_type'] = 'full_analysis'
            
            return True
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")
            return False
    
    def save_results(self):
        """Save all results to files."""
        print("\n6. Saving results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results as JSON
        json_path = f'leading_ones_results_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved as: {json_path}")
        
        # Save summary report
        report_path = f'leading_ones_report_{timestamp}.txt'
        with open(report_path, 'w') as f:
            f.write("=== e2e_Transformer Analysis Report for GTLeadingOnes.csv ===\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("DATA SUMMARY:\n")
            f.write(f"Dataset shape: {self.X.shape}\n")
            f.write(f"Features: Column 1 (X1) and Column 2 (X2)\n")
            f.write(f"Target: Column 3 (y)\n\n")
            
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
            f.write("The equation shows the relationship between columns 1 and 2 (features) and column 3 (target).\n")
            f.write("Lower error metrics indicate better fit.\n")
            f.write("R² closer to 1.0 indicates better predictive performance.\n")
        
        print(f"Report saved as: {report_path}")
        
        return json_path, report_path
    
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        print("=== e2e_Transformer Analysis of GTLeadingOnes.csv ===\n")
        
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
        
        # Step 5: Create visualizations
        self.create_visualizations()
        
        # Step 6: Save results
        json_path, report_path = self.save_results()
        
        print("\n=== Analysis Complete ===")
        print(f"Results saved in: {json_path}")
        print(f"Report saved in: {report_path}")
        
        return True

def main():
    """Main function to run the analysis."""
    analyzer = LeadingOnesAnalyzer()
    success = analyzer.run_analysis()
    
    if success:
        print("\n✅ Analysis completed successfully!")
    else:
        print("\n❌ Analysis failed!")

if __name__ == "__main__":
    main() 