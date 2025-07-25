# Symbolic Regression Results Analysis Script
# (Refactored from evaluation to analysis)

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sympy as sp
import traceback
from typing import List, Dict, Any

# ... existing imports ...

class ResultsAnalyzer:
    """
    Main class for analyzing symbolic regression results.
    """
    def __init__(self, results_file: str):
        self.results_file = results_file
        self.results_dir = os.path.dirname(results_file)
        self.data = None
        self.ground_truth = None
        self.algorithms = []
        self.analysis_outputs = {}
        # ... other initialization ...

    def load_data(self):
        # ...
        pass

    def analyze_equation(self, equation: str, data: pd.DataFrame) -> Dict[str, Any]:
        # ...
        pass

    def run_analysis(self):
        """Run the complete analysis pipeline."""
        print("=== Symbolic Regression Results Analysis ===\n")
        # ...
        print(f"\n=== Analysis Complete ===")
        return True

# ... rest of the code, refactoring all 'evaluation' to 'analysis' ...

def main():
    parser = argparse.ArgumentParser(description="Analyze symbolic regression results.")
    parser.add_argument("results_file", help="Path to the results CSV file.")
    args = parser.parse_args()
    analyzer = ResultsAnalyzer(args.results_file)
    try:
        success = analyzer.run_analysis()
        if success:
            print("\n✅ Analysis completed successfully!")
        else:
            print("\n❌ Analysis failed!")
    except Exception as e:
        print("\n❌ Analysis failed with exception:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 