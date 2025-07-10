# e2e_Transformer Analysis for GTLeadingOnes.csv

This directory contains scripts to analyze the GTLeadingOnes.csv dataset using the e2e_Transformer symbolic regression model.

## Files

- `analyze_leading_ones.py` - Main analysis script
- `run_analysis.sh` - Shell script to run the analysis
- `model1.pt` - Pre-trained e2e_Transformer model
- `README_ANALYSIS.md` - This file

## Prerequisites

1. **Conda Environment**: Make sure you have the `e2e_transformer` conda environment set up
2. **Data**: The script expects `GTLeadingOnes.csv` to be located at `../../DataSets/Ground_Truth/LeadingOnes/continuous/GTLeadingOnes.csv`

## Quick Start

### Option 1: Using the shell script (Recommended)
```bash
./run_analysis.sh
```

### Option 2: Manual execution
```bash
# Activate the conda environment
conda activate e2e_transformer

# Run the analysis
python analyze_leading_ones.py
```

## What the Script Does

1. **Loads the pre-trained model** from `model1.pt`
2. **Loads the data** from GTLeadingOnes.csv
3. **Analyzes the data** using e2e_Transformer to find the equation relating columns 1 and 2 to column 3
4. **Evaluates the equation** by calculating MSE, RMSE, MAE, and R² metrics
5. **Creates visualizations** including:
   - Predictions vs Actual values
   - Residual plot
   - Distribution comparison
   - Error distribution
6. **Saves results** in multiple formats:
   - JSON file with detailed results
   - Text report with summary
   - PNG file with visualizations

## Output Files

After running the analysis, you'll get:

- `leading_ones_results_YYYYMMDD_HHMMSS.json` - Detailed results in JSON format
- `leading_ones_report_YYYYMMDD_HHMMSS.txt` - Human-readable summary report
- `leading_ones_analysis_plots.png` - Visualization plots

## Data Format

The script expects GTLeadingOnes.csv to have 3 columns:
- **Column 1**: First feature (X1)
- **Column 2**: Second feature (X2) 
- **Column 3**: Target variable (y)

The script will find an equation of the form: `y = f(X1, X2)`

## Troubleshooting

### Common Issues

1. **Environment not found**: Make sure the `e2e_transformer` conda environment exists
2. **Model file missing**: Ensure `model1.pt` is in the current directory
3. **Data file not found**: Check that the path to GTLeadingOnes.csv is correct
4. **CUDA issues**: The script will automatically fall back to CPU if CUDA is not available

### Error Messages

- `ModuleNotFoundError`: Make sure all dependencies are installed in the conda environment
- `FileNotFoundError`: Check file paths for model and data files
- `RuntimeError`: Usually indicates CUDA/GPU issues - the script should handle this automatically

## Customization

You can modify the analysis by editing `analyze_leading_ones.py`:

- Change model parameters in the `SymbolicTransformerRegressor` initialization
- Modify visualization settings
- Add additional evaluation metrics
- Change output file formats

## Example Output

The script will output something like:

```
=== e2e_Transformer Analysis of GTLeadingOnes.csv ===

1. Loading pre-trained model...
Model loaded on GPU: cuda:0

2. Loading data...
Data loaded successfully. Shape: (32, 3)
Features (X) shape: (32, 2)
Target (y) shape: (32,)

3. Analyzing data with e2e_Transformer...
Fitting the model...
Retrieving the best equation...

Raw equation string: (x1 + x2) / (x1 * x2 + 1)
Parsed equation: (x1 + x2)/(x1*x2 + 1)

4. Evaluating the found equation...
Evaluation Metrics:
MSE: 0.000123
RMSE: 0.011090
MAE: 0.008765
R²: 0.999876

=== Analysis Complete ===
Results saved in: leading_ones_results_20241201_143022.json
Report saved in: leading_ones_report_20241201_143022.txt
```

## License

This analysis script is provided as-is for research purposes. 