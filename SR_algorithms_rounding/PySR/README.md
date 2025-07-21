# PySR Symbolic Regression Integration

This directory contains the wrapper and scripts for running symbolic regression using [PySR](https://github.com/MilesCranmer/PySR) as part of the DAC_SR project.

## Environment Setup

A dedicated conda environment is created for PySR:

```
conda create -n pysr_env python=3.10 -y
conda activate pysr_env
pip install -r requirements.txt
```

## Usage

The main entry point is the shell script:

```
Scripts/pysr.sh <path_to_csv_file>
```

- The script activates the `pysr_env` environment, runs symbolic regression on the provided CSV file, and updates or creates a `results.csv` file in the same directory as the input CSV, adding a `pysr` column with the best discovered equation.
- The Python wrapper `run_pysr.py` handles the regression and result extraction.

## Input Format

- The input CSV should have features in all columns except the last, which is the target variable.
- The script assumes no header, or that the header is compatible with pandas' default behavior.

## Output

- The best symbolic equation found is printed to stdout and written to the `results.csv` file in the same directory as the input CSV.

## Requirements

- See `requirements.txt` for dependencies.

## Notes

- PySR will automatically install Julia and its dependencies on first use.
- For more advanced usage and tuning, see the [PySR documentation](https://github.com/MilesCranmer/PySR). 