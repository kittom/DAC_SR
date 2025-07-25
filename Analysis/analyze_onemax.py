import re
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sklearn.metrics import r2_score, mean_squared_error
import argparse

ORDER = ["ground_truth", "linear", "deepsr", "pysr", "kan", "TPSR", "E2ET", "Q_lattice"]

def safe_eval(expr, variables, data, length):
    try:
        f = sp.lambdify(variables, expr, modules=["numpy"])
        result = f(*[data[var] for var in variables])
        result = np.array(result)
        if result.shape == ():
            result = np.full(length, result)
        elif result.shape == (length,):
            pass
        else:
            result = np.full(length, np.nan)
        if isinstance(result, np.ndarray) and result.dtype == object:
            try:
                result = result.astype(float)
            except Exception as e:
                print(f"  [ERROR] Could not convert result to float. First few values: {result[:5]}")
                print(f"  [ERROR] Exception: {e}")
                result = np.full(length, np.nan)
        return result
    except Exception as e:
        print(f"  [ERROR] Exception in safe_eval: {e}")
        return np.full(length, np.nan)

def increment_pysr_vars(eq):
    def repl(match):
        idx = int(match.group(1))
        return f"x{idx+1}"
    return re.sub(r"x(\d+)", repl, eq)

def get_modifier(results_csv):
    if "lib" in results_csv:
        return "library"
    elif "rounding" in results_csv:
        return "rounded"
    else:
        return "control"

def main():
    parser = argparse.ArgumentParser(description="Analyze OneMax results and generate plots/summary.")
    parser.add_argument("results_csv", help="Path to results CSV (e.g., results_lib.csv)")
    args = parser.parse_args()

    results_csv = args.results_csv
    modifier = get_modifier(results_csv)
    outdir = os.path.join(os.path.dirname(results_csv), f"analysis_{modifier}")
    os.makedirs(outdir, exist_ok=True)

    results = pd.read_csv(results_csv, header=0)
    eq_row = results.iloc[1] if len(results) > 1 else results.iloc[0]
    eqs = {col: eq_row[col] for col in results.columns if pd.notnull(eq_row[col])}
    algos = [col for col in ORDER if col in eqs]

    # Generate synthetic data: x1=2000, x2=0..2000
    x1_val = 2000
    x2_vals = np.arange(0, 2001)
    synth_data = pd.DataFrame({"x1": x1_val, "x2": x2_vals})
    synth_data["x_1"] = synth_data["x1"]
    synth_data["x_2"] = synth_data["x2"]
    synth_data["x0"] = synth_data["x1"]
    length = len(synth_data)

    # Compute ground truth y using the ground truth equation
    gt_eq = eqs.get("ground_truth", None)
    if gt_eq is not None:
        try:
            gt_expr = parse_expr(gt_eq, evaluate=False)
            gt_vars = sorted([str(s) for s in gt_expr.free_symbols])
            # Robustly handle invalid sqrt (e.g., negative values)
            with np.errstate(invalid='ignore', divide='ignore'):  # Ignore warnings for sqrt of negative
                synth_data["y"] = safe_eval(gt_expr, gt_vars, synth_data, length)
                synth_data["y"] = np.where(np.isfinite(synth_data["y"]), synth_data["y"], np.nan)
        except Exception:
            synth_data["y"] = np.nan
    else:
        synth_data["y"] = np.nan

    x = synth_data["x1"].values
    y_true = synth_data["y"].values

    # Bar/line plot for all algorithms
    plt.figure(figsize=(10, 6))
    for algo in algos:
        eq = eqs[algo]
        if algo == "pysr":
            eq = increment_pysr_vars(eq)
        try:
            expr = parse_expr(eq, evaluate=False)
            variables = sorted([str(s) for s in expr.free_symbols])
            print(f"\n[DEBUG] Plotting {algo}")
            print(f"  Equation: {eq}")
            print(f"  Variables in equation: {variables}")
            missing = [v for v in variables if v not in synth_data.columns]
            if missing:
                print(f"  [WARNING] Missing variables in data: {missing}")
                y_pred = np.full(length, np.nan)
            else:
                y_pred = safe_eval(expr, variables, synth_data, length)
                print(f"  Result type: {type(y_pred)}, shape: {getattr(y_pred, 'shape', None)}")
                if not isinstance(y_pred, np.ndarray) or y_pred.dtype == object:
                    print(f"  [WARNING] Evaluation did not return a numeric array!")
                    y_pred = np.full(length, np.nan)
        except Exception as e:
            print(f"  [ERROR] Exception while evaluating {algo}: {e}")
            y_pred = np.full(length, np.nan)
        plt.plot(synth_data["x2"].values, y_pred, label=algo)
    plt.xlabel("x2 (current state)")
    plt.ylabel("y (bitflip)")
    plt.title(f"OneMax: x1={x1_val}, Algorithm Predictions")
    plt.legend()
    plot_path = os.path.join(outdir, f"onemax_x1_{x1_val}.png")
    plt.savefig(plot_path)
    plt.close()

    # Logarithmic y-axis plot
    plt.figure(figsize=(10, 6))
    for algo in algos:
        eq = eqs[algo]
        if algo == "pysr":
            eq = increment_pysr_vars(eq)
        try:
            expr = parse_expr(eq, evaluate=False)
            variables = sorted([str(s) for s in expr.free_symbols])
            missing = [v for v in variables if v not in synth_data.columns]
            if missing:
                y_pred = np.full(length, np.nan)
            else:
                y_pred = safe_eval(expr, variables, synth_data, length)
                if not isinstance(y_pred, np.ndarray) or y_pred.dtype == object:
                    y_pred = np.full(length, np.nan)
        except Exception:
            y_pred = np.full(length, np.nan)
        plt.plot(synth_data["x2"].values, y_pred, label=algo)
    plt.xlabel("x2 (current state)")
    plt.ylabel("y (bitflip)")
    plt.title(f"OneMax: x1={x1_val}, Algorithm Predictions (Log Y)")
    plt.yscale('log')
    plt.legend()
    plot_path_log = os.path.join(outdir, f"onemax_x1_{x1_val}_log.png")
    plt.savefig(plot_path_log)
    plt.close()

    # For summary CSV: evaluate all algorithms on the full synthetic data
    summary = {"name": [], "equation": [], "R^2": [], "mse": [], "nmse": [], "complexity": []}
    gt_y = synth_data["y"].values
    valid_domain = synth_data["x2"].values < synth_data["x1"].values  # Only x2 < x1
    for algo in algos:
        eq = eqs[algo]
        if algo == "pysr":
            eq = increment_pysr_vars(eq)
        summary["name"].append(algo)
        summary["equation"].append(eq)
        if algo == "ground_truth":
            summary["R^2"].append(0.0)
            summary["mse"].append(0.0)
            summary["nmse"].append(0.0)
        else:
            try:
                expr = parse_expr(eq, evaluate=False)
                variables = sorted([str(s) for s in expr.free_symbols])
                print(f"\n[DEBUG] Metrics for {algo}")
                print(f"  Equation: {eq}")
                print(f"  Variables in equation: {variables}")
                missing = [v for v in variables if v not in synth_data.columns]
                if missing:
                    print(f"  [WARNING] Missing variables in data: {missing}")
                    y_pred = np.full(length, np.nan)
                else:
                    with np.errstate(invalid='ignore', divide='ignore'):
                        y_pred = safe_eval(expr, variables, synth_data, length)
                        y_pred = np.where(np.isfinite(y_pred), y_pred, np.nan)
                    print(f"  gt_y[:5]: {gt_y[:5]}")
                    print(f"  y_pred[:5]: {y_pred[:5]}")
                    print(f"  Result type: {type(y_pred)}, shape: {getattr(y_pred, 'shape', None)}")
                    if not isinstance(y_pred, np.ndarray) or y_pred.dtype == object:
                        print(f"  [WARNING] Evaluation did not return a numeric array!")
                        y_pred = np.full(length, np.nan)
                # Restrict to valid domain: x2 < x1 and both gt_y and y_pred are finite
                mask = valid_domain & np.isfinite(gt_y) & np.isfinite(y_pred)
                if np.sum(mask) == 0:
                    print(f"  [WARNING] No valid points for metric calculation for {algo} (x2 < x1 and finite values)")
                    r2 = mse = nmse = np.nan
                else:
                    r2 = float(r2_score(gt_y[mask], y_pred[mask]))
                    mse = float(mean_squared_error(gt_y[mask], y_pred[mask]))
                    nmse = float(mse / np.var(gt_y[mask])) if np.var(gt_y[mask]) > 0 else np.nan
                print(f"  R^2: {r2}, MSE: {mse}, NMSE: {nmse}")
            except Exception as e:
                print(f"  [ERROR] Exception while evaluating {algo}: {e}")
                r2 = mse = nmse = np.nan
            summary["R^2"].append(r2)
            summary["mse"].append(mse)
            summary["nmse"].append(nmse)
        try:
            expr = parse_expr(eq, evaluate=False)
            complexity = len(list(sp.preorder_traversal(expr)))
        except Exception:
            complexity = len(str(eq))
        summary["complexity"].append(complexity)

    summary_df = pd.DataFrame(summary)
    summary_csv_path = os.path.join(outdir, "onemax_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    # Bar charts for each metric
    metrics = ["R^2", "mse", "nmse", "complexity"]
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        if metric in ["R^2", "mse", "nmse"]:
            plot_df = summary_df[summary_df["name"] != "ground_truth"]
        else:
            plot_df = summary_df
        plt.bar(plot_df["name"], plot_df[metric].astype(float))
        plt.xlabel("Algorithm")
        plt.ylabel(metric)
        plt.title(f"OneMax: {metric} by Algorithm")
        plt.xticks(rotation=30)
        plt.tight_layout()
        if metric == "R^2":
            plt.axhline(0, color='black', linestyle='--', linewidth=1)
        bar_path = os.path.join(outdir, f"onemax_{metric.replace('^','2').replace(' ','_').lower()}_bar.png")
        plt.savefig(bar_path)
        plt.close()
        print(f"Saved bar chart: {bar_path}")

    print(f"Saved plot: {plot_path}")
    print(f"Saved summary CSV: {summary_csv_path}")
    print(f"Algorithms analyzed: {algos}")

if __name__ == "__main__":
    main() 