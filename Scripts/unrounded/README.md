# Unrounded Scripts Organization

This directory contains symbolic regression scripts organized into two subdirectories:

## control_library/
Contains scripts that run algorithms with the full mathematical function library:
- `deepsr.sh` - DeepSR with full library
- `pysr.sh` - PySR with full library  
- `kan.sh` - KAN with full library
- `tpsr.sh` - TPSR with full library
- `linear.sh` - Linear Regression
- `e2e_transformer.sh` - E2E Transformer
- `qlattice.sh` - Q-Lattice

## algorithm_library/
Contains scripts that run algorithms with minimal function libraries:
- `deepsr_lib.sh` - DeepSR with minimal library
- `pysr_lib.sh` - PySR with minimal library
- `kan_lib.sh` - KAN with minimal library
- `tpsr_lib.sh` - TPSR with minimal library
- `linear_lib.sh` - Linear Regression (same as control)

## Usage

### Control Experiments (Full Library)
```bash
# Run all algorithms with full library
./Scripts/run_all_sr.sh path/to/data.csv

# Run individual algorithm with full library
./Scripts/unrounded/control_library/pysr.sh path/to/data.csv
```

### Minimal Library Experiments
```bash
# Run all algorithms with minimal library
./Scripts/run_all_library.sh one_max path/to/data.csv

# Run individual algorithm with minimal library
./Scripts/unrounded/algorithm_library/pysr_lib.sh path/to/data.csv one_max
```

## Results Files
- `results.csv` - Results from control experiments (full library)
- `results_lib.csv` - Results from minimal library experiments 