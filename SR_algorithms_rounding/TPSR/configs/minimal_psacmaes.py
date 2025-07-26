# Minimal TPSR configuration for PSA-CMA-ES problem
# Ground truth: x1 * exp(x2 * (x5 - (x3 / x4)))
# Required functions: mul, sub, div, exp

MINIMAL_OPERATORS = {
    "mul": 2,  # Binary operator
    "sub": 2,  # Binary operator
    "div": 2,  # Binary operator
    "exp": 1,  # Unary operator
}

# Note: Removed all other operators that are not needed for PSA-CMA-ES:
# - add (addition)
# - abs, inv, sqrt, log, sin, cos, tan, arcsin, arccos, arctan
# - pow2, pow3 