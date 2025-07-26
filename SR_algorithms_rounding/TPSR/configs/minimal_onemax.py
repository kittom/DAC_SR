# Minimal TPSR configuration for OneMax problem
# Ground truth: sqrt(x1/(x1-x2))
# Required functions: add, sub, div, sqrt

MINIMAL_OPERATORS = {
    "add": 2,  # Binary operator
    "sub": 2,  # Binary operator  
    "div": 2,  # Binary operator
    "sqrt": 1, # Unary operator
}

# Note: Removed all other operators that are not needed for OneMax:
# - mul (multiplication)
# - abs, inv, log, exp, sin, cos, tan, arcsin, arccos, arctan
# - pow2, pow3 