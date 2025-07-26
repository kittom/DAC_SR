# Minimal TPSR configuration for LeadingOnes problem
# Ground truth: x1/(x2 + 1)
# Required functions: add, div

MINIMAL_OPERATORS = {
    "add": 2,  # Binary operator
    "div": 2,  # Binary operator
}

# Note: Removed all other operators that are not needed for LeadingOnes:
# - sub (subtraction)
# - mul (multiplication)
# - abs, inv, sqrt, log, exp, sin, cos, tan, arcsin, arccos, arctan
# - pow2, pow3 