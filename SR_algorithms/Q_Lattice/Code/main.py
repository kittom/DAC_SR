import feyn
import pandas as pd
from feyn.tools import split

# Load GTLeadingOnes dataset into a pandas dataframe
df_leading_ones = pd.read_csv('../../../DataSets/Ground_Truth/GTLeadingOnes.csv')

# Train/test split
train, test = split(df_leading_ones, ratio=[0.6, 0.4])

# Instantiate a QLattice
ql = feyn.QLattice()

models = ql.auto_run(
    data=train,
    output_name='Bitflip'
)
# Select the best Model
best = models[0]

best.plot_regression(data=train, filename="Bitflip-plot")
sympy_model = best.sympify(signif=3)
print(sympy_model.as_expr())