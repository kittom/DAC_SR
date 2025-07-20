import pandas as pd
import numpy as np

# Create a small dataset with 50 points
x = np.linspace(0, 5, 50)
y = 2*x**2 + 3*x + 1 + 0.1*np.random.normal(0, 1, 50)

df = pd.DataFrame({'x': x, 'y': y})
df.to_csv('test_small.csv', index=False)
print('Created test_small.csv with 50 data points')
print(f'Dataset shape: {df.shape}')
print('First few rows:')
print(df.head()) 