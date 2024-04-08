import pandas as pd

# Load the datasets
data_path1 = './data/final_model_dataset.csv'
data_path2 = 'final_model_dataset.csv'

df1 = pd.read_csv(data_path1)
df2 = pd.read_csv(data_path2)

# Optionally, sort both DataFrames if they contain any order-sensitive columns
# df1 = df1.sort_values(by=['some_column'])
# df2 = df2.sort_values(by=['some_column'])

# Check if both DataFrames are equal
are_equal = df1.equals(df2)

print(f"Are both datasets equal? {are_equal}")

if are_equal:
    print("Both datasets are the same. You can consider keeping one and deleting the other.")
else:
    print("The datasets differ. Consider keeping both and investigating the differences.")