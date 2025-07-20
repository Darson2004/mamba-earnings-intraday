import pandas as pd

# Read the CSV file
df = pd.read_csv('601988.SH.csv')

# Print column names
print("Column names:")
print(df.columns.tolist())

# Print first few rows
print("\nFirst 3 rows:")
print(df.head(3))

# Try to access the trade_date column
try:
    print("\nTrade date column:")
    print(df['trade_date'].head())
except KeyError as e:
    print(f"Error accessing trade_date: {e}")
    
# Check for any whitespace in column names
print("\nColumn names with repr:")
for col in df.columns:
    print(f"'{col}' -> {repr(col)}") 