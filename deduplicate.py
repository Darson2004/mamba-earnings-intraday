import pandas as pd

# Load the large CSV file
input_file = "rty_data.csv"       # ← Replace with your file path
output_file = "rty_data_deduplicated.csv"   # Output file

# Load CSV in chunks (optional for very large files)
df = pd.read_csv(input_file)

# Deduplicate based on ('ts_code', 'date') combination — keep first
df_dedup = df.drop_duplicates(subset=['ts_code', 'date', 'trade_time'], keep='first')

# Save to new CSV
df_dedup.to_csv(output_file, index=False)

print(f"Original rows: {len(df)}, Deduplicated rows: {len(df_dedup)}")
