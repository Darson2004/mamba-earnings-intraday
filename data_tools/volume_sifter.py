import pandas as pd

# Load the CSV file
file_path = "earnings_ohlcv_data_six.csv"  # Replace with your actual path if running locally
df = pd.read_csv(file_path)

# Convert DATE column to datetime
df['DATE'] = pd.to_datetime(df['DATE'])

# Sort and exclude the first and last 5 rows (by time) of each trading day for each stock
filtered_df = (
    df.sort_values(by=['NAME', 'DATE', 'TIME'])
    .groupby(['NAME', 'DATE'])
    .apply(lambda group: group.iloc[5:-5])  # Trim the first and last 5 rows
    .reset_index(drop=True)
)

# Sum volume for each NAME and DATE after trimming
adjusted_daily_volume = (
    filtered_df.groupby(['NAME', 'DATE'])['VOLUME']
    .sum()
    .reset_index()
    .sort_values(by=['NAME', 'DATE'])
    .reset_index(drop=True)
)

# Compare each trading day to its consecutive trading day
results = []
i = 0
while i < len(adjusted_daily_volume) - 1:
    current = adjusted_daily_volume.iloc[i]
    next_day = adjusted_daily_volume.iloc[i + 1]
    
    # Only compare if both entries are for the same stock and days are consecutive
    if current['NAME'] == next_day['NAME'] and (next_day['DATE'] - current['DATE']).days == 1:
        if current['VOLUME'] >= next_day['VOLUME']:
            results.append({
                'NAME': current['NAME'],
                'DATE': current['DATE'],
                'VOLUME': current['VOLUME'],
                'LABEL': 'BMO'
            })
        else:
            results.append({
                'NAME': next_day['NAME'],
                'DATE': next_day['DATE'],
                'VOLUME': next_day['VOLUME'],
                'LABEL': 'AMC'
            })
        i += 2  # Move to next pair
    else:
        i += 1  # Skip if not a valid consecutive pair

# Create final DataFrame and export
result_df = pd.DataFrame(results)
output_path = "volume_sifter.csv"
result_df.to_csv(output_path, index=False)
print(f"Saved trimmed high volume data with labels to: {output_path}")
