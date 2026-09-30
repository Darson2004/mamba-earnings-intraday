import pandas as pd
from datetime import datetime, timedelta

# Load the CSV file
file_path = "earnings_ohlcv_data_six.csv"  # Replace with your actual path if running locally
df = pd.read_csv(file_path)

# Convert DATE column to datetime
df['DATE'] = pd.to_datetime(df['DATE'])

# Sort by NAME, DATE, and TIME
filtered_df = df.sort_values(by=['NAME', 'DATE', 'TIME']).reset_index(drop=True)

# Function to filter data after 30 minutes from initial close
def get_data_after_30min(group):
    if len(group) == 0:
        return group
    
    # Get the initial close time (first row of the day)
    initial_time = group.iloc[0]['TIME']
    
    # Convert initial time to datetime for comparison
    initial_datetime = pd.to_datetime(initial_time)
    
    # Calculate 30 minutes after initial time
    thirty_min_after = initial_datetime + timedelta(minutes=30)
    
    # Filter rows where time is 30 minutes or more after initial time
    filtered_group = group[pd.to_datetime(group['TIME']) >= thirty_min_after]
    
    return filtered_group

# Apply the 30-minute filter to each day's data
filtered_df = (
    filtered_df.groupby(['NAME', 'DATE'])
    .apply(get_data_after_30min)
    .reset_index(drop=True)
)

# Calculate close price range and initial close price for each NAME and DATE
daily_close_analysis = (
    filtered_df.groupby(['NAME', 'DATE'])
    .agg({
        'CLOSE': ['max', 'min', 'first']  # max, min, and first (initial) close price
    })
    .reset_index()
)

# Flatten column names
daily_close_analysis.columns = ['NAME', 'DATE', 'MAX_CLOSE', 'MIN_CLOSE', 'INITIAL_CLOSE']

# Calculate the absolute percentage change
daily_close_analysis['CLOSE_RANGE'] = daily_close_analysis['MAX_CLOSE'] - daily_close_analysis['MIN_CLOSE']
daily_close_analysis['PERCENTAGE_CHANGE'] = abs(daily_close_analysis['CLOSE_RANGE'] / daily_close_analysis['INITIAL_CLOSE'])

# Sort by NAME and DATE
daily_close_analysis = daily_close_analysis.sort_values(by=['NAME', 'DATE']).reset_index(drop=True)

# Compare each trading day to its consecutive trading day
results = []
i = 0
while i < len(daily_close_analysis) - 1:
    current = daily_close_analysis.iloc[i]
    next_day = daily_close_analysis.iloc[i + 1]
    
    # Only compare if both entries are for the same stock and days are consecutive
    if current['NAME'] == next_day['NAME'] and (next_day['DATE'] - current['DATE']).days == 1:
        if current['PERCENTAGE_CHANGE'] >= next_day['PERCENTAGE_CHANGE']:
            results.append({
                'NAME': current['NAME'],
                'DATE': current['DATE'],
                'DAY1_PERCENTAGE': current['PERCENTAGE_CHANGE'],
                'DAY2_PERCENTAGE': next_day['PERCENTAGE_CHANGE'],
                'LABEL': 'BMO'
            })
        else:
            results.append({
                'NAME': next_day['NAME'],
                'DATE': next_day['DATE'],
                'DAY1_PERCENTAGE': current['PERCENTAGE_CHANGE'],
                'DAY2_PERCENTAGE': next_day['PERCENTAGE_CHANGE'],
                'LABEL': 'AMC'
            })
        i += 2  # Move to next pair
    else:
        i += 1  # Skip if not a valid consecutive pair

# Create final DataFrame and export
result_df = pd.DataFrame(results)
output_path = "sifted_percentage_change.csv"
result_df.to_csv(output_path, index=False)
print(f"Saved percentage change analysis with labels to: {output_path}")
