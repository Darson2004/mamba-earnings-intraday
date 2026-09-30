import pandas as pd
import pytz
from datetime import datetime

# Load the CSV file
file_path = "earnings_ohlcv_data_four.csv"  # Replace with actual file path
df = pd.read_csv(file_path)

# Convert DATE column to datetime
df['DATE'] = pd.to_datetime(df['DATE'])

# Combine DATE and TIME into one datetime column (assume naive UTC first)
df['DATETIME'] = pd.to_datetime(df['DATE'].astype(str) + ' ' + df['TIME'])

# Convert from UTC to US/Eastern with DST awareness
df['DATETIME_EST'] = df['DATETIME'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
df['TIME_EST'] = df['DATETIME_EST'].dt.strftime('%H:%M:%S')
df['DATE_EST'] = df['DATETIME_EST'].dt.date

# Sort data for processing
df = df.sort_values(by=['NAME', 'DATE_EST', 'DATETIME_EST'])

# Function to compute % changes per group
def compute_changes(group):
    group = group.sort_values('DATETIME_EST').reset_index(drop=True)

    # Find row closest to 9:30:00 AM EST (first close price)
    group['ABS_DIFF_930AM'] = (pd.to_datetime(group['TIME_EST'], format='%H:%M:%S') -
                               pd.to_datetime('09:30:00', format='%H:%M:%S')).abs()
    row_930am = group.loc[group['ABS_DIFF_930AM'].idxmin()]
    open_close = row_930am['CLOSE']

    # Find row closest to 4:00:00 PM EST (last close price)
    group['ABS_DIFF_400PM'] = (pd.to_datetime(group['TIME_EST'], format='%H:%M:%S') -
                               pd.to_datetime('16:00:00', format='%H:%M:%S')).abs()
    row_400pm = group.loc[group['ABS_DIFF_400PM'].idxmin()]
    close_last = row_400pm['CLOSE']

    # Find row closest to 10:00:00 EST
    group['ABS_DIFF_10AM'] = (pd.to_datetime(group['TIME_EST'], format='%H:%M:%S') -
                              pd.to_datetime('10:00:00', format='%H:%M:%S')).abs()
    row_10am = group.loc[group['ABS_DIFF_10AM'].idxmin()]
    close_10am = row_10am['CLOSE']

    # Calculate percentage changes
    change_30m = (close_10am - open_close) / open_close * 100
    net_change = (close_last - open_close) / open_close * 100

    return pd.Series({
        '30M % CHANGE': change_30m,
        'NET % CHANGE': net_change
    })

# Group by NAME and DATE_EST and apply the change calculation
change_df = df.groupby(['NAME', 'DATE_EST']).apply(compute_changes).reset_index()

# Save result to CSV
output_path = "percentage_changes_four.csv"
change_df.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")
