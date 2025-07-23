import pandas as pd
import argparse 
import os
import h5py 
import random
import numpy as np

def convert_xlsx_to_csv(input_file):
    """
    Get the close prices for every minute on the latest day from the xlsx file. 
    Dump the data into a csv file. 
    """
    df = pd.read_excel(input_file)
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Input file {input_file} did not load as a DataFrame.")
    if 'Date Only' in df.columns and 'Time Only' in df.columns:
        # xlsx format: Date Only, Time Only, Close, ...
        df = df[df['Date Only'] == df['Date Only'].max()]
        # Use Series directly
        df = pd.DataFrame({'Time': df['Time Only'], 'Close': df['Close']})
    elif 'Date' in df.columns: 
        # xlsx format: Date, Close, ...
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        df['Date Only'] = pd.Series(df['Date']).dt.date
        df = df[df['Date Only'] == df['Date Only'].max()]
        # .dt.time only works on Series, not DatetimeIndex
        df = df.copy()
        df['Time'] = pd.Series(df['Date']).dt.time
        df = pd.DataFrame({'Time': df['Time'], 'Close': df['Close']})
    else:
        raise ValueError(f"Input file {input_file} does not have a valid format")
    # Sort by time
    df = df.sort_values(by='Time').reset_index(drop=True)
    # Create complete time series from 09:30 to 16:00 using pandas date_range
    start_datetime = pd.Timestamp('2024-01-01 09:30:00')
    end_datetime = pd.Timestamp('2024-01-01 15:59:00')
    complete_times = pd.date_range(start=start_datetime, end=end_datetime, freq='1min')
    # Convert to time objects
    complete_times = [t.time() for t in complete_times]
    # Create complete dataframe with all times
    complete_df = pd.DataFrame({'Time': complete_times})
    # Merge with original data, keeping all times
    df = pd.merge(complete_df, df, on='Time', how='left')
    nan_count = df['Close'].isna().sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values in {input_file}. Filling with previous values.")
    # Fill missing close prices with previous values (forward fill, then backward fill)
    df['Close'] = df['Close'].ffill().bfill()
    # Create output filename
    base_name = input_file.split("/")[-1].split(".")[0]
    output_file = f"data/{base_name}.csv"
    # Dump the data into a csv file
    df.to_csv(output_file, index=False)
    return output_file

# if __name__ == '__main__':
#     argparser = argparse.ArgumentParser()
#     argparser.add_argument('--input_files', nargs='+', default=None, help='List of input files to convert')
#     args = argparser.parse_args()

#     if args.input_files is None:
#         # Get all xlsx files in raw_data/
#         args.input_files = [f for f in os.listdir("raw_data/") if f.endswith(".xlsx")]
    
#     for name in args.input_files:
#         input_file = "raw_data/" + name
#         try: 
#             print(f"Saved data to {convert_xlsx_to_csv(input_file)}")
#         except Exception as e:
#             print(f"Error processing {input_file}: {e}")

#     print(f"Saved {len(args.input_files)} files to data/")

def convert_csv_to_hdf5(input_file, output_file, overwrite=False):
    """
    Args: 
        input_file: str, the path to the csv file
        output_file: str, the path to the hdf5 file
        overwrite: bool, whether to overwrite existing datasets
    """
    df = pd.read_csv(input_file)
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Input file {input_file} did not load as a DataFrame.")
    # Group by ts_code and date
    grouped = df.groupby(['ts_code', 'date'])
    print(f"Grouped object type: {type(grouped)}")
    print(f"Grouped keys: {list(grouped.groups.keys())[:3]}")  # Show first 3 keys
    # Define the complete time range for a trading day
    start_time = pd.to_datetime('09:30:00').time()
    end_time = pd.to_datetime('15:59:00').time()
    all_times = [t.time() for t in pd.date_range('2024-01-01 09:30:00', '2024-01-01 15:59:00', freq='1min')]
    # Open HDF5 file for appending
    with h5py.File(output_file, 'a') as h5f:
        for key in grouped.groups.keys():
            ts_code, date = key  # type: ignore
            group = grouped.get_group(key)
            group = pd.DataFrame(group).copy()
            # Dataset name: ts_code-date
            dataset_name = f"{ts_code}-{date}"
            # Handle overwrite logic
            if dataset_name in h5f:
                if overwrite:
                    del h5f[dataset_name]
                else:
                    print(f"Warning: Dataset {dataset_name} already exists in {output_file}. Skipping...")
                    continue
            # Only keep rows within the time range 09:30:00 to 15:59:00
            group['trade_time_obj'] = pd.Series(pd.to_datetime(group['trade_time'], format='%H:%M:%S')).dt.time
            group = group[(group['trade_time_obj'] >= start_time) & (group['trade_time_obj'] <= end_time)]
            # Create a DataFrame with all times in the range
            complete_times_df = pd.DataFrame({'trade_time': [t.strftime('%H:%M:%S') for t in all_times]})
            # Merge to ensure all times are present
            merged = pd.merge(complete_times_df, group, on='trade_time', how='left')
            # Forward fill then backward fill close
            merged['close'] = merged['close'].ffill().bfill()
            # DEBUG: Warn if any negative close prices before writing to HDF5
            if merged['close'].min() < 0:
                print(f"WARNING: Negative close price for {ts_code}-{date} in merged DataFrame!")
                print(merged[['trade_time', 'close']].head(10))
                print(merged[['trade_time', 'close']].tail(10))
            # If after filling there are still NaNs, skip this group
            if bool(merged['close'].isna().any()):
                print(f"Could not fill all close values for {ts_code}-{date}. Skipping...")
                continue
            # Sort by trade_time to ensure order
            merged = merged.sort_values('trade_time').reset_index(drop=True)
            # Find open_value: the first available open at 09:30:00 to 09:34:00 in the original group
            open_value = None
            for t in ['09:30:00', '09:31:00', '09:32:00', '09:33:00', '09:34:00']:
                open_rows = group[group['trade_time'] == t]['open']
                if isinstance(open_rows, pd.Series) and len(open_rows) > 0:
                    open_value = float(open_rows.iloc[0])
                    break
            if open_value is None:
                print(f"No open row between 09:30:00 and 09:34:00 for {ts_code}-{date}. Skipping...")
                continue
            # Calculate percentage change for each row
            percentage_change = (merged['close'].astype(float) - open_value) / open_value
            merged['pct_chg'] = percentage_change
            # Fill missing values for all columns
            for col in ['open', 'high', 'low', 'close', 'volume', 'pe', 'turnover_rate', 'total_share', 'float_share']:
                merged[col] = merged[col].ffill().bfill()
            # Convert trade_time to minutes since 09:30
            def time_to_minutes(tstr):
                h, m, s = map(int, tstr.split(':'))
                return (h - 9) * 60 + (m - 30) + s / 60
            trade_time_min = merged['trade_time'].apply(time_to_minutes).to_numpy()
            # Stack all features into a 2D array: [open, high, low, close, volume, pe, turnover_rate, total_share, float_share, trade_time_min, pct_chg]
            features = np.stack([
                merged['open'].astype(float).to_numpy(),
                merged['high'].astype(float).to_numpy(),
                merged['low'].astype(float).to_numpy(),
                merged['close'].astype(float).to_numpy(),
                merged['volume'].astype(float).to_numpy(),
                merged['pe'].astype(float).to_numpy(),
                merged['turnover_rate'].astype(float).to_numpy(),
                merged['total_share'].astype(float).to_numpy(),
                merged['float_share'].astype(float).to_numpy(),
                trade_time_min,
                merged['pct_chg'].to_numpy()
            ], axis=1)
            h5f.create_dataset(dataset_name, data=features)

def visualize_hdf5(input_file):
    """
    List all the datasets in the hdf5 file. 
    Then print out two random datasets.
    """
    import h5py
    with h5py.File(input_file, 'r') as h5f:
        dataset_names = list(h5f.keys())
        print("Datasets in file:", dataset_names)
        n_to_show = min(3, len(dataset_names))
        random_datasets = random.sample(dataset_names, n_to_show)
        print("\nShowing two random datasets:")
        for ds in random_datasets:
            print(f"\nDataset: {ds}")
            obj = h5f[ds]
            if isinstance(obj, h5py.Dataset):
                data = obj[:]
                print(data)
            else:
                print("[Not a dataset]")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_file', required=True, help='Path to the input csv file')
    argparser.add_argument('--output_file', default='data/prices.h5', help='Path to the output HDF5 file')
    argparser.add_argument('--overwrite', action='store_true', help='Overwrite existing datasets')
    args = argparser.parse_args()

    convert_csv_to_hdf5(args.input_file, args.output_file, args.overwrite)
    visualize_hdf5(args.output_file)