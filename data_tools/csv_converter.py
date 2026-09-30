import pandas as pd
import pytz
import os
from datetime import datetime
from tqdm import tqdm

def convert_utc_to_est_and_filter(input_file, output_file):
    """
    Convert earnings_ohlcv CSV from UTC to EST and filter to trading hours (9:30 AM - 4:00 PM EST)
    """
    print(f"Processing {input_file}...")
    
    # Load the CSV file
    df = pd.read_csv(input_file)
    
    # Convert DATE column to datetime
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    # Combine DATE and TIME into one datetime column (assume UTC)
    df['DATETIME'] = pd.to_datetime(df['DATE'].astype(str) + ' ' + df['TIME'])
    
    # Convert from UTC to US/Eastern with DST awareness
    df['DATETIME_EST'] = df['DATETIME'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
    
    # Extract EST date and time
    df['DATE_EST'] = df['DATETIME_EST'].dt.date
    df['TIME_EST'] = df['DATETIME_EST'].dt.strftime('%H:%M:%S')
    
    # Filter to only include times between 9:30 AM and 4:00 PM EST
    def is_trading_hours(time_str):
        """Check if time is between 9:30 AM and 4:00 PM EST"""
        try:
            time_obj = pd.to_datetime(time_str).time()
            start_time = pd.to_datetime('09:30:00').time()
            end_time = pd.to_datetime('16:00:00').time()
            return start_time <= time_obj <= end_time
        except:
            return False
    
    # Apply the filter
    df_filtered = df[df['TIME_EST'].apply(is_trading_hours)].reset_index(drop=True)
    
    # Create output DataFrame with EST times
    output_df = df_filtered[['NAME', 'DATE_EST', 'TIME_EST', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']].copy()
    
    # Rename columns for clarity
    output_df.columns = ['NAME', 'DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
    
    # Convert DATE_EST back to string format for CSV
    output_df['DATE'] = output_df['DATE'].astype(str)
    
    # Save to CSV
    output_df.to_csv(output_file, index=False)
    
    print(f"✅ Converted and filtered data saved to {output_file}")
    print(f"   Original rows: {len(df)}")
    print(f"   Filtered rows: {len(output_df)}")
    print(f"   Removed rows: {len(df) - len(output_df)}")
    
    return output_df

def main():
    """
    Main function to process all earnings_ohlcv CSV files
    """
    # List of input files to process
    input_files = [
        "test_training_two.csv",
    ]
    
    # Filter to only existing files
    existing_files = [f for f in input_files if os.path.exists(f)]
    
    if not existing_files:
        print("❌ No input files found!")
        return
    
    print(f"🔄 Processing {len(existing_files)} files...")
    
    # Process each file with progress bar
    for input_file in tqdm(existing_files, desc="Converting files", unit="file"):
        # Create output filename with EST_ prefix
        if input_file == "earnings_ohlcv_data.csv":
            output_file = "EST_earnings_ohlcv_data.csv"
        else:
            # Extract the suffix (e.g., "four", "six", "three")
            suffix = input_file.replace("earnings_ohlcv_data_", "").replace(".csv", "")
            output_file = f"EST_earnings_ohlcv_data_{suffix}.csv"
        
        # Convert and filter the file
        convert_utc_to_est_and_filter(input_file, output_file)
    
    print("\n🎉 All files processed!")

if __name__ == "__main__":
    main()
