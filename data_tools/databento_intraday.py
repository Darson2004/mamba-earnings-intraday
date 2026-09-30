import pandas as pd
from datetime import timedelta
from databento import Historical
import os
from dotenv import load_dotenv
load_dotenv()

# Set up Databento API
client = Historical()

# Load Excel file
# Earnings calendar workbook (NAME column + one or more *DATE* columns); override via env var.
excel_path = os.environ.get('EARNINGS_XLSX', 'earnings_equity_first.xlsx')
df = pd.read_excel(excel_path)

# Standardize columns
df.columns = [col.upper().strip() for col in df.columns]
ticker_col = "NAME"
date_cols = [col for col in df.columns if "DATE" in col]

# Schema and dataset
dataset = "XNAS.ITCH"  # or correct dataset
schema = "ohlcv-1m"

# Output list
results = []

output_file = "test_training_two.csv"

# Process each stock in batch
for _, row in df.iterrows():
    ticker_full = row[ticker_col]
    ticker = str(ticker_full).split()[0]
    print(f"Processing {ticker}")

    # Collect all valid dates
    date_list = []
    for date_col in date_cols:
        val = row[date_col]
        if pd.notna(val):
            try:
                date_list.append(pd.to_datetime(val).normalize())
            except:
                continue

    if not date_list:
        continue

    # Batch pull data from min(date) to max(date)+1
    min_date = min(date_list)
    max_date = max(date_list) + timedelta(days=1)
    start_dt = min_date.strftime("%Y-%m-%d")
    end_dt = max_date.strftime("%Y-%m-%d")

    try:
        store = client.timeseries.get_range(
            dataset=dataset,
            schema=schema,
            symbols=[ticker],
            start=start_dt,
            end=end_dt,
        )
        df_ohlcv = store.to_df()
    except Exception as e:
        print(f"Failed to get OHLCV for {ticker} from {start_dt} to {end_dt}: {e}")
        continue

    # Ensure Index is Datetime
    if not isinstance(df_ohlcv.index, pd.DatetimeIndex):
        df_ohlcv.index = pd.to_datetime(df_ohlcv.index)

    # Loosen Date Filtering
    for earnings_date in date_list:
        # Ensure earnings_date is timezone-aware (UTC)
        if earnings_date.tzinfo is None:
            earnings_date = earnings_date.tz_localize('UTC')
        start = earnings_date
        end = earnings_date + timedelta(days=2)
        mask = (df_ohlcv.index >= start) & (df_ohlcv.index < end)
        df_filtered = df_ohlcv[mask]
        for idx, minute_row in df_filtered.iterrows():
            results.append({
                "NAME": ticker_full,
                "DATE": idx.date(),
                "TIME": idx.time(),
                "OPEN": minute_row["open"],
                "HIGH": minute_row["high"],
                "LOW": minute_row["low"],
                "CLOSE": minute_row["close"],
                "VOLUME": minute_row["volume"],
            })
        
        # After processing this ticker, append results to CSV and clear results
        if results:
            import os
            write_header = not os.path.exists(output_file)
            pd.DataFrame(results).to_csv(output_file, mode='a', header=write_header, index=False)
            print(f"✅ Saved {len(results)} rows for {ticker} to {output_file}")
            results.clear()
