import pandas as pd

# Read the main and extras CSV files
main_df = pd.read_csv("mightymerge.csv")
extras_df = pd.read_csv("rty_extras_filled_all_stocks_FIXED.csv")

# Merge the extras columns into the main dataframe on ts_code and date
merged = pd.merge(
    main_df,
    extras_df[["ts_code", "date", "pe", "turnover_rate", "total_share", "float_share"]],
    on=["ts_code", "date"],
    how="left"
)

# Save the merged dataframe to a new CSV file
merged.to_csv("mightymerge_with_extras.csv", index=False)

print("Merged file saved as mightymerge_with_extras.csv") 