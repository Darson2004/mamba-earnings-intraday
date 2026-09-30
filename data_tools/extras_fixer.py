import pandas as pd
from datetime import timedelta

df = pd.read_csv("rty_extras.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["ts_code", "date"]).reset_index(drop=True)

filled_rows = []

for ts_code, group in df.groupby("ts_code"):
    group = group.sort_values("date").reset_index(drop=True)

    last_valid_pe = None
    last_valid_total_share = None
    last_valid_float_share = None

    for i in range(len(group) - 1):
        current = group.iloc[i]
        next_row = group.iloc[i + 1]

        # Update last known values if present
        if pd.notna(current["pe"]):
            last_valid_pe = current["pe"]
        if pd.notna(current["total_share"]):
            last_valid_total_share = current["total_share"]
        if pd.notna(current["float_share"]):
            last_valid_float_share = current["float_share"]

        filled_rows.append(current.to_dict())

        current_date = current["date"]
        next_date = next_row["date"]

        # Fill missing daily rows between current and next
        while current_date + timedelta(days=1) < next_date:
            current_date += timedelta(days=1)
            new_row = {
                "ts_code": ts_code,
                "date": current_date,
                "pe": last_valid_pe,
                "turnover_rate": "",  # intentionally blank
                "total_share": last_valid_total_share,
                "float_share": last_valid_float_share
            }
            filled_rows.append(new_row)

    filled_rows.append(group.iloc[-1].to_dict())

final_df = pd.DataFrame(filled_rows)
final_df = final_df.sort_values(["ts_code", "date"]).reset_index(drop=True)
final_df.to_csv("rty_extras_filled_all_stocks_FIXED.csv", index=False)
