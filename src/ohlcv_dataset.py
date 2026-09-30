# ohlcv_dataset.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import torch

class OHLCVDataset(Dataset):
    def __init__(self, csv_path, seq_len=60, pred_len=1):
        df = pd.read_csv(csv_path)
        df = df.sort_values('ts_code')
        # --- Convert trade_time to minutes since 09:30 ---
        def time_to_minutes(t):
            if isinstance(t, str):
                # Handle HH:MM:SS format
                parts = t.split(":")
                if len(parts) == 3:  # HH:MM:SS
                    h, m, s = map(int, parts)
                elif len(parts) == 2:  # HH:MM
                    h, m = map(int, parts)
                else:
                    raise ValueError(f"Unexpected time format: {t}")
            else:
                t = pd.to_datetime(t)
                h, m = t.hour, t.minute
            return (h - 9) * 60 + (m - 30)

        df["time_min"] = df["trade_time"].apply(time_to_minutes)

        self.data = df[["open", "high", "low", "close", "volume", "time_min"]].values.astype(np.float32)

        scaler = MinMaxScaler()
        self.data = scaler.fit_transform(self.data)

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.samples = [
            (self.data[i:i+seq_len], self.data[i+seq_len:i+seq_len+pred_len, 3])  # 'close' as target
            for i in range(len(self.data) - seq_len - pred_len)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
