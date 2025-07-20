import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

class MambaHDF5Dataset(Dataset):
    def __init__(self, h5_path, seq_len=60, pred_len=1):
        self.samples = []
        with h5py.File(h5_path, "r") as f:
            for symbol in f:
                # Convert h5py dataset to numpy array
                data = np.array(f[symbol])  # shape: [T,] = % change from 09:30
                
                # Check for invalid data
                if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                    print(f"Warning: Found NaN or Inf values in dataset {symbol}. Skipping...")
                    continue
                
                # Check for extremely large values (e.g., > 1000% change)
                if np.any(np.abs(data) > 10.0):  # 1000% change threshold
                    print(f"Warning: Found extremely large values (>1000%) in dataset {symbol}. Skipping...")
                    continue
                
                if len(data) < seq_len + pred_len:
                    continue
                    
                for i in range(len(data) - seq_len - pred_len):
                    x = data[i:i+seq_len]
                    y = data[i+seq_len:i+seq_len+pred_len]
                    
                    # Additional validation for individual sequences
                    if np.any(np.isnan(x)) or np.any(np.isinf(x)) or np.any(np.isnan(y)) or np.any(np.isinf(y)):
                        continue
                    if np.any(np.abs(x) > 10.0) or np.any(np.abs(y) > 10.0):
                        continue
                        
                    self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return (
            torch.tensor(x, dtype=torch.float32).unsqueeze(-1),  # shape: [seq_len, 1]
            torch.tensor(y, dtype=torch.float32).unsqueeze(-1),  # shape: [pred_len, 1]
        )
