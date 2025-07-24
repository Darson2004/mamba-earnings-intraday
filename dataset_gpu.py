import torch
from torch.utils.data import Dataset, Sampler
import random
import pandas as pd
import h5py
import numpy as np

class LegacyDataset(Dataset):
    def __init__(self, input_file: str):
        """
        Args:
            input_file: str, the path to the csv file
        
        CSV format: 
        Time, Close
        09:30:00, 100
        09:31:00, 101
        09:32:00, 102
        ...
        """
        self.data = pd.read_csv(input_file)
        # Convert close prices to numpy array for easier indexing
        self.close_prices = self.data['Close'].values

    def __len__(self):
        return len(self.close_prices)

    def __getitem__(self, idx):
        """
        Return the close price for everything before and including the idx-th row.
        """
        # Return all close prices from index 0 to idx (inclusive)
        print(idx)
        print(self.close_prices[:idx].shape)
        return torch.tensor(self.close_prices[:idx], dtype=torch.float32)

class ClosePrice(Dataset):
    def __init__(self, input_file: str):
        """
        output_file format: 
        dataset name: NAME-DATE (e.g. JELD UN Equity-2018-05-08)
        Each dataset: shape [390, 13] (open, high, low, close, volume, trade_time, pct_chg, ... , turnover_rate, turnover_rate_mask)
        Args: 
            input_file: str, the path to the hdf5 file
        """
        self.day_length = 390 # between 09:30:00 and 16:00:00
        self.seq_len = 30
        # Load h5py file
        self.h5file = h5py.File(input_file, 'r')
        
        # Get all dataset names (e.g. JELD UN Equity-2018-05-08)
        self.name_date = list(self.h5file.keys())

        self.n_stocks = len(self.name_date)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # CRITICAL: Store normalization parameters for percentage changes (feature index 10)
        self.pct_change_mean = None
        self.pct_change_std = None
        self.normalization_params = {}  # Store params for all features
        
        # STEP 1: Collect ALL data first (before any normalization)
        all_raw_data = []
        
        for name in self.name_date:
            arr = np.array(self.h5file[name])
            if not hasattr(arr, 'shape'):
                arr = np.array(arr)
            # Assume turnover_rate is at index 6 and pe is at index 7
            turnover_idx = 6
            pe_idx = 5
            # Create masks
            turnover_nan_mask = np.isnan(arr[:, turnover_idx]).astype(np.float32)  # 1 if NaN, 0 if present
            pe_nan_mask = np.isnan(arr[:, pe_idx]).astype(np.float32)  # 1 if NaN, 0 if present
            # Impute NaNs in turnover_rate and pe
            arr[:, turnover_idx] = np.nan_to_num(arr[:, turnover_idx], nan=0.0)
            arr[:, pe_idx] = np.nan_to_num(arr[:, pe_idx], nan=0.0)
            # Add masks as new features
            arr = np.concatenate([arr, turnover_nan_mask[:, None], pe_nan_mask[:, None]], axis=1)
            
            # CRITICAL: Remove any remaining NaNs (don't just warn!)
            if np.isnan(arr).any():
                nan_count = np.isnan(arr).sum()
                print(f"CRITICAL: Removing {nan_count} NaNs from dataset {name}")
                # Replace ANY remaining NaNs with zeros
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                
            # CRITICAL: Detect and cap extreme outliers
            for col_idx in range(arr.shape[1]):
                col_data = arr[:, col_idx]
                # Use percentile-based outlier detection
                q1, q99 = np.percentile(col_data, [1, 99])
                # Cap extreme values (more conservative than removal)
                arr[:, col_idx] = np.clip(col_data, q1, q99)
            
            all_raw_data.append(arr)
            
        # STEP 2: Calculate PER-STOCK normalization parameters and apply normalization (GPU-optimized)
        print("📊 Calculating per-stock normalization parameters and applying normalization (GPU)...")
        data_list = []
        self.per_stock_normalization_params = []  # Store per-stock mean/std for each feature as tensors
        for i, arr in enumerate(all_raw_data):
            arr_torch = torch.from_numpy(arr).float().to(self.device)  # [day_length, features] on GPU
            col_means = arr_torch.mean(dim=0)  # [features], on GPU
            col_stds = arr_torch.std(dim=0)    # [features], on GPU
            self.per_stock_normalization_params.append({'mean': col_means, 'std': col_stds})
            # Avoid division by zero
            col_stds_safe = torch.where(col_stds > 1e-8, col_stds, torch.ones_like(col_stds))
            arr_norm = (arr_torch - col_means) / col_stds_safe
            arr_norm = torch.clamp(arr_norm, -3.0, 3.0)
            # Final validation
            if torch.isnan(arr_norm).any() or torch.isinf(arr_norm).any():
                raise ValueError(f"Dataset {self.name_date[i]} still contains NaN/Inf after processing!")
            if arr_norm.shape != (self.day_length, 13):
                raise ValueError(f"Dataset {self.name_date[i]} has shape {arr_norm.shape}, expected ({self.day_length}, 13)")
            data_list.append(arr_norm)
        print(f"✅ Per-stock normalization (GPU) applied to all {len(data_list)} stocks!")
        self.data = torch.stack(data_list).to(self.device)  # [n_stocks, day_length, features] on GPU
        print("self.data shape:", self.data.shape)
        print("self.data device:", self.data.device)
        print("self.data dtype:", self.data.dtype)
        print("Expected memory (GB):", self.data.numel() * self.data.element_size() / 1e9)
        
        # CRITICAL: Report data statistics after normalization
        print("\n=== DATA STATISTICS AFTER PREPROCESSING ===")
        for i in range(self.data.shape[2]):  # For each feature
            feature_data = self.data[:, :, i].flatten()
            print(f"Feature {i}: min={feature_data.min():.3f}, max={feature_data.max():.3f}, "
                  f"mean={feature_data.mean():.3f}, std={feature_data.std():.3f}")
        
        # Verify no NaN/Inf in final tensor
        if torch.isnan(self.data).any() or torch.isinf(self.data).any():
            raise ValueError("CRITICAL: Final tensor contains NaN/Inf!")
        
        print("✅ Dataset preprocessing complete - no NaN/Inf detected")
        self.h5file.close()

    def denormalize_predictions(self, normalized_predictions):
        """
        Convert normalized predictions back to original percentage change scale.
        
        Args:
            normalized_predictions: numpy array or torch tensor of normalized predictions
            
        Returns:
            denormalized predictions in original percentage change scale
        """
        if self.pct_change_mean is None or self.pct_change_std is None:
            raise ValueError("Normalization parameters not available. Dataset may not be properly initialized.")
        
        if isinstance(normalized_predictions, torch.Tensor):
            normalized_predictions = normalized_predictions.cpu().numpy()
        normalized_predictions = np.asarray(normalized_predictions, dtype=np.float64)
        # Apply inverse normalization: denormalized = normalized * std + mean
        denormalized = normalized_predictions * self.pct_change_std + self.pct_change_mean
        
        return denormalized
    
    def get_normalization_info(self):
        """Return normalization parameters for debugging"""
        return {
            'pct_change_mean': self.pct_change_mean,
            'pct_change_std': self.pct_change_std,
            'all_params': self.normalization_params
        }


    def __len__(self):
        return self.n_stocks * self.day_length

    def __getitem__(self, idx, seq_len=None):
        stock_idx = idx // self.day_length
        time_idx = idx % self.day_length
        seq_start = max(0, time_idx - self.seq_len)
        features = self.data[stock_idx, seq_start:time_idx, :]  # [<=seq_len, features]
        
        # Pad if needed
        if features.shape[0] < self.seq_len:
            pad_len = self.seq_len - features.shape[0]
            # CRITICAL: Create pad_tensor on same device as features
            pad_tensor = torch.zeros((pad_len, features.shape[1]), 
                                   dtype=features.dtype, 
                                   device=features.device)  # Same device as features
            features = torch.cat([pad_tensor, features], dim=0)

        return (features, self.name_date[stock_idx], time_idx)

class TimeAlignedSampler(Sampler):
    """Sampler that ensures each batch contains different stocks at the same time point"""
    
    def __init__(self, dataset: ClosePrice, batch_size: int, start_time: int, end_time = None, shuffle: bool = False):
        """
        Args:
            dataset: StockPriceDataset instance
            batch_size: Number of stocks per batch
            shuffle: Whether to shuffle the order of time points
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_stocks = self.dataset.n_stocks
        self.day_length = self.dataset.day_length
        self.start_time = start_time
        self.end_time = self.day_length if end_time is None else end_time
            
    def __iter__(self):
        # Generate batches where each batch contains different stocks at the same time point
        # Use generator pattern to avoid storing all batches in memory
        
        # Create time points sequence (shuffled if requested)
        stock_indices = list(range(self.n_stocks))
        time_points = list(range(self.start_time, self.end_time))
        if self.shuffle:
            random.shuffle(stock_indices)
            random.shuffle(time_points)
        
        # For each time point, yield batches of stocks
        for time_idx in time_points:
            # Create batches of stocks for this time point
            for i in range(0, self.n_stocks, self.batch_size):
                batch_stocks = stock_indices[i:min(i + self.batch_size, self.n_stocks)]
                # Convert stock indices to dataset indices for this time point
                # Use list comprehension for efficiency
                batch_indices = [stock_idx * self.day_length + time_idx for stock_idx in batch_stocks]
                yield batch_indices
    
    def __len__(self):
        # Calculate total number of batches
        # For each time point, we have ceil(n_stocks / batch_size) batches
        batches_per_time = (self.dataset.n_stocks + self.batch_size - 1) // self.batch_size
        return self.day_length * batches_per_time

def collate_fn(batch):
    close_prices = torch.stack([item[0] for item in batch])
    stock_names = [item[1] for item in batch]
    time_index = batch[0][2]
    
    return {
        'close_prices': close_prices,      # [batch_size, time_idx+1]
        'stock_names': stock_names,        # List of stock names
        'time_index': time_index       # int
    }


if __name__ == '__main__':
    # Simple test for ClosePrice and TimeAlignedSampler
    try:
        # Test ClosePrice dataset
        dataset = ClosePrice('h5_rty_data.h5')
        print(f"Dataset loaded: {len(dataset)} total items")
        print(f"Number of stocks: {dataset.n_stocks}")
        print(f"Day length: {dataset.day_length}")
        
        # Test a few items
        for i in range(3):
            item = dataset[i]
            features = item[0]
            if hasattr(features, 'shape'):
                print(f"Item {i}: close_prices shape {features.shape}, stock: {item[1]}, time: {item[2]}")
            else:
                print(f"Item {i}: close_prices type {type(features)}, stock: {item[1]}, time: {item[2]}")
        
        # Test TimeAlignedSampler
        batch_size = 1
        sampler = TimeAlignedSampler(dataset, batch_size=batch_size, start_time=30, shuffle=False)
        print(f"\nSampler created: {len(sampler)} total batches")
        
        # Test a few batches
        batch_iter = iter(sampler)
        for i in range(3):
            batch_indices = next(batch_iter)
            print(f"Batch {i}: indices {batch_indices}")
            
            # Verify all items in batch have same time_idx but different stock_idx
            batch_items = [dataset[idx] for idx in batch_indices]
            time_indices = [item[2] for item in batch_items]
            stock_names = [item[1] for item in batch_items]
            
            print(f"  Time indices: {time_indices}")
            print(f"  Stock names: {stock_names}")
            print(f"  All same time: {len(set(time_indices)) == 1}")
            print(f"  All different stocks: {len(set(stock_names)) == len(stock_names)}")
            
    except FileNotFoundError:
        print("Test file 'data/test_converter.h5' not found. Please create a test HDF5 file first.")
    except Exception as e:
        print(f"Test failed with error: {e}")
