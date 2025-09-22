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
    def __init__(self, input_file: str, load_to_gpu: bool = False):
        """
        Load preprocessed HDF5 data with mask columns already included.
        
        Args: 
            input_file: str, the path to the preprocessed hdf5 file
            load_to_gpu: bool, if True, loads all data to GPU at initialization
        """
        self.day_length = 390 # between 09:30:00 and 16:00:00
        self.seq_len = 30
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get all dataset names (e.g. JELD UN Equity-2018-05-08)
        self.h5file = h5py.File(input_file, 'r')
        self.name_date = list(self.h5file.keys())
        
        # Limit number of stocks for memory management
        max_stocks = 200  # Adjust this based on your GPU memory
        if len(self.name_date) > max_stocks:
            print(f"⚠️  Limiting stocks from {len(self.name_date)} to {max_stocks} for memory management")
            self.name_date = self.name_date[:max_stocks]
        
        self.n_stocks = len(self.name_date)
        
        if load_to_gpu:
            # Load all data directly to GPU at initialization
            print(f"Loading all data directly to GPU: {self.n_stocks} stocks, {self.day_length} time points, 13 features")
            print(f"Expected GPU memory: {self.n_stocks * 390 * 13 * 4 / 1e9:.1f} GB")
            
            all_data = []
            for i, name in enumerate(self.name_date):
                arr = np.array(self.h5file[name])
                if not hasattr(arr, 'shape'):
                    arr = np.array(arr)
                
                # Verify data has the expected shape (13 features including masks)
                if arr.shape[1] != 13:
                    raise ValueError(f"Expected 13 features for preprocessed data, got {arr.shape[1]} for {name}")
                
                # Clean data if needed
                if np.isnan(arr).any():
                    print(f"Warning: Found NaNs in data for {name}, cleaning...")
                    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                
                all_data.append(arr)
                
                # Progress indicator for large datasets
                if (i + 1) % 100 == 0:
                    print(f"Loaded {i + 1}/{self.n_stocks} stocks to GPU...")
            
            # Stack all data into a tensor [n_stocks, day_length, features] and move to GPU
            self._data = torch.stack([
                torch.from_numpy(arr).float() for arr in all_data
            ]).to(self.device)
            
            print(f"✅ Loaded all data to GPU: {self._data.shape}")
            print(f"GPU memory usage: {self._data.numel() * self._data.element_size() / 1e9:.1f} GB")
            print(f"Device: {self.device}")
            
            # Close the HDF5 file since we don't need it anymore
            self.h5file.close()
            self.h5file = None
            
        else:
            # MEMORY-EFFICIENT VERSION: Loads data on-demand instead of all at once
            print(f"Memory-efficient loading: {self.n_stocks} stocks, {self.day_length} time points, 13 features")
            print(f"Expected memory per stock: {390 * 13 * 4 / 1e6:.1f} MB")
            print(f"Total memory if loaded all: {self.n_stocks * 390 * 13 * 4 / 1e9:.1f} GB")
            print(f"Memory-efficient approach: Load only current batch (~{min(32, self.n_stocks) * 390 * 13 * 4 / 1e6:.1f} MB)")
            
            # Pre-validate data shapes without loading all data
            print("Validating data shapes...")
            for i, name in enumerate(self.name_date[:5]):  # Check first 5 stocks
                arr = np.array(self.h5file[name])
                if not hasattr(arr, 'shape'):
                    arr = np.array(arr)
                
                print(f"  {name}: {arr.shape}")
                
                # Verify data has the expected shape (13 features including masks)
                if arr.shape[1] != 13:
                    raise ValueError(f"Expected 13 features for preprocessed data, got {arr.shape[1]} for {name}")
                
                # Check for NaNs
                if np.isnan(arr).any():
                    print(f"Warning: Found NaNs in preprocessed data for {name}, will clean on-demand")
            
            print("✅ Data validation complete. Using memory-efficient lazy loading.")

    def __getitem__(self, idx, seq_len=None):
        stock_idx = idx // self.day_length
        time_idx = idx % self.day_length
        seq_len = self.seq_len if seq_len is None else seq_len
        seq_start = max(0, time_idx - seq_len)
        
        # Check if data is already loaded to GPU
        if hasattr(self, '_data') and self._data is not None:
            # Use pre-loaded GPU data
            stock_data = self._data[stock_idx]  # [day_length, features]
        else:
            # MEMORY OPTIMIZATION: Load only the specific stock data needed
            # Check if HDF5 file is still open
            if not hasattr(self, 'h5file') or self.h5file is None:
                raise RuntimeError("HDF5 file is not available. It may have been closed or not properly initialized.")
            
            stock_name = self.name_date[stock_idx]
            try:
                arr = np.array(self.h5file[stock_name])
                if not hasattr(arr, 'shape'):
                    arr = np.array(arr)
                
                # Clean data if needed (on-demand)
                if np.isnan(arr).any():
                    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Convert to tensor and move to device
                stock_data = torch.from_numpy(arr).float().to(self.device)
            except Exception as e:
                print(f"Error loading data for {stock_name}: {e}")
                # Create zero array as fallback
                stock_data = torch.zeros((self.day_length, 13), dtype=torch.float32, device=self.device)
        
        # Extract the sequence we need
        features = stock_data[seq_start:time_idx, :]  # [<=seq_len, features]
        
        # Pad if needed
        if features.shape[0] < seq_len:
            pad_len = seq_len - features.shape[0]
            pad_tensor = torch.zeros((pad_len, features.shape[1]), dtype=features.dtype, device=features.device)
            features = torch.cat([pad_tensor, features], dim=0)
        
        # Rolling normalization: mean/std for this window only (causal)
        mean = features.mean(dim=0, keepdim=True)  # [1, features]
        std = features.std(dim=0, keepdim=True)
        std = torch.where(std > 1e-8, std, torch.ones_like(std))
        features_norm = (features - mean) / std
        
        return (features_norm, mean, std, self.name_date[stock_idx], time_idx)

    def __len__(self):
        return self.n_stocks * self.day_length

    @property
    def data(self):
        """
        Load all data into memory for backward compatibility with existing code.
        This is used by train_two_gpu.py which expects dataset.data to be available.
        """
        # If data is already loaded to GPU, return it
        if hasattr(self, '_data') and self._data is not None:
            return self._data
        
        # Otherwise, load all data (for backward compatibility)
        if not hasattr(self, '_data'):
            print("Loading all data into memory for backward compatibility...")
            print(f"⚠️  This will use ~{self.n_stocks * 390 * 13 * 4 / 1e9:.1f} GB of GPU memory")
            
            # Check if HDF5 file is still open
            if not hasattr(self, 'h5file') or self.h5file is None:
                raise RuntimeError("HDF5 file is not available. It may have been closed or not properly initialized.")
            
            all_data = []
            for i, name in enumerate(self.name_date):
                try:
                    arr = np.array(self.h5file[name])
                    if not hasattr(arr, 'shape'):
                        arr = np.array(arr)
                    
                    # Clean data if needed
                    if np.isnan(arr).any():
                        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    all_data.append(arr)
                    
                    # Progress indicator for large datasets
                    if (i + 1) % 50 == 0:
                        print(f"Loaded {i + 1}/{self.n_stocks} stocks...")
                except Exception as e:
                    print(f"Error loading data for {name}: {e}")
                    # Create zero array as fallback
                    fallback_arr = np.zeros((self.day_length, 13), dtype=np.float32)
                    all_data.append(fallback_arr)
            
            # Stack all data into a tensor [n_stocks, day_length, features]
            self._data = torch.stack([
                torch.from_numpy(arr).float() for arr in all_data
            ]).to(self.device)
            
            print(f"✅ Loaded all data: {self._data.shape}")
            print(f"Memory usage: {self._data.numel() * self._data.element_size() / 1e9:.1f} GB")
        
        return self._data

    def close(self):
        """Close the HDF5 file"""
        try:
            if hasattr(self, 'h5file') and self.h5file is not None:
                self.h5file.close()
                self.h5file = None
        except Exception as e:
            # Silently handle any errors during file closing
            pass

    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.close()
        except Exception as e:
            # Silently handle any errors during cleanup
            pass

    def denormalize_predictions(self, normalized_predictions):
        """
        Convert normalized predictions back to original percentage change scale.
        
        Args:
            normalized_predictions: numpy array or torch tensor of normalized predictions
            
        Returns:
            denormalized predictions in original percentage change scale
        """
        # The normalization parameters are no longer stored globally,
        # so we cannot denormalize directly here.
        # This method needs to be updated to accept mean/std if they were stored.
        # For now, we'll return the normalized predictions as is,
        # or raise an error if mean/std are not available.
        # Assuming mean/std are not available for denormalization in this new setup.
        print("Warning: Denormalization is not directly possible without global mean/std.")
        print("Please ensure mean/std are passed to this method if they were stored.")
        return normalized_predictions
    
    def get_normalization_info(self):
        """Return normalization parameters for debugging"""
        # The normalization parameters are no longer stored globally,
        # so this method needs to be updated to return None or raise an error.
        print("Normalization parameters are not available in this new setup.")
        return None


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
        # Test ClosePrice dataset - load all data to GPU
        print("=== Loading all data to GPU ===")
        dataset = ClosePrice('/home/ubuntu/Moldova/h5_rty_data_processed.h5', load_to_gpu=True)
        print(f"Dataset loaded: {len(dataset)} total items")
        print(f"Number of stocks: {dataset.n_stocks}")
        print(f"Day length: {dataset.day_length}")
        
        # Access the full dataset data on GPU
        full_data = dataset.data
        print(f"Full dataset shape: {full_data.shape}")
        print(f"Data device: {full_data.device}")
        print(f"Data dtype: {full_data.dtype}")
        
        # Test a few items
        for i in range(3):
            item = dataset[i]
            features = item[0]
            if hasattr(features, 'shape'):
                print(f"Item {i}: features shape {features.shape}, stock: {item[3]}, time: {item[4]}")
            else:
                print(f"Item {i}: features type {type(features)}, stock: {item[3]}, time: {item[4]}")
        
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
            time_indices = [item[4] for item in batch_items]
            stock_names = [item[3] for item in batch_items]
            
            print(f"  Time indices: {time_indices}")
            print(f"  Stock names: {stock_names}")
            print(f"  All same time: {len(set(time_indices)) == 1}")
            print(f"  All different stocks: {len(set(stock_names)) == len(stock_names)}")
            
    except FileNotFoundError:
        print("Test file '/home/ubuntu/Moldova/h5_rty_data_processed.h5' not found.")
    except Exception as e:
        print(f"Test failed with error: {e}")
