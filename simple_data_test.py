#!/usr/bin/env python3
"""
Simple test to just load data and measure memory usage
"""

import torch
import time
from dataset import ClosePrice

def test_simple_data_loading():
    """Just load data and measure memory - no training, no complex processing"""
    print("🧪 Simple Data Loading Test")
    print("=" * 50)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Clear GPU memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        memory_before = torch.cuda.memory_allocated() / 1e9
        print(f"GPU Memory before test: {memory_before:.1f} GB")
    
    try:
        # Test 1: Just create the dataset (should be minimal memory)
        print("\n🔄 Test 1: Creating dataset (should be minimal memory)...")
        start_time = time.time()
        
        dataset = ClosePrice('h5_rty_data_combined_processed.h5')
        
        load_time = time.time() - start_time
        print(f"✅ Dataset created in {load_time:.2f} seconds")
        print(f"   Total items: {len(dataset)}")
        print(f"   Number of stocks: {dataset.n_stocks}")
        
        if device.type == 'cuda':
            memory_after_create = torch.cuda.memory_allocated() / 1e9
            print(f"GPU Memory after dataset creation: {memory_after_create:.1f} GB")
            print(f"Memory increase from creation: {memory_after_create - memory_before:.1f} GB")
        
        # Test 2: Access dataset.data property (this loads all data)
        print("\n🔄 Test 2: Accessing dataset.data (this loads all data)...")
        start_time = time.time()
        
        data = dataset.data
        
        load_time = time.time() - start_time
        print(f"✅ Data loaded in {load_time:.2f} seconds")
        print(f"   Data shape: {data.shape}")
        print(f"   Data device: {data.device}")
        print(f"   Data dtype: {data.dtype}")
        
        if device.type == 'cuda':
            memory_after_data = torch.cuda.memory_allocated() / 1e9
            max_memory = torch.cuda.max_memory_allocated() / 1e9
            print(f"GPU Memory after data loading: {memory_after_data:.1f} GB")
            print(f"Max GPU Memory used: {max_memory:.1f} GB")
            print(f"Memory increase from data loading: {memory_after_data - memory_after_create:.1f} GB")
            print(f"Total memory increase: {max_memory - memory_before:.1f} GB")
            
            # Calculate expected memory
            expected_memory = data.numel() * data.element_size() / 1e9
            print(f"Expected memory based on data size: {expected_memory:.1f} GB")
            
            if max_memory < 5.0:
                print("✅ SUCCESS: Memory usage is reasonable!")
            else:
                print(f"⚠️  WARNING: Memory usage is {max_memory:.1f}GB (expected <5GB)")
        
        # Test 3: Access a few individual items (should be minimal additional memory)
        print("\n🔄 Test 3: Accessing individual items...")
        for i in range(3):
            start_time = time.time()
            item = dataset[i]
            load_time = time.time() - start_time
            
            features, mean, std, stock_name, time_idx = item
            print(f"   Item {i}: {stock_name} at time {time_idx}")
            print(f"   Features shape: {features.shape}")
            print(f"   Load time: {load_time:.4f} seconds")
            
            if device.type == 'cuda':
                current_memory = torch.cuda.memory_allocated() / 1e9
                print(f"   GPU Memory: {current_memory:.1f} GB")
        
        # Clean up
        dataset.close()
        
        if device.type == 'cuda':
            final_memory = torch.cuda.memory_allocated() / 1e9
            max_memory = torch.cuda.max_memory_allocated() / 1e9
            print(f"\n📊 Final GPU Memory: {final_memory:.1f} GB")
            print(f"📊 Max GPU Memory used: {max_memory:.1f} GB")
            print(f"📊 Total memory used: {max_memory - memory_before:.1f} GB")
        
        print("\n✅ Simple data loading test completed!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_data_loading() 