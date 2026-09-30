#!/usr/bin/env python3
"""
Test script to verify memory optimizations in train_two_gpu.py
"""

import os as _os
import sys as _sys
# Project modules live in src/; make them importable when run from tests/.
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))

import torch
import time
from dataset import ClosePrice

def test_memory_optimizations():
    """Test the memory optimizations"""
    print("🧪 Testing Memory Optimizations")
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
        # Test dataset loading with memory monitoring
        print("\n🔄 Testing dataset loading...")
        start_time = time.time()
        
        dataset = ClosePrice('h5_rty_data_combined_processed.h5')
        
        load_time = time.time() - start_time
        print(f"✅ Dataset loaded in {load_time:.2f} seconds")
        print(f"   Total items: {len(dataset)}")
        print(f"   Number of stocks: {dataset.n_stocks}")
        
        if device.type == 'cuda':
            memory_after_load = torch.cuda.memory_allocated() / 1e9
            max_memory = torch.cuda.max_memory_allocated() / 1e9
            print(f"GPU Memory after loading: {memory_after_load:.1f} GB")
            print(f"Max GPU Memory used: {max_memory:.1f} GB")
            print(f"Memory increase: {memory_after_load - memory_before:.1f} GB")
            
            if max_memory < 5.0:
                print("✅ SUCCESS: Memory usage is under 5GB as expected!")
            else:
                print(f"⚠️  WARNING: Memory usage is {max_memory:.1f}GB (expected <5GB)")
        
        # Test loading a few items to simulate training
        print("\n🔄 Testing item loading (simulating training)...")
        for i in range(5):
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
        
        # Test batch loading
        print("\n🔄 Testing batch loading...")
        batch_size = 16  # Same as train_two_gpu.py
        batch_items = []
        
        start_time = time.time()
        for i in range(batch_size):
            item = dataset[i]
            batch_items.append(item)
        batch_time = time.time() - start_time
        
        print(f"   Loaded {batch_size} items in {batch_time:.2f} seconds")
        print(f"   Average time per item: {batch_time/batch_size:.4f} seconds")
        
        if device.type == 'cuda':
            batch_memory = torch.cuda.memory_allocated() / 1e9
            print(f"   GPU Memory after batch: {batch_memory:.1f} GB")
        
        # Clean up
        dataset.close()
        
        if device.type == 'cuda':
            final_memory = torch.cuda.memory_allocated() / 1e9
            max_memory = torch.cuda.max_memory_allocated() / 1e9
            print(f"\n📊 Final GPU Memory: {final_memory:.1f} GB")
            print(f"📊 Max GPU Memory used: {max_memory:.1f} GB")
            print(f"📊 Total memory used: {max_memory - memory_before:.1f} GB")
            
            if max_memory < 5.0:
                print("✅ SUCCESS: Memory optimizations working correctly!")
            else:
                print(f"⚠️  WARNING: Memory usage is {max_memory:.1f}GB (expected <5GB)")
        
        print("\n✅ Memory optimization test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_memory_optimizations() 