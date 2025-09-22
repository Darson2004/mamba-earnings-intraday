#!/usr/bin/env python3
"""
Test script to verify memory-efficient loading works correctly.
"""

import torch
import time
from dataset import ClosePrice

def test_memory_efficient_loading():
    """Test the memory-efficient dataset loading"""
    print("🧪 Testing Memory-Efficient Dataset Loading")
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
        memory_before = torch.cuda.memory_allocated() / 1e9
        print(f"GPU Memory before test: {memory_before:.1f} GB")
    
    try:
        # Load dataset with memory-efficient approach
        print("\n🔄 Loading dataset...")
        start_time = time.time()
        
        dataset = ClosePrice('h5_rty_data_combined_processed.h5')
        
        load_time = time.time() - start_time
        print(f"✅ Dataset loaded in {load_time:.2f} seconds")
        print(f"   Total items: {len(dataset)}")
        print(f"   Number of stocks: {dataset.n_stocks}")
        print(f"   Day length: {dataset.day_length}")
        
        if device.type == 'cuda':
            memory_after_load = torch.cuda.memory_allocated() / 1e9
            print(f"GPU Memory after loading: {memory_after_load:.1f} GB")
            print(f"Memory increase from loading: {memory_after_load - memory_before:.1f} GB")
        
        # Test loading a few items
        print("\n🔄 Testing item loading...")
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
        
        # Test batch loading
        print("\n🔄 Testing batch loading...")
        batch_size = 32
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
            print(f"\n📊 Final GPU Memory: {final_memory:.1f} GB")
            print(f"📊 Total memory used: {final_memory - memory_before:.1f} GB")
            
            if final_memory < 5.0:
                print("✅ SUCCESS: Memory usage is under 5GB as expected!")
            else:
                print(f"⚠️  WARNING: Memory usage is {final_memory:.1f}GB (expected <5GB)")
        
        print("\n✅ Memory-efficient loading test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_memory_efficient_loading() 