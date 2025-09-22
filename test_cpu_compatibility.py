#!/usr/bin/env python3
"""
Test script to verify CPU compatibility of train_two.py
"""

import torch
import sys
import os

def test_cpu_compatibility():
    """Test if the training code can run on CPU"""
    
    print("🔍 Testing CPU compatibility...")
    
    # Test 1: Check PyTorch installation
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA not available - will use CPU")
    
    # Test 2: Check device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Selected device: {device}")
    
    # Test 3: Check if required files exist
    required_files = [
        "h5_rty_data_test.h5",
        "mambastock.pth"
    ]
    
    print("\n📁 Checking required files:")
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"⚠️  {file} not found")
    
    # Test 4: Test basic tensor operations on CPU
    print("\n🧪 Testing tensor operations on CPU:")
    try:
        x = torch.randn(10, 10).to(device)
        y = torch.randn(10, 10).to(device)
        z = torch.mm(x, y)
        print(f"✓ Matrix multiplication works on {device}")
        print(f"✓ Result shape: {z.shape}")
    except Exception as e:
        print(f"❌ Tensor operations failed: {e}")
        return False
    
    # Test 5: Test model import
    print("\n🤖 Testing model import:")
    try:
        from mambastock_model import MambaStock
        model = MambaStock(input_size=13, seq_len=30, pred_len=2).to(device)
        print(f"✓ Model created successfully on {device}")
        print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        return False
    
    # Test 6: Test dataset import
    print("\n📊 Testing dataset import:")
    try:
        from dataset import ClosePrice
        print("✓ Dataset class imported successfully")
    except Exception as e:
        print(f"❌ Dataset import failed: {e}")
        return False
    
    # Test 7: Test basic training operations
    print("\n🎯 Testing basic training operations:")
    try:
        # Create dummy data
        dummy_input = torch.randn(2, 30, 13).to(device)
        dummy_target = torch.randn(2, 2).to(device)
        
        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ Forward pass works: {output.shape}")
        
        # Loss calculation
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(output, dummy_target)
        print(f"✓ Loss calculation works: {loss.item():.6f}")
        
        # Backward pass
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("✓ Backward pass and optimization work")
        
    except Exception as e:
        print(f"❌ Training operations failed: {e}")
        return False
    
    print("\n✅ All CPU compatibility tests passed!")
    print("🚀 Ready to run train_two.py on CPU")
    return True

if __name__ == "__main__":
    success = test_cpu_compatibility()
    if success:
        print("\n🎉 CPU compatibility verified!")
        print("💡 You can now run: python train_two.py")
    else:
        print("\n❌ CPU compatibility test failed!")
        print("🔧 Please check the errors above and fix them")
        sys.exit(1) 