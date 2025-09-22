# MambaStock Training - CPU Version

This version of the training code has been adapted to work on CPU while remaining GPU-ready for future use.

## 🖥️ CPU Compatibility

The code has been modified to:
- ✅ Work on CPU without CUDA
- ✅ Automatically detect and use GPU when available
- ✅ Include CPU-specific optimizations
- ✅ Provide progress indicators for slower CPU training
- ✅ Handle memory management for CPU

## 🚀 Quick Start

### 1. Test CPU Compatibility
```bash
python test_cpu_compatibility.py
```

### 2. Run Training
```bash
python train_two.py
```

## ⚙️ CPU Optimizations

The code automatically applies these optimizations when running on CPU:

- **Reduced batch size**: From 8 to 4 to prevent memory issues
- **Reduced Monte Carlo samples**: From 20 to 10 for faster training
- **Progress indicators**: Shows training progress every 5 steps
- **Memory cleanup**: Regular garbage collection during training
- **Thread optimization**: Uses optimal number of CPU threads

## 📊 Expected Performance

### CPU Training Speed
- **Training time**: ~2-5x slower than GPU
- **Memory usage**: Lower than GPU version
- **Accuracy**: Same as GPU version

### GPU vs CPU Comparison
| Metric | CPU | GPU |
|--------|-----|-----|
| Training Speed | 1x | 3-5x |
| Memory Usage | Lower | Higher |
| Setup Complexity | Simple | Requires CUDA |

## 🔧 Requirements

### Minimum Requirements
- Python 3.8+
- PyTorch (CPU version)
- NumPy
- Matplotlib
- Scikit-learn

### Installation
```bash
# Install PyTorch CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install numpy matplotlib scikit-learn h5py
```

## 📁 Required Files

Before running, ensure you have:
- `h5_rty_data_test.h5` - Your dataset file
- `mambastock.pth` - Pre-trained model weights
- `mambastock_model.py` - Model definition
- `dataset.py` - Dataset class

## 🎯 Features

### Scrolling Window Training
- Starts with 30 minutes of historical data
- Predicts next 2 minutes each step
- Appends actual data to sequence after each prediction
- Continues until 3:30 PM (210 minutes from 9:30 AM)

### Confidence Intervals
- Monte Carlo dropout with 10 samples (CPU) or 20 samples (GPU)
- 95% confidence intervals for predictions
- Coverage rate monitoring during training

### General Predictions
- At each timestep, predicts remaining minutes of trading day
- Updates predictions as sequence grows
- Provides uncertainty estimates

## 📈 Output Files

The training will generate:
- `mambastock_scrolling_final.pth` - Final trained model
- `training_confidence_intervals.png` - Training confidence intervals
- `scrolling_window_immediate_predictions.png` - Immediate predictions
- `scrolling_window_general_predictions.png` - General predictions with CI
- `scrolling_window_accuracy_over_time.png` - Accuracy over time
- `log_two.txt` - Training log

## 🔄 GPU Migration

When you're ready to use GPU:

1. **Install CUDA PyTorch**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Run the same code**:
   ```bash
   python train_two.py
   ```

The code will automatically detect and use GPU, with these improvements:
- ✅ Faster training (3-5x speedup)
- ✅ More Monte Carlo samples (20 instead of 10)
- ✅ Larger batch sizes
- ✅ No CPU-specific optimizations

## 🐛 Troubleshooting

### Common Issues

**"CUDA not available"**
- This is expected for CPU training
- The code will automatically use CPU

**"Out of memory"**
- The code automatically reduces batch size for CPU
- If still having issues, reduce `batch_size` in the config

**"Training is slow"**
- This is normal for CPU training
- Consider using GPU for better performance

**"File not found"**
- Ensure all required files are in the same directory
- Check file paths in the config section

## 📝 Configuration

Key parameters in `train_two.py`:

```python
# --- CONFIG ---
h5_path = "h5_rty_data_test.h5"  # Dataset file
seq_len = 30                      # Initial sequence length
pred_len = 2                      # Minutes to predict each step
batch_size = 8                    # Will be reduced to 4 for CPU
epochs = 1                        # Number of training epochs
lr = 1e-4                        # Learning rate
```

## 🎉 Success Indicators

When training completes successfully, you should see:
- ✅ "Scrolling window training and evaluation completed!"
- ✅ Generated plot files
- ✅ Final model saved
- ✅ Training statistics printed

## 📞 Support

If you encounter issues:
1. Run `python test_cpu_compatibility.py` to diagnose problems
2. Check the `log_two.txt` file for detailed error messages
3. Ensure all required files are present and accessible 