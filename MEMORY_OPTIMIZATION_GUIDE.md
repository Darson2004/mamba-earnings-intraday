# Memory Optimization Guide for MambaStock Training

## Summary of Changes Made

The following optimizations have been applied to `train_two.py` to reduce CUDA memory usage:

### 1. **Batch Size Reduction**
- **Before**: `batch_size = 8`
- **After**: `batch_size = 2` (75% reduction)
- **Memory Impact**: ~75% reduction in batch memory usage

### 2. **Monte Carlo Samples Reduction**
- **Before**: `n_samples = 20`
- **After**: `n_samples = 10` (50% reduction)
- **Memory Impact**: ~50% reduction in uncertainty estimation memory

### 3. **Sequence Length Reduction**
- **Before**: `seq_len = 30`
- **After**: `seq_len = 20` (33% reduction)
- **Memory Impact**: ~33% reduction in sequence memory

### 4. **Prediction Length Reduction**
- **Before**: `pred_len = 20`
- **After**: `pred_len = 10` (50% reduction)
- **Memory Impact**: ~50% reduction in prediction memory

### 5. **Memory Management Improvements**
- Added `torch.cuda.empty_cache()` calls
- Added gradient clearing after backward pass
- Added memory monitoring functions
- Reduced debug output frequency

## Configuration Settings

The script now uses a configurable memory optimization system:

```python
# MEMORY OPTIMIZATION SETTINGS
MEMORY_SAVING_MODE = True  # Set to False to use original settings
BATCH_SIZE = 2 if MEMORY_SAVING_MODE else 8
MONTE_CARLO_SAMPLES = 10 if MEMORY_SAVING_MODE else 20
SEQUENCE_LENGTH = 20 if MEMORY_SAVING_MODE else 30
PREDICTION_LENGTH = 10 if MEMORY_SAVING_MODE else 20
```

## Additional Memory Reduction Options

If you still encounter OOM errors, try these additional optimizations:

### Option 1: Further Reduce Batch Size
```python
BATCH_SIZE = 1  # Single sample training
```

### Option 2: Reduce Monte Carlo Samples
```python
MONTE_CARLO_SAMPLES = 5  # Minimum for uncertainty estimation
```

### Option 3: Reduce Sequence Length
```python
SEQUENCE_LENGTH = 15  # Shorter sequences
```

### Option 4: Use Gradient Accumulation
```python
# Add gradient accumulation to simulate larger batch sizes
ACCUMULATION_STEPS = 4  # Accumulate gradients over 4 steps
```

### Option 5: Mixed Precision Training
```python
# Enable automatic mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### Option 6: Model Parallelism
```python
# Split model across multiple GPUs if available
model = torch.nn.DataParallel(model)
```

## Memory Usage Monitoring

The script now includes memory monitoring:
- GPU memory allocation tracking
- Memory cleanup after each step
- Progress reporting with memory usage

## Expected Memory Reduction

With all optimizations applied:
- **Batch processing**: ~75% reduction
- **Monte Carlo sampling**: ~50% reduction  
- **Sequence processing**: ~33% reduction
- **Prediction memory**: ~50% reduction
- **Overall**: ~60-70% total memory reduction

## Troubleshooting

### If you still get OOM errors:

1. **Check GPU memory usage**:
   ```python
   print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
   ```

2. **Reduce batch size further**:
   ```python
   BATCH_SIZE = 1
   ```

3. **Reduce Monte Carlo samples**:
   ```python
   MONTE_CARLO_SAMPLES = 5
   ```

4. **Use CPU training temporarily**:
   ```python
   device = torch.device("cpu")
   ```

5. **Enable gradient checkpointing** (if model supports it):
   ```python
   model.gradient_checkpointing_enable()
   ```

## Performance Impact

These optimizations may slightly reduce:
- Training speed (smaller batches)
- Uncertainty estimation accuracy (fewer Monte Carlo samples)
- Model capacity (shorter sequences)

However, the model should still train effectively with these settings.

## Recommended Settings for Different GPU Memory

| GPU Memory | Recommended Settings |
|------------|---------------------|
| 8GB | `BATCH_SIZE=1, MONTE_CARLO_SAMPLES=5, SEQUENCE_LENGTH=15` |
| 12GB | `BATCH_SIZE=2, MONTE_CARLO_SAMPLES=8, SEQUENCE_LENGTH=20` |
| 16GB | `BATCH_SIZE=4, MONTE_CARLO_SAMPLES=10, SEQUENCE_LENGTH=25` |
| 24GB+ | `BATCH_SIZE=8, MONTE_CARLO_SAMPLES=20, SEQUENCE_LENGTH=30` | 