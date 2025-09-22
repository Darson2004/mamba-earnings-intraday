import sys
import os
sys.path.append("/home/ubuntu/Moldova/")  # Adjust if you cloned elsewhere
from mambastock_model import MambaStock
from dataset import ClosePrice, TimeAlignedSampler, collate_fn
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
import time

# Memory optimization settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cudnn.benchmark = False  # Disable for memory
torch.backends.cudnn.deterministic = True

# MEMORY OPTIMIZATION APPROACH:
# 1. Batched stock processing (STOCK_BATCH_SIZE = 500)
# 2. TRUE gradient accumulation across stock batches (accum_steps)
# 3. Automatic Mixed Precision (AMP) for efficiency
# 4. Minimal memory cleanup (removed frequent empty_cache)
# This allows us to use full trading day history for more realistic predictions.

# --- CONFIG ---
# MEMORY OPTIMIZATION SETTINGS (adjust these if you still get OOM errors):
MEMORY_SAVING_MODE = True  # Set to False to use original settings
BATCH_SIZE = 1 if MEMORY_SAVING_MODE else 8  # Further reduced for memory
MONTE_CARLO_SAMPLES = 15 if MEMORY_SAVING_MODE else 20  # Further reduced for memory
SEQUENCE_LENGTH = 90 if MEMORY_SAVING_MODE else 90  # Increased to 90 minutes for 120-minute window
PREDICTION_LENGTH = 20  # Target window: model predicts 1 value representing 20-minute trend
STOCK_BATCH_SIZE = 500 if MEMORY_SAVING_MODE else 500  # Number of stocks per batch
GRADIENT_ACCUMULATION_STEPS = 4  # Accumulate gradients over 4 batches before stepping
EPOCHS = 5 
LEARNING_RATE = 1e-4  # Lower learning rate for fine-tuning from pretrained model

# IMPROVEMENTS FOR BETTER ACCURACY:
# 1. Increased ci_scale max from 5.0 to 10.0
# 2. Added input noise (std=0.01) during training
# 3. Simplified loss focusing on accuracy with smoothening regularization
# 4. Loss weights: alpha=0.9 (accuracy), beta=0.1 (smoothening regularization)

h5_path = "training_data.h5"  # <-- Training HDF5 file path
seq_len = SEQUENCE_LENGTH
pred_len = PREDICTION_LENGTH
batch_size = BATCH_SIZE
epochs = EPOCHS
lr = LEARNING_RATE

# Device detection - CPU for now, but GPU-ready for future
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    # Memory optimization for GPU
    torch.cuda.empty_cache()
    print(f"GPU Memory before training: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
else:
    print("Device: CPU (GPU not available)")
    print("Note: Code is GPU-ready and will automatically use GPU when available")
    print("⚠️  CPU training will be slower. Consider using GPU for better performance.")
    print("💡 To enable GPU: install PyTorch with CUDA support")

# Add stability monitoring
def check_model_health(model, step):
    """Check for NaN/Inf in model weights"""
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"⚠️  WARNING: {name} has NaN/Inf at step {step}")
            return False
    return True

def check_gradients(model, step):
    """Monitor gradient magnitudes"""
    total_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            if torch.isnan(param.grad).any():
                print(f"⚠️  WARNING: NaN gradient in {name} at step {step}")
                return False
    total_norm = total_norm ** (1. / 2)
    print(f"Gradient norm at step {step}: {total_norm:.4f}")  # Always print
    if total_norm > 1.0:  # Lower threshold for warning
        print(f"⚠️  WARNING: Large gradient norm {total_norm:.2f} at step {step}")
    return True

def monitor_memory_usage():
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
        return allocated, reserved
    return None, None

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log_two.txt', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# Force immediate flushing
for handler in logger.handlers:
    if isinstance(handler, logging.FileHandler):
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

def force_log_write(message):
    """Force immediate write to log file"""
    logger.info(message)
    # Force flush all file handlers
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()

# Log start of training
logger.info("="*60)
logger.info("STARTING MAMBA STOCK TRAINING (SCROLLING WINDOW)")
logger.info("="*60)
logger.info(f"Config: seq_len={seq_len}, pred_len={pred_len}, batch_size={batch_size}, epochs={epochs}, lr={lr}")
logger.info(f"Device: {device}")
logger.info(f"Dataset: {h5_path}")
logger.info("="*60)
logger.info("Training Loss Log:")
logger.info("Epoch | Loss")
logger.info("-" * 20)
logger.info("="*60)

# --- MODEL/LOSS/OPTIMIZER INIT MOVED INTO __main__ ---

# Verify model is ready for training
def verify_model_training_setup(model, device, input_size):
    """Verify that the model is properly set up for training"""
    print("🔍 Verifying model training setup...")
    
    # Check model is on correct device
    print(f"   Model device: {next(model.parameters()).device}")
    print(f"   Target device: {device}")
    
    # Check model parameters
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"   ✓ {name}: {param.shape}, requires_grad={param.requires_grad}")
        else:
            print(f"   ⚠️  {name}: {param.shape}, requires_grad={param.requires_grad}")
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    try:
        test_input = torch.randn(2, seq_len, input_size).to(device)
        test_input.requires_grad_(True)
        model.train()
        test_output = model(test_input)
        print(f"   ✓ Forward pass works: {test_output.shape}")
        print(f"   ✓ Input requires grad: {test_input.requires_grad}")
        print(f"   ✓ Output requires grad: {test_output.requires_grad}")
        
        # Test backward pass
        test_loss = torch.nn.functional.mse_loss(test_output, torch.randn_like(test_output))
        test_loss.backward()
        print(f"   ✓ Backward pass works")
        
        # Check gradients
        grad_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_count += 1
        
        print(f"   ✓ Gradients computed for {grad_count} parameters")
        
    except Exception as e:
        print(f"   ❌ Model verification failed: {e}")
        return False
    
    print("✅ Model training setup verified!")
    return True

# (Model verification is called after model is defined inside __main__)

# --- SCROLLING WINDOW TRAINING FUNCTIONS ---
def general_prediction_single_horizon(model, input_seq, device):
    """
    Make a single-horizon prediction using the last window only.
    
    Args:
        model: The trained model
        input_seq: Current input sequence [seq_len, features]
        device: Device to run on
    
    Returns:
        prediction: Single scalar prediction
    """
    model.eval()
    
    with torch.no_grad():
        # Apply advanced financial normalization for general prediction
        from advanced_normalization import apply_advanced_normalization
        
        # Ensure normalizer exists
        if not hasattr(model, '_advanced_normalizer'):
            raise RuntimeError("Advanced normalizer not initialized! Call create_advanced_normalizer() first.")
        
        # Ensure input is exactly seq_len (should already be sliced by caller)
        if input_seq.shape[0] != seq_len:
            raise ValueError(f"Input sequence must be exactly {seq_len} minutes, got {input_seq.shape[0]}")
        
        # Reshape for normalizer: [1, seq_len, features]
        input_reshaped = input_seq.unsqueeze(0) if input_seq.dim() == 2 else input_seq
        window_norm = apply_advanced_normalization(input_reshaped, seq_len, model._advanced_normalizer)
        window_norm = window_norm.squeeze(0)  # Back to [seq_len, features]
        
        # Make single prediction
        model_output = model(window_norm.unsqueeze(0))  # [1, 1]
        # Handle potential tuple output from model
        if isinstance(model_output, tuple):
            pred = model_output[0]
        else:
            pred = model_output
        pred = pred.squeeze().detach().cpu().item()
    
    return pred

def unified_scrolling_window_training_step(model, dataset, current_seq, current_seq_len, step_idx, 
                                         optimizer, loss_fn, device, pred_len=20,
                                         stock_batch_size=None, global_batch_counter=None, 
                                         scaler=None, accum_steps=4):
    """
    UNIFIED training function with TRUE gradient accumulation and AMP support.
    
    This single function handles both batched and non-batched training with:
    - Advanced financial normalization (log-relative prices, de-seasonalized volume, etc.)
    - Consistent target: 20-minute log return of close price
    - TRUE gradient accumulation across batches
    - Automatic Mixed Precision (AMP) support
    - Memory-efficient processing
    
    Args:
        model: The model to train
        dataset: The dataset  
        current_seq: Current sequence [n_stocks, current_seq_len, features]
        current_seq_len: Current sequence length
        step_idx: Current step index
        optimizer: The optimizer
        loss_fn: Loss function
        device: Device to run on
        pred_len: Number of minutes to predict (20)
        stock_batch_size: If provided, use batched processing
        global_batch_counter: For batched mode, tracks global batches
        scaler: GradScaler for AMP
        accum_steps: Number of batches to accumulate before stepping
    
    Returns:
        loss: Training loss for this step
        updated_seq: Updated sequence with new minute appended
        updated_seq_len: Updated sequence length
        stats: Training statistics
    """
    model.train()
    n_stocks = dataset.n_stocks
    
    # Apply advanced financial normalization (THE 13 methods in one place)
    from advanced_normalization import apply_advanced_normalization
    
    # Ensure normalizer exists (should be created once globally)
    if not hasattr(model, '_advanced_normalizer'):
        raise RuntimeError("Advanced normalizer not initialized! Call create_advanced_normalizer() first.")
    
    if stock_batch_size is not None and stock_batch_size < n_stocks:
        # BATCHED PROCESSING MODE with TRUE gradient accumulation
        avg_loss, avg_coverage = _unified_training_step_batched(
            model, dataset, current_seq, current_seq_len, step_idx,
            optimizer, loss_fn, device, pred_len, stock_batch_size, global_batch_counter,
            scaler, accum_steps
        )
        # Handle sequence update in main unified function
        loss_value = avg_loss
        directional_accuracy = avg_coverage
    else:
        # FULL STOCK PROCESSING MODE with AMP and proper accumulation
        # CRITICAL: Slice to last seq_len before normalization
        input_window = current_seq[:, -seq_len:, :]  # [n_stocks, seq_len, features]
        input_norm = apply_advanced_normalization(input_window, seq_len, model._advanced_normalizer)
        
        # Add training noise for regularization
        noise_std = 0.01
        input_norm = input_norm + torch.randn_like(input_norm) * noise_std
        
        # Vectorized target: 20-minute log return using close price (col=3)
        eps = 1e-8
        p0 = dataset.data[:, seq_len+step_idx-1, 3].to(device)  # [n_stocks]
        p1 = dataset.data[:, seq_len+step_idx+pred_len-1, 3].to(device)  # [n_stocks]
        actuals = (p1.add(eps).log() - p0.add(eps).log()).unsqueeze(-1)  # [n_stocks, 1]
        
        # Forward pass with AMP
        with autocast(enabled=(device.type == "cuda" and scaler is not None)):
            model_output = model(input_norm)  # [n_stocks, 1, 1]
            # Handle potential tuple output from model
            if isinstance(model_output, tuple):
                train_pred = model_output[0]
            else:
                train_pred = model_output
            train_pred = train_pred.squeeze(-1)  # [n_stocks, 1]
            loss = loss_fn(train_pred, actuals)
        
        # Directional accuracy
        directional_accuracy = (torch.sign(train_pred) == torch.sign(actuals)).float().mean().item() * 100.0
        
        # Scale loss for accumulation and backward pass
        scaled_loss = loss / accum_steps
        if scaler is not None and device.type == "cuda":
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # Only step optimizer every accum_steps (this is for single-batch mode, accumulation not typical)
        if global_batch_counter is not None:
            global_batch_counter[0] += 1  # FIX: Increment counter in non-batched path too
            if global_batch_counter[0] % accum_steps == 0:
                # FIX: Unscale before clipping when using AMP
                if scaler is not None and device.type == "cuda":
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                if scaler is not None and device.type == "cuda":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        else:
            # Fallback: step immediately (for compatibility, no accumulation)
            if scaler is not None and device.type == "cuda":
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            if scaler is not None and device.type == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        # Debug logging (every 100 steps)
        if step_idx % 100 == 0:
            print(f"📊 Training Step {step_idx}: Loss={loss.item():.6f}, Dir_Acc={directional_accuracy:.2f}%")
        
        loss_value = loss.item()
    
    # Vectorized: update sequence with next minute data (same for both modes)
    next_minute_data = dataset.data[:, seq_len+step_idx:seq_len+step_idx+1, :].to(device)  # [n_stocks, 1, features]
    
    # Update sequence
    updated_seq = torch.cat([current_seq, next_minute_data], dim=1)  # [n_stocks, current_seq_len+1, features]
    updated_seq_len = current_seq_len + 1
    
    # Return stats
    stats = {
        'directional_accuracy': directional_accuracy if 'directional_accuracy' in locals() else 0.0
    }
    
    return loss_value, updated_seq, updated_seq_len, stats

def _unified_training_step_batched(model, dataset, current_seq, current_seq_len, step_idx,
                                 optimizer, loss_fn, device, pred_len, stock_batch_size, global_batch_counter,
                                 scaler, accum_steps):
    """
    Helper function for batched processing with TRUE gradient accumulation.
    Uses the same advanced normalization and consistent target as the main function.
    Implements proper gradient accumulation across stock batches.
    """
    from advanced_normalization import apply_advanced_normalization
    
    n_stocks = dataset.n_stocks
    num_stock_batches = (n_stocks + stock_batch_size - 1) // stock_batch_size
    
    total_loss = 0
    all_coverage_rates = []
    
    # Process stocks in batches
    for batch_idx in range(num_stock_batches):
        start_stock = batch_idx * stock_batch_size
        end_stock = min((batch_idx + 1) * stock_batch_size, n_stocks)
        batch_stocks = list(range(start_stock, end_stock))
        
        # Extract batch data and slice to last seq_len
        batch_seq = current_seq[batch_stocks].clone()  # [batch_size, current_seq_len, features]
        batch_window = batch_seq[:, -seq_len:, :]  # [batch_size, seq_len, features]
        
        # Apply the SAME advanced financial normalization as unified function
        input_norm = apply_advanced_normalization(batch_window, seq_len, model._advanced_normalizer)
        
        # Add training noise
        noise_std = 0.01
        input_norm = input_norm + torch.randn_like(input_norm) * noise_std
        
        # Vectorized target: 20-minute log return using close price (col=3)
        eps = 1e-8
        batch_stocks_tensor = torch.tensor(batch_stocks, dtype=torch.long)
        p0 = dataset.data[batch_stocks_tensor, seq_len+step_idx-1, 3].to(device)  # [batch_size]
        p1 = dataset.data[batch_stocks_tensor, seq_len+step_idx+pred_len-1, 3].to(device)  # [batch_size]
        actuals = (p1.add(eps).log() - p0.add(eps).log()).unsqueeze(-1)  # [batch_size, 1]
        
        # Forward pass with AMP
        model.train()
        with autocast(enabled=(device.type == "cuda" and scaler is not None)):
            model_output = model(input_norm)  # [batch_size, 1, 1]
            # Handle potential tuple output from model
            if isinstance(model_output, tuple):
                train_pred = model_output[0]
            else:
                train_pred = model_output
            train_pred = train_pred.squeeze(-1)  # [batch_size, 1]
            batch_loss = loss_fn(train_pred, actuals)
        
        # Directional accuracy for monitoring
        coverage_rate = (torch.sign(train_pred) == torch.sign(actuals)).float().mean().item() * 100.0

        # TRUE gradient accumulation: scale loss and accumulate
        scaled_loss = batch_loss / accum_steps
        if scaler is not None and device.type == "cuda":
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # Only step optimizer every accum_steps batches
        global_batch_counter[0] += 1
        should_step = global_batch_counter[0] % accum_steps == 0
        if should_step:
            # FIX: Unscale before clipping when using AMP
            if scaler is not None and device.type == "cuda":
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            if scaler is not None and device.type == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        # Store batch statistics  
        total_loss += batch_loss.item()
        all_coverage_rates.append(coverage_rate)
        
        # Store batch statistics before cleanup
        batch_loss_value = batch_loss.item()
        
        # Clear batch data from memory (removed frequent empty_cache for performance)
        del batch_seq, batch_window, input_norm
        del actuals, train_pred, batch_loss

        # Save model every 100 effective steps (after accumulation)
        current_batch = global_batch_counter[0]
        if should_step and (current_batch // accum_steps) % 100 == 0:
            torch.save(model.state_dict(), "mambastock.pth")
            print(f"💾 Model saved at global batch {current_batch} (effective step {current_batch // accum_steps})")
            print(f"   Latest batch loss (MSE): {batch_loss_value:.6f}, Directional Accuracy: {coverage_rate:.2f}%")
    
    # FIX: Flush any residual gradients if last partial accumulation didn't trigger step
    if (global_batch_counter[0] % accum_steps) != 0:
        print(f"🔧 Flushing residual gradients: {global_batch_counter[0] % accum_steps} batches pending")
        # Unscale before clipping when using AMP
        if scaler is not None and device.type == "cuda":
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        if scaler is not None and device.type == "cuda":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    
    # Calculate overall statistics
    avg_loss = total_loss / num_stock_batches
    avg_coverage = np.mean(all_coverage_rates)
    
    # Return same format as main function: (loss, avg_coverage, updated_seq, updated_seq_len, stats)
    # Sequence update is handled by caller
    stats = {
        'directional_accuracy': avg_coverage
    }
    
    return avg_loss, avg_coverage

# REMOVE OLD FUNCTION - keeping only for backward compatibility temporarily
# OLD FUNCTION REMOVED - All training now unified in unified_scrolling_window_training_step()

def train_scrolling_window(dataset, model, optimizer, loss_fn, device, epochs=1, 
                         seq_len=90, pred_len=20, end_time=300,
                         use_batched_training=True, stock_batch_size=50, accum_steps=4):
    """
    Train the model using growing window approach with confidence intervals.
    
    The sequence grows naturally from seq_len (90 minutes) to end_time (300 minutes),
    using all available historical data for each prediction. This provides more
    realistic predictions by incorporating full market context.
    
    Args:
        dataset: The dataset
        model: The model to train
        optimizer: The optimizer
        loss_fn: Loss function
        device: Device to run on
        epochs: Number of epochs
        seq_len: Initial sequence length (90 minutes)
        pred_len: Number of minutes to predict each step
        end_time: End time in minutes (2:30 PM = 300 minutes from 9:30 AM)
        n_samples: Number of Monte Carlo samples for uncertainty estimation
        use_batched_training: Whether to use batched stock processing
        stock_batch_size: Number of stocks to process in each batch
    """
    # No Monte Carlo sampling needed
    n_stocks = dataset.n_stocks
    total_steps = dataset.day_length
    
    # Calculate maximum scroll steps (until 3:30 PM)
    max_scroll_steps = min(end_time - seq_len, total_steps - seq_len - pred_len)
    
    print(f"Growing window training: seq_len={seq_len}, pred_len={pred_len}")
    print(f"Window grows from {seq_len} to {end_time} minutes")
    print(f"Total available steps: {total_steps}, Max scroll steps: {max_scroll_steps}")
    print(f"Training until: {end_time} minutes (2:30 PM)")
    print(f"Using deterministic forward pass with TRUE gradient accumulation")
    print(f"Batched training: {'✅ Enabled' if use_batched_training else '❌ Disabled'}")
    print(f"Gradient accumulation steps: {accum_steps}")
    if use_batched_training:
        print(f"Stock batch size: {stock_batch_size}")
    
    # Store training statistics
    all_coverage_rates = []
    
    # Initialize gradient scaler for AMP
    scaler = GradScaler(enabled=(device.type == "cuda"))
    
    # Track total batches across all epochs
    total_batches = 0
    global_batch_counter = [0]  # Use list to allow modification inside function
    
    # Force create/overwrite mambastock.pth at start
    torch.save(model.state_dict(), "mambastock.pth")
    print(f"💾 Initial model saved to mambastock.pth (overwriting if exists)")
    
    # Also save a backup with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_filename = f"mambastock_backup_{timestamp}.pth"
    torch.save(model.state_dict(), backup_filename)
    print(f"💾 Backup saved to {backup_filename}")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # No smoothing/chaining state retained between steps
        
        # Initialize with first seq_len minutes
        current_seq = dataset.data[:, :seq_len, :].clone().to(device)  # [n_stocks, seq_len, features]
        current_seq_len = seq_len
        
        total_loss = 0
        step_count = 0
        epoch_coverage_rates = []
        
        for step_idx in range(max_scroll_steps):
            # Show progress for CPU training
            if device.type == 'cpu' and step_idx % 5 == 0:
                print(f"CPU Training Progress: {step_idx}/{max_scroll_steps} ({step_idx/max_scroll_steps*100:.1f}%)")
            
            # UNIFIED training step with TRUE gradient accumulation and AMP
            result = unified_scrolling_window_training_step(
                model, dataset, current_seq, current_seq_len, step_idx,
                optimizer, loss_fn, device, pred_len,
                stock_batch_size=stock_batch_size if use_batched_training else None,
                global_batch_counter=global_batch_counter,
                scaler=scaler,
                accum_steps=accum_steps
            )
            
            # Check if training step failed
            if result[0] is None:
                print(f"⚠️  Training step {step_idx} failed, skipping...")
                continue
                
            loss, current_seq, current_seq_len, ci_stats = result
            
            total_loss += loss
            step_count += 1
            
            # Store training statistics
            if ci_stats is not None and 'directional_accuracy' in ci_stats:
                epoch_coverage_rates.append(ci_stats['directional_accuracy'])
            
            # Log progress every 10 steps
            if step_count % 10 == 0:
                avg_loss = total_loss / step_count
                avg_coverage = np.mean(epoch_coverage_rates[-10:]) if epoch_coverage_rates else 0
                
                print(f"Step {step_count}/{max_scroll_steps}, Avg Loss: {avg_loss:.6f}")
                print(f"  Directional Accuracy: {avg_coverage:.2f}%")
                logger.info(f"Epoch {epoch+1}, Step {step_count}/{max_scroll_steps}, Avg Loss: {avg_loss:.6f}")
                logger.info(f"  Directional Accuracy: {avg_coverage:.2f}%")
                
                # Memory monitoring (removed frequent empty_cache for performance)
                monitor_memory_usage()
            
            # Calculate batches processed in this step
            if use_batched_training:
                batches_this_step = (dataset.n_stocks + stock_batch_size - 1) // stock_batch_size
            else:
                batches_this_step = 1
            
            # Update total batches
            total_batches += batches_this_step
            
            # Debug: Print progress every 10 steps
            if step_count % 10 == 0:
                print(f"🔍 Debug: Step {step_count}, Global batches: {global_batch_counter[0]}")
        
        # Store epoch statistics
        all_coverage_rates.extend(epoch_coverage_rates)
        
        # Final save for this epoch
        torch.save(model.state_dict(), "mambastock.pth")
        
        if step_count > 0:
            avg_total_loss = total_loss / step_count
            avg_epoch_coverage = np.mean(epoch_coverage_rates) if epoch_coverage_rates else 0
            
            print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_total_loss:.6f} (Total: {total_loss:.2f} across {step_count:,} steps)")
            print(f"  Epoch Avg Directional Accuracy: {avg_epoch_coverage:.2f}%")
            force_log_write(f"{epoch+1} | Avg: {avg_total_loss:.6f} | Total: {total_loss:.2f} | Steps: {step_count}")
            force_log_write(f"  Directional Accuracy: {avg_epoch_coverage:.2f}%")
        else:
            print(f"Epoch {epoch+1}/{epochs} - No valid steps processed")
            force_log_write(f"{epoch+1} | No valid steps")
    
    # Return training statistics
    return {
        'coverage_rates': all_coverage_rates
    }

def evaluate_scrolling_window_with_general_predictions(dataset, model, seq_len=90, pred_len=20, 
                                                    device=None, end_time=300):
    """
    Evaluate the model using growing window approach with general predictions.
    
    The sequence grows naturally from seq_len (90 minutes) to end_time (300 minutes),
    using all available historical data for each prediction. This provides more
    realistic predictions by incorporating full market context.
    
    Args:
        dataset: The dataset
        model: The trained model
        seq_len: Initial sequence length (90 minutes)
        pred_len: Number of minutes to predict each step
        device: Device to run on
        end_time: End time in minutes (2:30 PM = 300 minutes from 9:30 AM)
        n_samples: Number of Monte Carlo samples for uncertainty estimation
    
    Returns:
        results: Dictionary containing evaluation results
    """
    model.eval()
    n_stocks = dataset.n_stocks
    total_steps = dataset.day_length
    
    # Calculate maximum scroll steps (until 2:30 PM = 300 minutes)
    max_scroll_steps = min(end_time - seq_len, total_steps - seq_len - pred_len)
    
    print(f"Evaluating growing window: seq_len={seq_len}, pred_len={pred_len}")
    print(f"Window grows from {seq_len} to {end_time} minutes")
    print(f"Total available steps: {total_steps}, Max scroll steps: {max_scroll_steps}")
    print(f"Evaluating until: {end_time} minutes (2:30 PM)")
    
    all_preds = []
    all_trues = []
    all_general_preds = []
    all_timesteps = []
    
    # Initialize with first seq_len minutes
    current_seq = dataset.data[:, :seq_len, :].clone().to(device)  # [n_stocks, seq_len, features]
    current_seq_len = seq_len
    
    for step_idx in range(max_scroll_steps):
        # Make single-horizon general predictions
        general_predictions = []
        
        for stock_idx in range(n_stocks):
            # CRITICAL: Use only the last seq_len window for prediction
            stock_window = current_seq[stock_idx, -seq_len:, :]  # [seq_len, features]
            
            # Make single-horizon general prediction
            pred = general_prediction_single_horizon(model, stock_window, device)
            general_predictions.append(pred)
        
        # Apply advanced financial normalization for evaluation (same as training)
        from advanced_normalization import apply_advanced_normalization
        
        # Ensure normalizer exists (should be created once globally)
        if not hasattr(model, '_advanced_normalizer'):
            raise RuntimeError("Advanced normalizer not initialized! Call create_advanced_normalizer() first.")
        
        # CRITICAL: Slice to last seq_len before normalization
        input_window = current_seq[:, -seq_len:, :]  # [n_stocks, seq_len, features]
        input_norm = apply_advanced_normalization(input_window, seq_len, model._advanced_normalizer)
        
        # Make immediate predictions
        model_output = model(input_norm)  # [n_stocks, 1, 1]
        # Handle potential tuple output from model
        if isinstance(model_output, tuple):
            preds = model_output[0]
        else:
            preds = model_output
        preds = preds.squeeze(-1)  # [n_stocks, 1] - remove last dimension
        
        # Vectorized target: 20-minute log return using close price (col=3)
        eps = 1e-8
        p0 = dataset.data[:, seq_len+step_idx-1, 3].to(device)  # [n_stocks]
        p1 = dataset.data[:, seq_len+step_idx+pred_len-1, 3].to(device)  # [n_stocks]
        actuals = (p1.add(eps).log() - p0.add(eps).log()).unsqueeze(-1)  # [n_stocks, 1]
        
        # Store results
        all_preds.append(preds.detach().cpu())
        all_trues.append(actuals.cpu())
        all_general_preds.append(general_predictions)
        all_timesteps.append(step_idx)
        
        # Vectorized: append the actual data for the next minute to the sequence
        next_minute_data = dataset.data[:, seq_len+step_idx:seq_len+step_idx+1, :].to(device)  # [n_stocks, 1, features]
        
        # Update sequence by appending the new minute
        current_seq = torch.cat([current_seq, next_minute_data], dim=1)  # [n_stocks, current_seq_len+1, features]
        current_seq_len += 1
        
        # Keep all available history - no artificial truncation
        # The sequence will grow naturally from seq_len to end_time
        # This provides more realistic predictions using full market context
        # Memory is managed through gradient accumulation and batched processing
        
        # Log progress every 10 steps
        if (step_idx + 1) % 10 == 0:
            print(f"Evaluation step {step_idx + 1}/{max_scroll_steps}")
    
    # Concatenate all predictions
    if all_preds:
        all_preds = torch.cat(all_preds, dim=1)  # [n_stocks, total_pred_len]
        all_trues = torch.cat(all_trues, dim=1)  # [n_stocks, total_pred_len]
    else:
        # Handle case where no predictions were made
        all_preds = torch.empty((n_stocks, 0))
        all_trues = torch.empty((n_stocks, 0))
    
    # Debug shapes
    print(f"Debug - all_preds shape: {all_preds.shape}")
    print(f"Debug - all_trues shape: {all_trues.shape}")
    
    # Ensure shapes match before flattening
    if all_preds.shape != all_trues.shape:
        print(f"⚠️  Shape mismatch! Truncating to smaller size")
        min_len = min(all_preds.shape[1], all_trues.shape[1])
        all_preds = all_preds[:, :min_len]
        all_trues = all_trues[:, :min_len]
        print(f"Debug - After truncation: all_preds shape: {all_preds.shape}, all_trues shape: {all_trues.shape}")
    
    # Calculate metrics
    all_preds_np = all_preds.numpy().flatten()
    all_trues_np = all_trues.numpy().flatten()
    
    print(f"Debug - all_preds_np shape: {all_preds_np.shape}")
    print(f"Debug - all_trues_np shape: {all_trues_np.shape}")
    
    valid = ~np.isnan(all_preds_np) & ~np.isnan(all_trues_np)
    mse = np.mean((all_preds_np[valid] - all_trues_np[valid]) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_preds_np[valid] - all_trues_np[valid]))
    
    # Directional accuracy
    directions = np.sign(all_preds_np[valid]) == np.sign(all_trues_np[valid])
    directional_acc = np.mean(directions) * 100
    
    # Correlation
    correlation = np.corrcoef(all_preds_np[valid], all_trues_np[valid])[0, 1] if np.sum(valid) > 1 else float('nan')
    if np.isnan(correlation):
        correlation = 0.0
    
    print("\nScrolling Window Evaluation Results:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Directional Accuracy: {directional_acc:.2f}%")
    print(f"  Correlation: {correlation:.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'directional_accuracy': directional_acc,
        'correlation': correlation,
        'predictions': all_preds_np,
        'actuals': all_trues_np,
        'general_predictions': all_general_preds,
        'timesteps': all_timesteps
    }

# --- TRAIN LOOP ---
if __name__ == "__main__":
    print("Starting growing window training...")
    
    # --- LOAD DATA ---
    dataset = ClosePrice(h5_path)
    print(f"Dataset loaded: {len(dataset)} total items, {dataset.n_stocks} stocks")
    logger.info(f"Dataset loaded: {len(dataset)} total items, {dataset.n_stocks} stocks")
    logger.info("="*60)
    print(f"Starting all calculations after {seq_len} data points (time index {seq_len}) to exclude first {seq_len} points")
    
    # CPU-specific optimizations
    if device.type == 'cpu':
        print("🔧 Applying CPU optimizations...")
        # Set number of threads for CPU operations
        torch.set_num_threads(min(8, torch.get_num_threads()))
        print(f"Using {torch.get_num_threads()} CPU threads")
        # Reduce batch size for CPU to prevent memory issues
        if batch_size > 4:
            print(f"⚠️  Reducing batch size from {batch_size} to 4 for CPU training")
            batch_size = 4
    
    # --- MODEL INIT --- (set input_size dynamically from dataset)
    inferred_input_size = int(dataset.data.shape[-1]) if hasattr(dataset, 'data') else 13
    model = MambaStock(input_size=inferred_input_size, seq_len=seq_len, pred_len=1).to(device)
    
    # No CI parameters needed
    try:
        state_dict = torch.load("mambastock.pth", map_location=device)
        model.load_state_dict(state_dict)
        print("Loaded weights from mambastock.pth")
    except FileNotFoundError:
        print("mambastock.pth not found, starting with random weights")
    except Exception as e:
        print(f"Error loading mambastock.pth: {e}")
    
    # --- LOSS FUNCTION ---
    loss_fn = nn.MSELoss().to(device)
    
    # --- OPTIMIZER ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Initialize advanced normalizer with training data for cross-sectional stats
    print("🔧 Initializing advanced financial normalizer...")
    from advanced_normalization import create_advanced_normalizer
    # FIX: Use only historical data to prevent temporal leakage
    model._advanced_normalizer = create_advanced_normalizer(dataset.data, use_historical_only=True)
    print("✅ Advanced normalizer initialized with historical data only (no temporal leakage)")
    
    # Verify model is ready for training
    verify_model_training_setup(model, device, inferred_input_size)
    
    # Train the model using scrolling window approach with TRUE gradient accumulation
    training_stats = train_scrolling_window(dataset, model, optimizer, loss_fn, device, 
                                         epochs=epochs, seq_len=seq_len, pred_len=pred_len, 
                                         end_time=300,
                                         use_batched_training=True, stock_batch_size=STOCK_BATCH_SIZE,
                                         accum_steps=GRADIENT_ACCUMULATION_STEPS)
    
    # Save final model
    torch.save(model.state_dict(), "mambastock.pth")
    logger.info("Final model saved to mambastock.pth")
    
    # Plot training statistics
    if training_stats and len(training_stats['coverage_rates']) > 0:
        print("\nCreating training statistics plots...")
        
        plt.figure(figsize=(10, 6))
        
        # Plot directional accuracy over training steps
        plt.plot(training_stats['coverage_rates'], 'g-', linewidth=2)
        plt.axhline(y=50, color='black', linestyle='--', alpha=0.7, label='Random Guess (50%)')
        plt.title('Directional Accuracy During Training')
        plt.xlabel('Training Steps')
        plt.ylabel('Directional Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_directional_accuracy.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print final training statistics
        print(f"\nFinal Training Statistics:")
        print(f"  Average Directional Accuracy: {np.mean(training_stats['coverage_rates']):.2f}%")
        print(f"  Directional Accuracy Range: [{np.min(training_stats['coverage_rates']):.2f}%, {np.max(training_stats['coverage_rates']):.2f}%]")
    
    # Evaluate the model
    print("\nEvaluating model with general predictions...")
    try:
        results = evaluate_scrolling_window_with_general_predictions(
            dataset, model, seq_len=seq_len, pred_len=20, 
            device=device, end_time=300
        )
    except Exception as e:
        print(f"⚠️  Evaluation failed: {e}")
        print("Continuing with plotting...")
        results = None
    
    # Create evaluation plots
    print("\nCreating evaluation plots...")
    
    if results is not None:
        # Plot 1: Immediate predictions vs actual
        plt.figure(figsize=(12, 8))
        
        # Plot predictions for first few stocks
        for stock_idx in range(min(3, dataset.n_stocks)):
            stock_preds = results['predictions'].reshape(dataset.n_stocks, -1)[stock_idx]
            stock_actuals = results['actuals'].reshape(dataset.n_stocks, -1)[stock_idx]
            
            plt.subplot(3, 1, stock_idx + 1)
            plt.plot(stock_actuals, 'b-', label='Actual', linewidth=2)
            plt.plot(stock_preds, 'r--', label='Predicted', linewidth=2)
            plt.title(f'Stock {stock_idx + 1}: Immediate Predictions vs Actual')
            plt.xlabel('Time Steps')
            plt.ylabel('Percentage Change')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('scrolling_window_immediate_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("⚠️  Skipping evaluation plots due to evaluation failure")
    
    # Plot 2: General predictions (single horizon, no CI)
    if results is not None and len(results['general_predictions']) > 0:
        plt.figure(figsize=(14, 10))
        
        # Plot general predictions for first stock at different timesteps
        stock_idx = 0
        timesteps_to_plot = [0, len(results['timesteps'])//4, len(results['timesteps'])//2, len(results['timesteps'])-1]
        
        for i, timestep in enumerate(timesteps_to_plot):
            if timestep < len(results['general_predictions']):
                # Single-horizon prediction (scalar value)
                general_pred = results['general_predictions'][timestep][stock_idx]
                
                plt.subplot(2, 2, i + 1)
                
                # Plot as single point prediction
                plt.scatter([0], [general_pred], color='red', s=100, label='Single-Horizon Prediction', zorder=3)
                plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                plt.axhline(y=general_pred, color='red', linestyle='-', alpha=0.7)
                plt.title(f'Single-Horizon Prediction at Timestep {timestep}')
                plt.xlabel('Time (Current = 0)')
                plt.ylabel('Predicted Log Return')
                plt.xlim([-0.5, 0.5])
                plt.legend()
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('scrolling_window_general_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("⚠️  Skipping general predictions plot due to evaluation failure or no data")
    
    # Plot 3: Prediction accuracy over time
    if results is not None and len(results['predictions']) >= 10:
        plt.figure(figsize=(12, 8))
        
                # Calculate scrolling accuracy (not rolling)
        window_size = 10
        if len(results['predictions']) >= window_size:
            scrolling_mse = []
            scrolling_dir_acc = []
            
            # Calculate metrics for each scrolling window step
            for i in range(0, len(results['predictions']) - window_size + 1):
                window_preds = results['predictions'][i:i+window_size]
                window_actuals = results['actuals'][i:i+window_size]
                
                # Only calculate if we have valid data
                valid_mask = ~np.isnan(window_preds) & ~np.isnan(window_actuals)
                if np.sum(valid_mask) > 0:
                    window_mse = np.mean((window_preds[valid_mask] - window_actuals[valid_mask]) ** 2)
                    window_dir_acc = np.mean(np.sign(window_preds[valid_mask]) == np.sign(window_actuals[valid_mask])) * 100
                    
                    scrolling_mse.append(window_mse)
                    scrolling_dir_acc.append(window_dir_acc)
            
            timesteps = range(len(scrolling_mse))
            
            plt.subplot(2, 1, 1)
            plt.plot(timesteps, scrolling_mse, 'b-', linewidth=2)
            plt.title('Scrolling MSE Over Time')
            plt.xlabel('Scrolling Window Steps')
            plt.ylabel('MSE')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 1, 2)
            plt.plot(timesteps, scrolling_dir_acc, 'r-', linewidth=2)
            plt.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='Random Guess')
            plt.title('Scrolling Directional Accuracy Over Time')
            plt.xlabel('Scrolling Window Steps')
            plt.ylabel('Directional Accuracy (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('scrolling_window_accuracy_over_time.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("⚠️  Skipping accuracy over time plot due to evaluation failure or insufficient data")
    
    print("\n✅ Growing window training and evaluation completed!")
    print(f"📁 Saved plots:")
    print(f"   • 'training_confidence_intervals.png' - Training confidence intervals")
    print(f"   • 'scrolling_window_immediate_predictions.png' - Immediate predictions")
    print(f"   • 'scrolling_window_general_predictions.png' - General predictions with CI")
    print(f"   • 'scrolling_window_accuracy_over_time.png' - Accuracy over time")
    print(f"📊 Model files saved:")
    print(f"   • 'mambastock.pth' - Final trained model")
    print(f"   • 'mambastock.pth' - Intermediate training saves (every 100 batches)")
    print(f"🎯 Growing window approach implemented!")
    print(f"   • Window grows from {seq_len} to 300 minutes")
    print(f"   • Uses all available historical data for realistic predictions")
    print(f"   • Single-horizon predictions with advanced normalization") 