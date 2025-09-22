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

# --- CONFIG ---
# MEMORY OPTIMIZATION SETTINGS (adjust these if you still get OOM errors):
MEMORY_SAVING_MODE = True  # Set to False to use original settings
BATCH_SIZE = 1 if MEMORY_SAVING_MODE else 8  # Further reduced for memory
MONTE_CARLO_SAMPLES = 15 if MEMORY_SAVING_MODE else 20  # Further reduced for memory
SEQUENCE_LENGTH = 30 if MEMORY_SAVING_MODE else 30  # Further reduced for memory
PREDICTION_LENGTH = 20  # Target window: model predicts 1 value representing 20-minute trend
STOCK_BATCH_SIZE = 100 if MEMORY_SAVING_MODE else 100  # Number of stocks per batch
EPOCHS = 5 
LEARNING_RATE = 1e-4  # Lower learning rate for fine-tuning from pretrained model

# IMPROVEMENTS FOR BETTER ACCURACY:
# 1. Increased uncertainty_scale max from 5.0 to 10.0
# 2. Added input noise (std=0.01) during training
# 3. Simplified loss focusing on accuracy with smoothening regularization
# 4. Loss weights: alpha=0.9 (accuracy), beta=0.1 (smoothening regularization)

h5_path = "/home/ubuntu/Moldova/h5_rty_data_processed.h5"  # <-- Preprocessed HDF5 file path
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

# --- MODEL INIT ---
model = MambaStock(input_size=13, seq_len=seq_len, pred_len=1).to(device)  # Model predicts 1 value, but target is 20-minute window

# Add learnable uncertainty scaling parameter
uncertainty_scale = torch.nn.Parameter(torch.tensor(1.96, device=device))
model.register_parameter('uncertainty_scale', uncertainty_scale)
try:
    state_dict = torch.load("mambastock_scrolling_two.pth", map_location=device)
    model.load_state_dict(state_dict)
    print("Loaded weights from mambastock_scrolling_two.pth")
except FileNotFoundError:
    print("mambastock_scrolling_two.pth not found, starting with random weights")
except Exception as e:
    print(f"Error loading mambastock_scrolling_two.pth: {e}")

# --- LOSS FUNCTION ---
loss_fn = nn.MSELoss().to(device)

# --- OPTIMIZER ---
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

# Verify model is ready for training
def verify_model_training_setup(model, device):
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
        test_input = torch.randn(2, seq_len, 13).to(device)
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

# Verify model setup
verify_model_training_setup(model, device)

# --- SCROLLING WINDOW TRAINING FUNCTIONS ---
def general_prediction_with_confidence(model, input_seq, remaining_minutes, device, n_samples=MONTE_CARLO_SAMPLES):
    """
    Make a general prediction for the next remaining_minutes with 95% confidence intervals.
    
    Args:
        model: The trained model
        input_seq: Current input sequence [seq_len, features]
        remaining_minutes: Number of minutes to predict (390 - seq_len)
        device: Device to run on
        n_samples: Number of Monte Carlo samples for uncertainty estimation
    
    Returns:
        predictions: Mean predictions for each minute
        ci_lower: Lower confidence interval bounds
        ci_upper: Upper confidence interval bounds
        pred_stds: Standard deviations of predictions
    """
    model.eval()
    
    # Store original dropout states
    original_dropout_states = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            original_dropout_states[name] = module.training
    
    predictions = []
    ci_lower = []
    ci_upper = []
    pred_stds = []
    
    current_seq = input_seq.clone().to(device)
    
    for minute in range(remaining_minutes):
        # Monte Carlo dropout for uncertainty estimation
        pred_samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                # Enable dropout for uncertainty estimation
                for name, module in model.named_modules():
                    if isinstance(module, nn.Dropout):
                        module.train()
                
                # Normalize current sequence
                mean = current_seq.mean(dim=0, keepdim=True)
                std = current_seq.std(dim=0, keepdim=True)
                std = torch.where(std > 1e-8, std, torch.ones_like(std))
                current_seq_norm = (current_seq - mean) / std
                
                # Make prediction
                pred_sample = model(current_seq_norm.unsqueeze(0))  # [1, pred_len]
                pred_sample = pred_sample.squeeze().detach().cpu().numpy()
                pred_samples.append(pred_sample)
        
        pred_samples = np.array(pred_samples)  # [n_samples]
        
        # Calculate mean prediction and confidence intervals
        pred_mean = np.mean(pred_samples)
        pred_std = np.std(pred_samples)
        
        # Add small epsilon to prevent zero std
        pred_std = pred_std + 1e-6
        
        # Use learnable uncertainty scaling for confidence intervals
        uncertainty_scale = 2.0  # Use a reasonable default for evaluation
        ci_lower_val = pred_mean - uncertainty_scale * pred_std
        ci_upper_val = pred_mean + uncertainty_scale * pred_std
        
        predictions.append(pred_mean)
        ci_lower.append(ci_lower_val)
        ci_upper.append(ci_upper_val)
        pred_stds.append(pred_std)
        
        # Update sequence for next prediction (use predicted value)
        next_pred = torch.zeros_like(current_seq[0])
        next_pred[10] = torch.tensor(pred_mean, dtype=next_pred.dtype, device=next_pred.device)
        current_seq = torch.cat([current_seq, next_pred.unsqueeze(0)], dim=0)
        current_seq = current_seq[-seq_len:, :]  # Keep only last seq_len elements
    
    # Restore original dropout states
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.train(original_dropout_states[name])
    
    return np.array(predictions), np.array(ci_lower), np.array(ci_upper), np.array(pred_stds)

def scrolling_window_training_step_with_ci(model, dataset, current_seq, current_seq_len, step_idx, 
                                         optimizer, loss_fn, device, pred_len=20, n_samples=MONTE_CARLO_SAMPLES):
    """
    Perform one step of scrolling window training with confidence intervals.
    
    Args:
        model: The model to train
        dataset: The dataset
        current_seq: Current sequence [n_stocks, current_seq_len, features]
        current_seq_len: Current sequence length
        step_idx: Current step index
        optimizer: The optimizer
        loss_fn: Loss function
        device: Device to run on
        pred_len: Number of minutes to predict
        n_samples: Number of Monte Carlo samples for uncertainty estimation
    
    Returns:
        loss: Training loss for this step
        updated_seq: Updated sequence with new minute appended
        updated_seq_len: Updated sequence length
        ci_stats: Confidence interval statistics
    """
    model.train()
    n_stocks = dataset.n_stocks
    
    # Normalize current sequence per stock with gradients enabled
    input_norm = torch.zeros_like(current_seq)
    for stock_idx in range(n_stocks):
        window = current_seq[stock_idx]  # [current_seq_len, features]
        mean = window.mean(dim=0, keepdim=True)  # [1, features]
        std = window.std(dim=0, keepdim=True)
        std = torch.where(std > 1e-8, std, torch.ones_like(std))
        input_norm[stock_idx] = (window - mean) / std
    
    # Ensure input has gradients for training
    input_norm.requires_grad_(True)
    
    # Add noise to inputs during training to increase uncertainty
    noise_std = 0.01
    input_norm = input_norm + torch.randn_like(input_norm) * noise_std
    
    # Verify input properties (only every 100 steps)
    if step_idx % 100 == 0:
        print(f"🔍 Input Check - Step {step_idx}:")
        print(f"   Input shape: {input_norm.shape}")
        print(f"   Input device: {input_norm.device}")
        print(f"   Input dtype: {input_norm.dtype}")
        print(f"   Input requires grad: {input_norm.requires_grad}")
        print(f"   Input has NaN: {torch.isnan(input_norm).any()}")
        print(f"   Input has Inf: {torch.isinf(input_norm).any()}")
    
    # Make predictions with Monte Carlo dropout for uncertainty estimation
    pred_samples = []
    ci_lower = []
    ci_upper = []
    pred_stds = []
    
    # Enable dropout for uncertainty estimation
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    
    # Monte Carlo dropout for uncertainty estimation (without gradients)
    pred_samples = []
    
    # Store original dropout states
    original_dropout_states = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            original_dropout_states[name] = module.training
    
    with torch.no_grad():
        for i in range(n_samples):
            # Enable dropout for uncertainty estimation
            for name, module in model.named_modules():
                if isinstance(module, nn.Dropout):
                    module.train()
            
            # Make prediction with dropout enabled
            pred_sample = model(input_norm)  # [n_stocks, 1, 1]
            pred_sample = pred_sample.squeeze(-1)  # [n_stocks, 1] - remove last dimension
            pred_samples.append(pred_sample.detach())
            
            # Debug: Check if predictions are actually different (only every 100 steps)
            if step_idx % 100 == 0 and i < 5:  # Only print first 5 samples every 100 steps
                print(f"   Monte Carlo Sample {i}: {pred_sample[0,0].item():.6f}")
        
        # Restore original dropout states
        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                module.train(original_dropout_states[name])
        
        # Calculate mean predictions and confidence intervals
        pred_samples = torch.stack(pred_samples)  # [n_samples, n_stocks, 1]
        pred_mean = pred_samples.mean(dim=0)  # [n_stocks, 1]
        pred_std = pred_samples.std(dim=0)  # [n_stocks, 1]
        
        # Debug: Check if Monte Carlo samples are actually different (only every 100 steps)
        if step_idx % 100 == 0:
            print(f"🔍 Monte Carlo Debug - Step {step_idx}:")
            print(f"   Number of samples: {pred_samples.shape[0]}")
            print(f"   Sample shape: {pred_samples.shape}")
            print(f"   All samples identical: {torch.all(pred_samples == pred_samples[0])}")
            print(f"   Max difference between samples: {(pred_samples.max() - pred_samples.min()).item():.6f}")
            print(f"   Sample 0 vs Sample 1: {pred_samples[0,0,0].item():.6f} vs {pred_samples[1,0,0].item():.6f}")
        
        # Add small epsilon to prevent zero std
        pred_std = pred_std + 1e-6
    
    # Use learnable uncertainty scaling from model - increased max for better coverage
    uncertainty_scale = torch.clamp(model.uncertainty_scale, min=1.0, max=10.0)
    
    ci_lower_val = pred_mean - uncertainty_scale * pred_std
    ci_upper_val = pred_mean + uncertainty_scale * pred_std
    ci_width = ci_upper_val - ci_lower_val
    
    # Get actual values for the predicted period (average of next 20 minutes)
    actuals = []
    for stock_idx in range(n_stocks):
        # Get the next 20 minutes and calculate their average
        next_20_minutes = dataset.data[stock_idx, seq_len+step_idx:seq_len+step_idx+20, 10]  # [20] - next 20 minutes
        actual_avg = next_20_minutes.mean()  # Single value representing 20-minute trend
        actuals.append(actual_avg)
    actuals = torch.stack(actuals).to(device)  # [n_stocks]
    actuals = actuals.unsqueeze(-1)  # [n_stocks, 1] - add dimension to match train_pred
    
    # Make a separate prediction for training (with gradients)
    model.train()
    train_pred = model(input_norm)  # [n_stocks, 1, 1]
    train_pred = train_pred.squeeze(-1)  # [n_stocks, 1] - remove last dimension
    
    # Debug: Check if predictions are reasonable (only every 100 steps)
    if step_idx % 100 == 0:
        print(f"🔍 Prediction Debug - Step {step_idx}:")
        print(f"   Train pred range: [{train_pred.min().item():.6f}, {train_pred.max().item():.6f}]")
        print(f"   Actuals range: [{actuals.min().item():.6f}, {actuals.max().item():.6f}]")
        print(f"   Train pred mean: {train_pred.mean().item():.6f}")
        print(f"   Actuals mean: {actuals.mean().item():.6f}")
        print(f"   Train pred std: {train_pred.std().item():.6f}")
        print(f"   Actuals std: {actuals.std().item():.6f}")
        print(f"   Monte Carlo pred_std range: [{pred_std.min().item():.6f}, {pred_std.max().item():.6f}]")
        print(f"   Monte Carlo pred_std mean: {pred_std.mean().item():.6f}")
        print(f"   CI width range: [{ci_width.min().item():.6f}, {ci_width.max().item():.6f}]")
        print(f"   CI width mean: {ci_width.mean().item():.6f}")
        
        # Debug: Check dropout states and model mode
        print(f"🔍 Dropout Debug - Step {step_idx}:")
        dropout_count = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                dropout_count += 1
                print(f"   {name}: training={module.training}, p={module.p}")
        print(f"   Total dropout layers: {dropout_count}")
        print(f"   Model training mode: {model.training}")
    
    # Apply temperature scaling for better calibration
    # Temperature scaling helps calibrate the model's uncertainty
    temperature = torch.clamp(model.uncertainty_scale, min=0.5, max=2.0)
    train_pred = train_pred / temperature
    
    # Calculate coverage mask for calibration loss
    coverage_mask = (actuals >= ci_lower_val.to(device)) & (actuals <= ci_upper_val.to(device))
    
    # Compute calibration-aware loss
    # 1. Prediction accuracy loss
    accuracy_loss = loss_fn(train_pred, actuals)
    
    # 2. Calibration loss - encourage better coverage
    # Calculate how well the confidence intervals cover the actual values
    coverage_loss = torch.mean(torch.relu(ci_lower_val.to(device) - actuals) + 
                              torch.relu(actuals - ci_upper_val.to(device)))
    
    # 3. Uncertainty calibration loss - encourage appropriate uncertainty
    # Penalize if confidence intervals are too narrow or too wide
    target_coverage = 0.95  # 95% target coverage
    current_coverage = coverage_mask.float().mean()
    calibration_loss = torch.abs(current_coverage - target_coverage)
    
    # Smoothening regularization - penalize large changes between consecutive predictions
    if hasattr(model, '_previous_predictions') and model._previous_predictions is not None:
        # Check if batch sizes match (same number of stocks)
        if train_pred.shape == model._previous_predictions.shape:
            prediction_diff = torch.abs(train_pred - model._previous_predictions)
            smoothening_penalty = torch.mean(prediction_diff)
        else:
            smoothening_penalty = torch.tensor(0.0, device=device)
    else:
        smoothening_penalty = torch.tensor(0.0, device=device)

    # Anti-collapse regularization - discourage constant/near-constant predictions
    # Penalize too-low variance in predictions across stocks
    pred_std_across_stocks = torch.clamp(train_pred.std(), min=0.0)
    anti_collapse_penalty = torch.relu(torch.tensor(1e-3, device=device) - pred_std_across_stocks)

    # Store current predictions for next iteration
    model._previous_predictions = train_pred.detach().clone()

    # Multi-objective loss: accuracy + coverage + calibration + smoothening + anti-collapse
    alpha = 0.6   # accuracy
    beta = 0.2    # coverage
    gamma = 0.1   # calibration
    delta = 0.08  # smoothening
    zeta = 0.02   # anti-collapse

    loss = (
        alpha * accuracy_loss
        + beta * coverage_loss
        + gamma * calibration_loss
        + delta * smoothening_penalty
        + zeta * anti_collapse_penalty
    )
    
    # Ensure loss is defined even if there are issues
    if 'loss' not in locals():
        loss = accuracy_loss
    
    # Only log when model saves (every 100 batches) to reduce verbosity
    if step_idx % 100 == 0:
        print(f"📊 Training Metrics - Step {step_idx}:")
        print(f"   Accuracy Loss: {accuracy_loss.item():.6f}")
        print(f"   Coverage Loss: {coverage_loss.item():.6f}")
        print(f"   Calibration Loss: {calibration_loss.item():.6f}")
        print(f"   Smoothening Penalty: {smoothening_penalty.item():.6f}")
        print(f"   Anti-collapse Penalty: {anti_collapse_penalty.item():.6f}")
        print(f"   Combined Loss: {loss.item():.6f}")
        print(f"   Current Coverage: {current_coverage.item():.2%}")
        print(f"   Uncertainty Scale: {uncertainty_scale.item():.3f}")
        print(f"   Temperature: {temperature.item():.3f}")
        print(f"   Loss Weights - α(acc): {alpha}, β(cov): {beta}, γ(cal): {gamma}, δ(smooth): {delta}, ζ(anti-collapse): {zeta}")
    
    # Comprehensive gradient checks (only every 100 steps)
    if step_idx % 100 == 0:
        print(f"🔍 Gradient Check - Step {step_idx}:")
        print(f"   Input requires grad: {input_norm.requires_grad}")
        print(f"   Train pred requires grad: {train_pred.requires_grad}")
        print(f"   Actuals requires grad: {actuals.requires_grad}")
        print(f"   Loss requires grad: {loss.requires_grad}")
        print(f"   Loss value: {loss.item():.6f}")
        
        # Check model parameters have gradients
        model_has_grads = any(p.requires_grad for p in model.parameters())
        print(f"   Model has trainable params: {model_has_grads}")
    
    if not loss.requires_grad:
        print(f"⚠️  WARNING: Loss doesn't require grad!")
        return None, current_seq, current_seq_len, None
    
    # Backward pass with memory optimization
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients after backward pass
    total_grad_norm = 0
    grad_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2)
            total_grad_norm += grad_norm.item() ** 2
            grad_count += 1
            if step_idx % 100 == 0:  # Only print every 100 steps to reduce output
                print(f"   {name}: grad_norm = {grad_norm.item():.6f}")
    
    if grad_count > 0:
        total_grad_norm = total_grad_norm ** (1. / 2)
        if step_idx % 100 == 0:  # Only print every 100 steps
            print(f"   Total gradient norm: {total_grad_norm:.6f}")
            print(f"   Parameters with gradients: {grad_count}")
    else:
        print(f"⚠️  WARNING: No gradients computed!")
        return None, current_seq, current_seq_len, None
    
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    
    # Clear gradients to free memory
    for param in model.parameters():
        if param.grad is not None:
            param.grad.detach_()
            param.grad = None
    
    # Calculate confidence interval statistics
    avg_ci_width = ci_width.mean().item()
    avg_pred_std = pred_std.mean().item()
    
    # Calculate coverage rate (how many actual values fall within confidence intervals)
    coverage_rate = coverage_mask.float().mean().item() * 100
    
    ci_stats = {
        'avg_ci_width': avg_ci_width,
        'avg_pred_std': avg_pred_std,
        'coverage_rate': coverage_rate,
        'pred_mean': pred_mean,
        'pred_std': pred_std,
        'ci_lower': ci_lower_val,
        'ci_upper': ci_upper_val
    }
    
    # Append the actual data for the next minute to the sequence
    next_minute_data = []
    for stock_idx in range(n_stocks):
        next_data = dataset.data[stock_idx, seq_len+step_idx:seq_len+step_idx+1, :]  # [1, features]
        next_minute_data.append(next_data)
    next_minute_data = torch.stack(next_minute_data).to(device)  # [n_stocks, 1, features]
    
    # Update sequence by appending the new minute
    updated_seq = torch.cat([current_seq, next_minute_data], dim=1)  # [n_stocks, current_seq_len+1, features]
    updated_seq_len = current_seq_len + 1
    
    # Keep only the last seq_len minutes to prevent memory issues
    if updated_seq_len > seq_len:
        updated_seq = updated_seq[:, -seq_len:, :]
        updated_seq_len = seq_len
    
    return loss.item(), updated_seq, updated_seq_len, ci_stats

def scrolling_window_training_step_batched(model, dataset, current_seq, current_seq_len, step_idx, 
                                         optimizer, loss_fn, device, pred_len=20, n_samples=MONTE_CARLO_SAMPLES,
                                         stock_batch_size=50, global_batch_counter=None):
    """
    Perform one step of scrolling window training with batched stock processing.
    
    Args:
        model: The model to train
        dataset: The dataset
        current_seq: Current sequence [n_stocks, current_seq_len, features]
        current_seq_len: Current sequence length
        step_idx: Current step index
        optimizer: The optimizer
        loss_fn: Loss function
        device: Device to run on
        pred_len: Number of minutes to predict
        n_samples: Number of Monte Carlo samples for uncertainty estimation
        stock_batch_size: Number of stocks to process in each batch
    
    Returns:
        loss: Training loss for this step
        updated_seq: Updated sequence with new minute appended
        updated_seq_len: Updated sequence length
        ci_stats: Confidence interval statistics
    """
    model.train()
    n_stocks = dataset.n_stocks
    
    # Calculate number of stock batches
    num_stock_batches = (n_stocks + stock_batch_size - 1) // stock_batch_size
    
    print(f"🔄 Processing {n_stocks} stocks in {num_stock_batches} batches of {stock_batch_size}")
    
    total_loss = 0
    all_ci_widths = []
    all_pred_stds = []
    all_coverage_rates = []
    
    # Process stocks in batches
    for batch_idx in range(num_stock_batches):
        start_stock = batch_idx * stock_batch_size
        end_stock = min((batch_idx + 1) * stock_batch_size, n_stocks)
        batch_stocks = list(range(start_stock, end_stock))
        
        print(f"📊 Processing stock batch {batch_idx + 1}/{num_stock_batches} (stocks {start_stock}-{end_stock-1})")
        
        # Update global batch counter
        if global_batch_counter is not None:
            global_batch_counter[0] += 1
            current_batch = global_batch_counter[0]
            
            # Save model every 100 batches
            if current_batch % 100 == 0:
                torch.save(model.state_dict(), "mambastock_scrolling_two.pth")
                print(f"💾 Model saved at global batch {current_batch}")
                print(f"📁 Saved to: mambastock_scrolling_two.pth (overwriting)")
                
                # Verify the save worked
                try:
                    test_load = torch.load("mambastock_scrolling_two.pth", map_location=device)
                    print(f"✅ Save verification successful - file size: {len(test_load)} parameters")
                except Exception as e:
                    print(f"❌ Save verification failed: {e}")
        
        # Extract batch data
        batch_seq = current_seq[batch_stocks].clone()  # [batch_size, seq_len, features]
        
        # Normalize batch sequence
        input_norm = torch.zeros_like(batch_seq)
        for i, stock_idx in enumerate(batch_stocks):
            window = batch_seq[i]  # [seq_len, features]
            mean = window.mean(dim=0, keepdim=True)
            std = window.std(dim=0, keepdim=True)
            std = torch.where(std > 1e-8, std, torch.ones_like(std))
            input_norm[i] = (window - mean) / std
        
        input_norm.requires_grad_(True)
        
        # Add noise for uncertainty
        noise_std = 0.01
        input_norm = input_norm + torch.randn_like(input_norm) * noise_std
        
        # Monte Carlo dropout for uncertainty estimation
        pred_samples = []
        original_dropout_states = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                original_dropout_states[name] = module.training
        
        with torch.no_grad():
            for i in range(n_samples):
                # Enable dropout for uncertainty estimation
                for name, module in model.named_modules():
                    if isinstance(module, nn.Dropout):
                        module.train()
                
                # Make prediction
                pred_sample = model(input_norm)  # [batch_size, 1, 1]
                pred_sample = pred_sample.squeeze(-1)  # [batch_size, 1] - remove last dimension
                pred_samples.append(pred_sample.detach())
            
            # Restore dropout states
            for name, module in model.named_modules():
                if isinstance(module, nn.Dropout):
                    module.train(original_dropout_states[name])
            
            # Calculate statistics
            pred_samples = torch.stack(pred_samples)  # [n_samples, batch_size, 1]
            pred_mean = pred_samples.mean(dim=0)  # [batch_size, 1]
            pred_std = pred_samples.std(dim=0)  # [batch_size, 1]
            pred_std = pred_std + 1e-6  # Add epsilon
        
        # Get actual values for this batch (average of next 20 minutes)
        actuals = []
        for stock_idx in batch_stocks:
            # Get the next 20 minutes and calculate their average
            next_20_minutes = dataset.data[stock_idx, seq_len+step_idx:seq_len+step_idx+20, 10]  # [20] - next 20 minutes
            actual_avg = next_20_minutes.mean()  # Single value representing 20-minute trend
            actuals.append(actual_avg)
        actuals = torch.stack(actuals).to(device)  # [batch_size]
        actuals = actuals.unsqueeze(-1)  # [batch_size, 1] - add dimension to match train_pred
        
        # Make training prediction
        model.train()
        train_pred = model(input_norm)  # [batch_size, 1, 1]
        train_pred = train_pred.squeeze(-1)  # [batch_size, 1] - remove last dimension
        
        # Calculate confidence intervals
        uncertainty_scale = torch.clamp(model.uncertainty_scale, min=1.0, max=10.0)
        ci_lower_val = pred_mean - uncertainty_scale * pred_std
        ci_upper_val = pred_mean + uncertainty_scale * pred_std
        
        # Calculate coverage
        coverage_mask = (actuals >= ci_lower_val.to(device)) & (actuals <= ci_upper_val.to(device))
        coverage_rate = coverage_mask.float().mean().item() * 100
        # Coverage and calibration losses
        coverage_loss = torch.mean(
            torch.relu(ci_lower_val.to(device) - actuals) + torch.relu(actuals - ci_upper_val.to(device))
        )
        current_coverage = coverage_mask.float().mean()
        calibration_loss = torch.abs(current_coverage - 0.95)
        
        # Calculate primary loss (accuracy)
        accuracy_loss = loss_fn(train_pred, actuals)
        
        # Smoothening regularization - penalize large changes between consecutive predictions
        if hasattr(model, '_previous_predictions') and model._previous_predictions is not None:
            # Check if batch sizes match (same number of stocks)
            if train_pred.shape == model._previous_predictions.shape:
                # Calculate the difference between current and previous predictions
                prediction_diff = torch.abs(train_pred - model._previous_predictions)
                smoothening_penalty = torch.mean(prediction_diff)
            else:
                # Batch sizes don't match, skip smoothening for this batch
                smoothening_penalty = torch.tensor(0.0, device=device)
        else:
            # No previous predictions available, use zero penalty
            smoothening_penalty = torch.tensor(0.0, device=device)
        
        # Anti-collapse regularization - discourage constant predictions within batch
        pred_std_across_batch = torch.clamp(train_pred.std(), min=0.0)
        anti_collapse_penalty = torch.relu(torch.tensor(1e-3, device=device) - pred_std_across_batch)

        # Store current predictions for next iteration
        model._previous_predictions = train_pred.detach().clone()
        
        # Multi-objective loss (hybrid): accuracy + coverage + calibration + smoothening + anti-collapse
        alpha, beta, gamma, delta, zeta = 0.6, 0.2, 0.1, 0.08, 0.02
        batch_loss = (
            alpha * accuracy_loss
            + beta * coverage_loss
            + gamma * calibration_loss
            + delta * smoothening_penalty
            + zeta * anti_collapse_penalty
        )
        
        # Ensure batch_loss is defined even if there are issues
        if 'batch_loss' not in locals():
            batch_loss = accuracy_loss
        
        # Backward pass for this batch
        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        # Clear gradients and cache
        for param in model.parameters():
            if param.grad is not None:
                param.grad.detach_()
                param.grad = None
        
        # Store batch statistics
        total_loss += batch_loss.item()
        all_ci_widths.append((ci_upper_val - ci_lower_val).mean().item())
        all_pred_stds.append(pred_std.mean().item())
        all_coverage_rates.append(coverage_rate)
        
        # Store batch statistics before cleanup
        batch_loss_value = batch_loss.item()
        
        # Clear batch data from memory
        del batch_seq, input_norm, pred_samples, pred_mean, pred_std
        del actuals, train_pred, ci_lower_val, ci_upper_val, coverage_mask
        del batch_loss, accuracy_loss, smoothening_penalty, anti_collapse_penalty
        torch.cuda.empty_cache()
        
        print(f"✅ Batch {batch_idx + 1} completed. Loss: {batch_loss_value:.6f}, Coverage: {coverage_rate:.2f}%")
    
    # Calculate overall statistics
    avg_loss = total_loss / num_stock_batches
    avg_ci_width = np.mean(all_ci_widths)
    avg_pred_std = np.mean(all_pred_stds)
    avg_coverage = np.mean(all_coverage_rates)
    
    ci_stats = {
        'avg_ci_width': avg_ci_width,
        'avg_pred_std': avg_pred_std,
        'coverage_rate': avg_coverage
    }
    
    # Update sequence with next minute data (process all stocks)
    next_minute_data = []
    for stock_idx in range(n_stocks):
        next_data = dataset.data[stock_idx, seq_len+step_idx:seq_len+step_idx+1, :]  # [1, features]
        next_minute_data.append(next_data)
    next_minute_data = torch.stack(next_minute_data).to(device)  # [n_stocks, 1, features]
    
    # Update sequence
    updated_seq = torch.cat([current_seq, next_minute_data], dim=1)  # [n_stocks, current_seq_len+1, features]
    updated_seq_len = current_seq_len + 1
    
    # Keep only last seq_len minutes
    if updated_seq_len > seq_len:
        updated_seq = updated_seq[:, -seq_len:, :]
        updated_seq_len = seq_len
    
    return avg_loss, updated_seq, updated_seq_len, ci_stats

def train_scrolling_window(dataset, model, optimizer, loss_fn, device, epochs=1, 
                         seq_len=30, pred_len=20, end_time=360, n_samples=MONTE_CARLO_SAMPLES,
                         use_batched_training=True, stock_batch_size=50):  # 360 = 3:30 PM (6 hours from 9:30 AM)
    """
    Train the model using scrolling window approach with confidence intervals.
    
    Args:
        dataset: The dataset
        model: The model to train
        optimizer: The optimizer
        loss_fn: Loss function
        device: Device to run on
        epochs: Number of epochs
        seq_len: Initial sequence length
        pred_len: Number of minutes to predict each step
        end_time: End time in minutes (3:30 PM = 360 minutes from 9:30 AM)
        n_samples: Number of Monte Carlo samples for uncertainty estimation
        use_batched_training: Whether to use batched stock processing
        stock_batch_size: Number of stocks to process in each batch
    """
    # CPU-specific optimizations
    if device.type == 'cpu':
        # Keep more Monte Carlo samples for better uncertainty estimation
        if n_samples < MONTE_CARLO_SAMPLES:
            print(f"⚠️  Increasing Monte Carlo samples to {MONTE_CARLO_SAMPLES} for better uncertainty estimation")
            n_samples = MONTE_CARLO_SAMPLES
    n_stocks = dataset.n_stocks
    total_steps = dataset.day_length
    
    # Calculate maximum scroll steps (until 3:30 PM)
    max_scroll_steps = min(end_time - seq_len, total_steps - seq_len - pred_len)
    
    print(f"Scrolling window training: seq_len={seq_len}, pred_len={pred_len}")
    print(f"Total available steps: {total_steps}, Max scroll steps: {max_scroll_steps}")
    print(f"Training until: {end_time} minutes (3:30 PM)")
    print(f"Using {n_samples} Monte Carlo samples for confidence intervals (REDUCED for memory)")
    print(f"Batched training: {'✅ Enabled' if use_batched_training else '❌ Disabled'}")
    if use_batched_training:
        print(f"Stock batch size: {stock_batch_size}")
    
    # Store confidence interval statistics during training
    all_ci_widths = []
    all_pred_stds = []
    all_coverage_rates = []
    
    # Track total batches across all epochs
    total_batches = 0
    global_batch_counter = [0]  # Use list to allow modification inside function
    
    # Force create/overwrite mambastock_scrolling_two.pth at start
    torch.save(model.state_dict(), "mambastock_scrolling_two.pth")
    print(f"💾 Initial model saved to mambastock_scrolling_two.pth (overwriting if exists)")
    
    # Also save a backup with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_filename = f"mambastock_scrolling_two_backup_{timestamp}.pth"
    torch.save(model.state_dict(), backup_filename)
    print(f"💾 Backup saved to {backup_filename}")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Initialize smoothening for this epoch
        model._previous_predictions = None
        
        # Initialize with first seq_len minutes
        current_seq = dataset.data[:, :seq_len, :].clone().to(device)  # [n_stocks, seq_len, features]
        current_seq_len = seq_len
        
        total_loss = 0
        step_count = 0
        epoch_ci_widths = []
        epoch_pred_stds = []
        epoch_coverage_rates = []
        
        for step_idx in range(max_scroll_steps):
            # Show progress for CPU training
            if device.type == 'cpu' and step_idx % 5 == 0:
                print(f"CPU Training Progress: {step_idx}/{max_scroll_steps} ({step_idx/max_scroll_steps*100:.1f}%)")
            
            # Perform training step with confidence intervals
            if use_batched_training:
                result = scrolling_window_training_step_batched(
                    model, dataset, current_seq, current_seq_len, step_idx,
                    optimizer, loss_fn, device, pred_len, n_samples, stock_batch_size, global_batch_counter
                )
            else:
                result = scrolling_window_training_step_with_ci(
                    model, dataset, current_seq, current_seq_len, step_idx,
                    optimizer, loss_fn, device, pred_len, n_samples
                )
            
            # Check if training step failed
            if result[0] is None:
                print(f"⚠️  Training step {step_idx} failed, skipping...")
                continue
                
            loss, current_seq, current_seq_len, ci_stats = result
            
            total_loss += loss
            step_count += 1
            
            # Store confidence interval statistics
            if ci_stats is not None:
                epoch_ci_widths.append(ci_stats['avg_ci_width'])
                epoch_pred_stds.append(ci_stats['avg_pred_std'])
                epoch_coverage_rates.append(ci_stats['coverage_rate'])
            
            # Log progress every 10 steps
            if step_count % 10 == 0:
                avg_loss = total_loss / step_count
                avg_ci_width = np.mean(epoch_ci_widths[-10:]) if epoch_ci_widths else 0
                avg_pred_std = np.mean(epoch_pred_stds[-10:]) if epoch_pred_stds else 0
                avg_coverage = np.mean(epoch_coverage_rates[-10:]) if epoch_coverage_rates else 0
                
                print(f"Step {step_count}/{max_scroll_steps}, Avg Loss: {avg_loss:.6f}")
                print(f"  Avg CI Width: {avg_ci_width:.6f}, Avg Pred Std: {avg_pred_std:.6f}, Coverage: {avg_coverage:.2f}%")
                logger.info(f"Epoch {epoch+1}, Step {step_count}/{max_scroll_steps}, Avg Loss: {avg_loss:.6f}")
                logger.info(f"  Avg CI Width: {avg_ci_width:.6f}, Avg Pred Std: {avg_pred_std:.6f}, Coverage: {avg_coverage:.2f}%")
                
                # Memory monitoring and cleanup
                monitor_memory_usage()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif device.type == 'cpu':
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
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
        all_ci_widths.extend(epoch_ci_widths)
        all_pred_stds.extend(epoch_pred_stds)
        all_coverage_rates.extend(epoch_coverage_rates)
        
        # Final save for this epoch
        torch.save(model.state_dict(), "mambastock_scrolling_two.pth")
        
        if step_count > 0:
            avg_total_loss = total_loss / step_count
            avg_epoch_ci_width = np.mean(epoch_ci_widths) if epoch_ci_widths else 0
            avg_epoch_pred_std = np.mean(epoch_pred_stds) if epoch_pred_stds else 0
            avg_epoch_coverage = np.mean(epoch_coverage_rates) if epoch_coverage_rates else 0
            
            print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_total_loss:.6f} (Total: {total_loss:.2f} across {step_count:,} steps)")
            print(f"  Epoch Avg CI Width: {avg_epoch_ci_width:.6f}, Avg Pred Std: {avg_epoch_pred_std:.6f}, Coverage: {avg_epoch_coverage:.2f}%")
            force_log_write(f"{epoch+1} | Avg: {avg_total_loss:.6f} | Total: {total_loss:.2f} | Steps: {step_count}")
            force_log_write(f"  CI Width: {avg_epoch_ci_width:.6f} | Pred Std: {avg_epoch_pred_std:.6f} | Coverage: {avg_epoch_coverage:.2f}%")
        else:
            print(f"Epoch {epoch+1}/{epochs} - No valid steps processed")
            force_log_write(f"{epoch+1} | No valid steps")
    
    # Return training statistics
    return {
        'ci_widths': all_ci_widths,
        'pred_stds': all_pred_stds,
        'coverage_rates': all_coverage_rates
    }

def evaluate_scrolling_window_with_general_predictions(dataset, model, seq_len=30, pred_len=20, 
                                                    device=None, end_time=360, n_samples=MONTE_CARLO_SAMPLES):
    """
    Evaluate the model using scrolling window approach with general predictions.
    
    Args:
        dataset: The dataset
        model: The trained model
        seq_len: Initial sequence length
        pred_len: Number of minutes to predict each step
        device: Device to run on
        end_time: End time in minutes (3:30 PM = 360 minutes from 9:30 AM)
        n_samples: Number of Monte Carlo samples for uncertainty estimation
    
    Returns:
        results: Dictionary containing evaluation results
    """
    model.eval()
    n_stocks = dataset.n_stocks
    total_steps = dataset.day_length
    
    # Calculate maximum scroll steps (until 3:30 PM)
    max_scroll_steps = min(end_time - seq_len, total_steps - seq_len - pred_len)
    
    print(f"Evaluating scrolling window: seq_len={seq_len}, pred_len={pred_len}")
    print(f"Total available steps: {total_steps}, Max scroll steps: {max_scroll_steps}")
    print(f"Evaluating until: {end_time} minutes (3:30 PM)")
    
    all_preds = []
    all_trues = []
    all_general_preds = []
    all_general_ci_lower = []
    all_general_ci_upper = []
    all_general_pred_stds = []
    all_timesteps = []
    
    # Initialize with first seq_len minutes
    current_seq = dataset.data[:, :seq_len, :].clone().to(device)  # [n_stocks, seq_len, features]
    current_seq_len = seq_len
    
    for step_idx in range(max_scroll_steps):
        # Make general prediction for remaining minutes
        remaining_minutes = 360 - current_seq_len  # 360 = 3:30 PM (6 hours from 9:30 AM)
        
        general_predictions = []
        general_ci_lower = []
        general_ci_upper = []
        general_pred_stds = []
        
        for stock_idx in range(n_stocks):
            stock_seq = current_seq[stock_idx]  # [current_seq_len, features]
            
            # Make general prediction with confidence intervals
            preds, ci_low, ci_up, pred_std = general_prediction_with_confidence(
                model, stock_seq, remaining_minutes, device, n_samples
            )
            
            general_predictions.append(preds)
            general_ci_lower.append(ci_low)
            general_ci_upper.append(ci_up)
            general_pred_stds.append(pred_std)
        
        # Normalize current sequence per stock for immediate predictions
        input_norm = torch.zeros_like(current_seq)
        for stock_idx in range(n_stocks):
            window = current_seq[stock_idx]  # [current_seq_len, features]
            mean = window.mean(dim=0, keepdim=True)  # [1, features]
            std = window.std(dim=0, keepdim=True)
            std = torch.where(std > 1e-8, std, torch.ones_like(std))
            input_norm[stock_idx] = (window - mean) / std
        
        # Make immediate predictions
        preds = model(input_norm)  # [n_stocks, 1, 1]
        preds = preds.squeeze(-1)  # [n_stocks, 1] - remove last dimension
        
        # Get actual values for the predicted period (average of next 20 minutes)
        actuals = []
        for stock_idx in range(n_stocks):
            # Get the next 20 minutes and calculate their average
            next_20_minutes = dataset.data[stock_idx, seq_len+step_idx:seq_len+step_idx+20, 10]  # [20] - next 20 minutes
            actual_avg = next_20_minutes.mean()  # Single value representing 20-minute trend
            actuals.append(actual_avg)
        actuals = torch.stack(actuals).to(device)  # [n_stocks]
        actuals = actuals.unsqueeze(-1)  # [n_stocks, 1] - add dimension to match train_pred
        
        # Store results
        all_preds.append(preds.detach().cpu())
        all_trues.append(actuals.cpu())
        all_general_preds.append(general_predictions)
        all_general_ci_lower.append(general_ci_lower)
        all_general_ci_upper.append(general_ci_upper)
        all_general_pred_stds.append(general_pred_stds)
        all_timesteps.append(step_idx)
        
        # Append the actual data for the next minute to the sequence
        next_minute_data = []
        for stock_idx in range(n_stocks):
            next_data = dataset.data[stock_idx, seq_len+step_idx:seq_len+step_idx+1, :]  # [1, features]
            next_minute_data.append(next_data)
        next_minute_data = torch.stack(next_minute_data).to(device)  # [n_stocks, 1, features]
        
        # Update sequence by appending the new minute
        current_seq = torch.cat([current_seq, next_minute_data], dim=1)  # [n_stocks, current_seq_len+1, features]
        current_seq_len += 1
        
        # Keep only the last seq_len minutes to prevent memory issues
        if current_seq_len > seq_len:
            current_seq = current_seq[:, -seq_len:, :]
            current_seq_len = seq_len
        
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
        'general_ci_lower': all_general_ci_lower,
        'general_ci_upper': all_general_ci_upper,
        'general_pred_stds': all_general_pred_stds,
        'timesteps': all_timesteps
    }

# --- TRAIN LOOP ---
if __name__ == "__main__":
    print("Starting scrolling window training...")
    
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
    
    # --- MODEL INIT ---
    model = MambaStock(input_size=13, seq_len=seq_len, pred_len=pred_len).to(device)
    
    # Add learnable uncertainty scaling parameter
    uncertainty_scale = torch.nn.Parameter(torch.tensor(1.96, device=device))
    model.register_parameter('uncertainty_scale', uncertainty_scale)
    try:
        state_dict = torch.load("mambastock_scrolling_two.pth", map_location=device)
        model.load_state_dict(state_dict)
        print("Loaded weights from mambastock_scrolling_two.pth")
    except FileNotFoundError:
        print("mambastock_scrolling_two.pth not found, starting with random weights")
    except Exception as e:
        print(f"Error loading mambastock_scrolling_two.pth: {e}")
    
    # --- LOSS FUNCTION ---
    loss_fn = nn.MSELoss().to(device)
    
    # --- OPTIMIZER ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Verify model is ready for training
    verify_model_training_setup(model, device)
    
    # Train the model using scrolling window approach with confidence intervals
    training_stats = train_scrolling_window(dataset, model, optimizer, loss_fn, device, 
                                         epochs=epochs, seq_len=seq_len, pred_len=pred_len, 
                                         end_time=360, n_samples=MONTE_CARLO_SAMPLES,
                                         use_batched_training=True, stock_batch_size=STOCK_BATCH_SIZE)
    
    # Save final model
    torch.save(model.state_dict(), "mambastock_scrolling_two.pth")
    logger.info("Final model saved to mambastock_scrolling_two.pth")
    
    # Plot training confidence interval statistics
    if training_stats and len(training_stats['ci_widths']) > 0:
        print("\nCreating training confidence interval plots...")
        
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Confidence interval width over training steps
        plt.subplot(1, 3, 1)
        plt.plot(training_stats['ci_widths'], 'b-', linewidth=2)
        plt.title('Confidence Interval Width During Training')
        plt.xlabel('Training Steps')
        plt.ylabel('Average CI Width')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Prediction standard deviation over training steps
        plt.subplot(1, 3, 2)
        plt.plot(training_stats['pred_stds'], 'r-', linewidth=2)
        plt.title('Prediction Standard Deviation During Training')
        plt.xlabel('Training Steps')
        plt.ylabel('Average Prediction Std')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Coverage rate over training steps
        plt.subplot(1, 3, 3)
        plt.plot(training_stats['coverage_rates'], 'g-', linewidth=2)
        plt.axhline(y=95, color='black', linestyle='--', alpha=0.7, label='Target 95% Coverage')
        plt.title('Coverage Rate During Training')
        plt.xlabel('Training Steps')
        plt.ylabel('Coverage Rate (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_confidence_intervals.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print final training statistics
        print(f"\nFinal Training Statistics:")
        print(f"  Average CI Width: {np.mean(training_stats['ci_widths']):.6f}")
        print(f"  Average Prediction Std: {np.mean(training_stats['pred_stds']):.6f}")
        print(f"  Average Coverage Rate: {np.mean(training_stats['coverage_rates']):.2f}%")
        print(f"  CI Width Range: [{np.min(training_stats['ci_widths']):.6f}, {np.max(training_stats['ci_widths']):.6f}]")
        print(f"  Prediction Std Range: [{np.min(training_stats['pred_stds']):.6f}, {np.max(training_stats['pred_stds']):.6f}]")
        print(f"  Coverage Rate Range: [{np.min(training_stats['coverage_rates']):.2f}%, {np.max(training_stats['coverage_rates']):.2f}%]")
    
    # Evaluate the model
    print("\nEvaluating model with general predictions...")
    try:
        results = evaluate_scrolling_window_with_general_predictions(
            dataset, model, seq_len=seq_len, pred_len=20, 
            device=device, end_time=360, n_samples=MONTE_CARLO_SAMPLES
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
    
    # Plot 2: General predictions with confidence intervals (for first stock)
    if results is not None and len(results['general_predictions']) > 0:
        plt.figure(figsize=(14, 10))
        
        # Plot general predictions for first stock at different timesteps
        stock_idx = 0
        timesteps_to_plot = [0, len(results['timesteps'])//4, len(results['timesteps'])//2, len(results['timesteps'])-1]
        
        for i, timestep in enumerate(timesteps_to_plot):
            if timestep < len(results['general_predictions']):
                general_preds = results['general_predictions'][timestep][stock_idx]
                ci_lower = results['general_ci_lower'][timestep][stock_idx]
                ci_upper = results['general_ci_upper'][timestep][stock_idx]
                
                plt.subplot(2, 2, i + 1)
                time_points = range(len(general_preds))
                
                plt.plot(time_points, general_preds, 'r-', label='General Prediction', linewidth=2)
                plt.fill_between(time_points, ci_lower, ci_upper, alpha=0.3, color='red', label='95% CI')
                plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                plt.title(f'General Prediction at Timestep {timestep}')
                plt.xlabel('Minutes from Current Time')
                plt.ylabel('Percentage Change')
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
    
    print("\n✅ Scrolling window training and evaluation completed!")
    print(f"📁 Saved plots:")
    print(f"   • 'training_confidence_intervals.png' - Training confidence intervals")
    print(f"   • 'scrolling_window_immediate_predictions.png' - Immediate predictions")
    print(f"   • 'scrolling_window_general_predictions.png' - General predictions with CI")
    print(f"   • 'scrolling_window_accuracy_over_time.png' - Accuracy over time")
    print(f"📊 Model files saved:")
    print(f"   • 'mambastock_scrolling_two.pth' - Final trained model")
    print(f"   • 'mambastock_scrolling_two.pth' - Intermediate training saves (every 50 steps)")
    print(f"🎯 Scrolling window approach with confidence intervals implemented!") 












    import argparse
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import ClosePrice
from mambastock_model import MambaStock

# Prefer the picky strategy file if available
try:
    from bare_strategy_picky import TradingStrategy, Trade
except Exception:
    from bare_strategy import TradingStrategy, Trade  # fallback


# ------------------------------ Reproducibility ------------------------------

def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------ Model ------------------------------

def load_model(device: torch.device, model_path: str, input_size: int, seq_len: int, pred_len: int) -> MambaStock:
    print("🔧 Creating model…")
    model = MambaStock(input_size=input_size, seq_len=seq_len, pred_len=pred_len).to(device)
    model.eval()
    if device.type == "cuda":
        try:
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("✅ Enabled cuDNN benchmark + TF32")
        except Exception:
            pass
    if model_path and os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"✅ Loaded weights: {model_path}")
    else:
        print(f"⚠️  Weights not found: {model_path}. Using random init.")
    return model


# ------------------------------ Date bucketing ------------------------------

def extract_date(name: str) -> str:
    # Simple patterns; retain "unknown_date" by request (critique #10 excluded)
    import re
    pats = [
        r"(\d{4})-(\d{1,2})-(\d{1,2})",
        r"(\d{4})/(\d{1,2})/(\d{1,2})",
        r"(\d{4})\.(\d{1,2})\.(\d{1,2})",
        r"(\d{4})(\d{2})(\d{2})",
        r"(\d{1,2})/(\d{1,2})/(\d{4})",
    ]
    for p in pats:
        m = re.search(p, name)
        if not m:
            continue
        g = m.groups()
        if len(g[0]) == 4:
            y, mo, d = g[0], g[1].zfill(2), g[2].zfill(2)
        else:
            mo, d, y = g[0].zfill(2), g[1].zfill(2), g[2]
        return f"{y}-{mo}-{d}"
    return "unknown_date"


def build_date_batches(name_date_list: List[str]) -> List[Tuple[str, List[int]]]:
    date_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, name in enumerate(name_date_list):
        date = extract_date(name)
        date_to_indices[date].append(i)
    # Keep unknown_date bucket as requested
    batches = sorted(list(date_to_indices.items()), key=lambda kv: kv[0])
    return batches


# ------------------------------ PnL ------------------------------

def realized_pnl(trades: List[Trade]) -> float:
    buys = [t for t in trades if t.side == "BUY"]
    sells = [t for t in trades if t.side == "SELL"]
    return sum(t.cash_flow for t in sells) - sum(t.cash_flow for t in buys)


# ------------------------------ CSV & plots ------------------------------

def save_predictions_csv(strategy: TradingStrategy, stock_indices: List[int], dataset: ClosePrice, out_path: str) -> None:
    import csv
    idx_by_name = {dataset.name_date[i]: i for i in stock_indices}
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stock", "timestamp", "mu", "action", "price"])
        w.writeheader()
        for p in strategy.all_predictions:
            name = p["stock"]
            t = p["time"]
            idx = idx_by_name.get(name)
            price = dataset.data[idx][t, 3].item() if idx is not None and 0 <= t < len(dataset.data[idx]) else ""
            w.writerow({"stock": name, "timestamp": t, "mu": p["mu"], "action": p["action"], "price": price})
    print(f"✅ Saved predictions: {out_path}")


def plot_mu_vs_price(strategy: TradingStrategy, stock_indices: List[int], dataset: ClosePrice, date: str, out_png: str) -> None:
    try:
        names = list({p["stock"] for p in strategy.all_predictions})
        if not names:
            print("    ⚠️  No predictions to plot.")
            return
        names = names[:2]
        fig, axes = plt.subplots(len(names), 1, figsize=(14, 4 * len(names)))
        if len(names) == 1:
            axes = [axes]
        for ax, name in zip(axes, names):
            idx = {dataset.name_date[i]: i for i in stock_indices}.get(name)
            preds = [p for p in strategy.all_predictions if p["stock"] == name]
            ts = [p["time"] for p in preds]
            mus = [p["mu"] * 100.0 for p in preds]
            prices = [dataset.data[idx][t, 3].item() for t in ts]
            ax.plot(ts, prices, label="Price", linewidth=2)
            ax2 = ax.twinx()
            ax2.plot(ts, mus, "--", label="mu20 (%)", linewidth=2)
            ax.set_title(f"{name} — {date}")
            ax.set_xlabel("Minute since 9:30")
            ax.set_ylabel("Price")
            ax2.set_ylabel("%")
            ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"    📊 Saved plot: {out_png}")
    except Exception as e:
        print(f"    ⚠️  Plot failed: {e}")


# ------------------------------ Backtest runner ------------------------------

@torch.inference_mode()
def run_backtest(
    h5_path: str,
    model_path: str,
    *,
    initial_portfolio: float = 100000.0,
    seq_len: int = 50,
    pred_len: int = 20,
    trade_start: int = 90,
    trade_end: int = 240,
    predict_start: int = 0,
    per_name_cap: float = 0.01,
    gross_add_cap: float = 0.02,
    top_k: int = None,
    spread_bps: float = 5.0,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
    buffer_bps: float = 0.0,
    clamp_k: float = 2.0,
    seed: int = 1337,
    save_csv: bool = True,
    make_plots: bool = True,
):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    dataset = ClosePrice(h5_path)
    print(f"📦 Dataset: {dataset.n_stocks} stocks | {dataset.day_length} minutes/day")

    # Infer input_size from dataset features
    input_size = dataset.data[0].shape[1]
    model = load_model(device, model_path, input_size=input_size, seq_len=seq_len, pred_len=pred_len)

    date_batches = build_date_batches(dataset.name_date)
    print(f"📅 Trading dates: {len(date_batches)} (including 'unknown_date' if present)")

    portfolio = float(initial_portfolio)
    total_realized = 0.0

    for day_idx, (date, indices) in enumerate(date_batches, start=1):
        print("=" * 72)
        print(f"📊 DAY {day_idx}/{len(date_batches)} — {date} | Universe={len(indices)} | Start=${portfolio:,.2f}")

        strat = TradingStrategy(
            model=model,
            device=device,
            dataset=dataset,
            considered_stocks=indices,
            portfolio_value=portfolio,
            seq_len=seq_len,
            pred_len=pred_len,
            predict_start_minute=predict_start,
            start_trading_minute=trade_start,
            trade_end_minute=trade_end,
            per_name_cap=per_name_cap,
            gross_add_cap=gross_add_cap,
            top_k=top_k,
            buy_threshold=0.001,
            stop_loss=-0.005,
            liquidate_eod=True,
            spread_bps=spread_bps,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            buffer_bps=buffer_bps,
            clamp_k=clamp_k,
            close_col=3,
        )

        trades, _ = strat.run_day()
        pnl = realized_pnl(trades)
        total_realized += pnl
        portfolio += pnl

        print(f"💵 {date}: Realized=${pnl:,.2f} | End=${portfolio:,.2f} | Trades={len(trades)}")
        strat.analyze_spread_impact()

        if save_csv:
            save_predictions_csv(strat, indices, dataset, f"predictions_{date}.csv")
        if make_plots:
            plot_mu_vs_price(strat, indices, dataset, date, f"mu_vs_price_{date}.png")

    print("=" * 72)
    print("🎯 FINAL")
    print(f"  Initial: ${initial_portfolio:,.2f}")
    print(f"  Final:   ${portfolio:,.2f}")
    print(f"  Return:  ${total_realized:,.2f}  ({(total_realized/initial_portfolio)*100:.2f}%)")


# ------------------------------ CLI ------------------------------

def main():
    p = argparse.ArgumentParser(description="Backtest TradingStrategy (picky, no‑chaining, cost‑aware)")
    p.add_argument("--h5", type=str, required=True, help="Path to HDF5 file")
    p.add_argument("--model", type=str, required=True, help="Path to model weights (.pth)")
    p.add_argument("--init", type=float, default=100000.0, help="Initial portfolio value")
    p.add_argument("--seq_len", type=int, default=50)
    p.add_argument("--pred_len", type=int, default=20)
    p.add_argument("--trade_start", type=int, default=90)
    p.add_argument("--trade_end", type=int, default=240)
    p.add_argument("--predict_start", type=int, default=0)
    p.add_argument("--per_name_cap", type=float, default=0.01)
    p.add_argument("--gross_add_cap", type=float, default=0.02)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--spread_bps", type=float, default=5.0)
    p.add_argument("--commission_bps", type=float, default=0.0)
    p.add_argument("--slippage_bps", type=float, default=0.0)
    p.add_argument("--buffer_bps", type=float, default=0.0)
    p.add_argument("--clamp_k", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--no_csv", action="store_true")
    p.add_argument("--no_plots", action="store_true")

    args = p.parse_args()

    run_backtest(
        h5_path=args.h5,
        model_path=args.model,
        initial_portfolio=args.init,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        trade_start=args.trade_start,
        trade_end=args.trade_end,
        predict_start=args.predict_start,
        per_name_cap=args.per_name_cap,
        gross_add_cap=args.gross_add_cap,
        top_k=args.top_k,
        spread_bps=args.spread_bps,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        buffer_bps=args.buffer_bps,
        clamp_k=args.clamp_k,
        seed=args.seed,
        save_csv=not args.no_csv,
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
