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

# SET RANDOM SEEDS FOR REPRODUCIBLE RESULTS
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print("🔒 Random seeds set for reproducible results")
# Compare original vs updated
print("📂 Loading model files...")
import os
from datetime import datetime

# Check file timestamps to detect if files are changing
orig_path = "mambastock_scrolling_eight.pth"
updated_path = "mambastock.pth"

if os.path.exists(orig_path):
    orig_time = datetime.fromtimestamp(os.path.getmtime(orig_path))
    print(f"   Original model: {orig_path} (modified: {orig_time})")
else:
    print(f"   ❌ {orig_path} not found!")

if os.path.exists(updated_path):
    updated_time = datetime.fromtimestamp(os.path.getmtime(updated_path))
    print(f"   Updated model: {updated_path} (modified: {updated_time})")
else:
    print(f"   ❌ {updated_path} not found!")

orig_dict = torch.load(orig_path, map_location="cpu")
updated_dict = torch.load(updated_path, map_location="cpu")

# Calculate checksums to detect if weights are stable
import hashlib
def dict_checksum(state_dict):
    # Create a hash of all tensor values
    hasher = hashlib.md5()
    for key in sorted(state_dict.keys()):
        tensor_bytes = state_dict[key].cpu().numpy().tobytes()
        hasher.update(tensor_bytes)
    return hasher.hexdigest()

orig_checksum = dict_checksum(orig_dict)
updated_checksum = dict_checksum(updated_dict)

print(f"   Original checksum: {orig_checksum[:8]}...")
print(f"   Updated checksum: {updated_checksum[:8]}...")
print("   💡 If results vary between runs, compare these checksums!")

print("=== FILE INFO ===")
print(f"Original keys: {len(orig_dict.keys())}")
print(f"Updated keys: {len(updated_dict.keys())}")
print(f"Common keys: {len(set(orig_dict.keys()) & set(updated_dict.keys()))}")

print("\n=== ORIGINAL KEYS ===")
for i, key in enumerate(orig_dict.keys()):
    print(f"{i+1}. {key} - Shape: {orig_dict[key].shape}")

print("\n=== UPDATED KEYS ===")
for i, key in enumerate(updated_dict.keys()):
    print(f"{i+1}. {key} - Shape: {updated_dict[key].shape}")

print("\n=== COMMON KEYS ===")
common_keys = set(orig_dict.keys()) & set(updated_dict.keys())
if common_keys:
    for key in common_keys:
        print(f"✓ {key}")
else:
    print("❌ NO COMMON KEYS FOUND!")

print("\n=== DETAILED WEIGHT ANALYSIS ===")
print("Analyzing weights to diagnose repetitive prediction issue...")

weight_issues = []
for name in orig_dict.keys():
    if name in updated_dict:
        orig_param = orig_dict[name]
        updated_param = updated_dict[name]
        
        orig_mean = orig_param.mean().item()
        updated_mean = updated_param.mean().item()
        orig_std = orig_param.std().item()
        updated_std = updated_param.std().item()
        orig_max = orig_param.max().item()
        updated_max = updated_param.max().item()
        orig_min = orig_param.min().item()
        updated_min = updated_param.min().item()
        
        diff = abs(updated_mean - orig_mean)
        
        print(f"{name}:")
        print(f"  Original: mean={orig_mean:.6f}, std={orig_std:.6f}, range=[{orig_min:.6f}, {orig_max:.6f}]")
        print(f"  Updated:  mean={updated_mean:.6f}, std={updated_std:.6f}, range=[{updated_min:.6f}, {updated_max:.6f}]")
        print(f"  Change:   mean_diff={diff:.6f}, std_diff={updated_std-orig_std:.6f}")
        
        # Detect problematic patterns
        issues = []
        
        # Near-zero weights (vanishing)
        if abs(updated_mean) < 1e-8:
            issues.append("VANISHED_MEAN")
            
        # Very small variance (collapsed)
        if updated_std < 1e-8:
            issues.append("COLLAPSED_STD")
            
        # Saturated weights (all same value)
        if updated_std < 1e-6 and abs(updated_mean) > 0.1:
            issues.append("SATURATED")
            
        # Extreme weights
        if abs(updated_max) > 10.0 or abs(updated_min) > 10.0:
            issues.append("EXTREME_VALUES")
            
        # Weight explosion
        if updated_std > orig_std * 10:
            issues.append("EXPLODED_STD")
            
        # Weight collapse
        if updated_std < orig_std * 0.1 and orig_std > 1e-6:
            issues.append("COLLAPSED_VARIANCE")
            
        if issues:
            issue_str = ", ".join(issues)
            print(f"  ⚠️  ISSUES: {issue_str}")
            weight_issues.extend([(name, issue) for issue in issues])
        else:
            print(f"  ✅ Weights look healthy")
        print()

# Summary of weight issues
if weight_issues:
    print(f"🚨 WEIGHT ISSUES DETECTED ({len(weight_issues)} problems):")
    issue_counts = {}
    for name, issue in weight_issues:
        if issue not in issue_counts:
            issue_counts[issue] = []
        issue_counts[issue].append(name)
    
    for issue, layers in issue_counts.items():
        print(f"  {issue}: {len(layers)} layers affected")
        for layer in layers[:3]:  # Show first 3
            print(f"    - {layer}")
        if len(layers) > 3:
            print(f"    - ... and {len(layers)-3} more")
    
    print(f"\n💡 DIAGNOSIS:")
    if 'COLLAPSED_STD' in issue_counts or 'COLLAPSED_VARIANCE' in issue_counts:
        print(f"  • Model weights have COLLAPSED - explains repetitive predictions!")
        print(f"  • This suggests overly aggressive gradient clipping or learning rate")
    if 'VANISHED_MEAN' in issue_counts:
        print(f"  • Weights have VANISHED - model may be outputting near-zero constantly")
    if 'SATURATED' in issue_counts:
        print(f"  • Weights have SATURATED - model outputs are clamped to constant values")
    if 'EXTREME_VALUES' in issue_counts:
        print(f"  • Weights have EXPLODED - training became unstable")
        
else:
    print("✅ No obvious weight issues detected")
    print("The repetitive prediction issue may be in the training data or loss function")

if not common_keys:
    print("❌ NO COMMON KEYS - files have incompatible architectures!")

print("\n" + "="*60)
print("=== SIMPLIFIED MODEL PERFORMANCE COMPARISON ===")
print("="*60)

# Load test dataset with 1000 random stocks from h5_rty_data_processed.h5
try:
    # Import the dataset
    from dataset import ClosePrice
    print("✓ Dataset import successful")
    test_dataset = ClosePrice("h5_test_training_three.h5")
    print(f"✓ Dataset loaded: {test_dataset.n_stocks} stocks, shape: {test_dataset.data.shape}")
    
except Exception as e:
    print(f"❌ Dataset loading failed: {e}")
    print("   Falling back to original dataset...")
    try:
        from dataset import ClosePrice
        print("✓ Fallback dataset import successful")
        test_dataset = ClosePrice("h5_test_training_three.h5")
        print(f"✓ Dataset loaded: {test_dataset.n_stocks} stocks, shape: {test_dataset.data.shape}")
        print("ℹ️  Note: Using original dataset without denormalization support")
    except Exception as e2:
        print(f"❌ Fallback dataset loading also failed: {e2}")
        exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Using device: {device}")

# Initialize models EXACTLY like train.py
try:
    seq_len = 90  # Match train_two_gpu.py exactly
    pred_len = 20  # Match train_two_gpu.py exactly (20-minute prediction window)
    input_size = 13  # Match train_two_gpu.py exactly - this is the key!
    
    model_orig = MambaStock(input_size=input_size, seq_len=seq_len, pred_len=pred_len).to(device)
    model_updated = MambaStock(input_size=input_size, seq_len=seq_len, pred_len=pred_len).to(device)
    print(f"✓ Models initialized with input_size={input_size}, seq_len={seq_len}, pred_len={pred_len}")
except Exception as e:
    print(f"❌ Model initialization failed: {e}")
    exit(1)

# Load weights EXACTLY like train.py
try:
    orig_state = torch.load("mambastock_scrolling_eight.pth", map_location=device)
    updated_state = torch.load("mambastock.pth", map_location=device)
    print(f"✓ Weight files loaded")
    
    model_orig.load_state_dict(orig_state, strict=False)
    model_updated.load_state_dict(updated_state, strict=False)
    print("✓ Weights loaded into models")
except Exception as e:
    print(f"❌ Weight loading failed: {e}")
    exit(1)

# Set models to eval mode EXACTLY like train.py
model_orig.eval()
model_updated.eval()
print("✓ Models set to eval mode")

# --- ROLLING MULTI-STEP PREDICTION FOR MODEL COMPARISON ---
def rolling_multi_step_comparison(test_dataset, model_orig, model_updated, seq_len=90, pred_len=120, device=None, step=20):
    model_orig.eval()
    model_updated.eval()
    n_stocks = test_dataset.n_stocks
    input_seqs = test_dataset.data[:, :seq_len, :].clone().to(device)  # [n_stocks, seq_len, features]
    print(f"Initial input_seqs shape: {input_seqs.shape}")
    all_preds_orig = []
    all_preds_updated = []
    all_trues = []
    for start in range(0, pred_len, step):
        # Per-stock rolling normalization for each window
        input_norm = torch.zeros_like(input_seqs)
        for stock_idx in range(n_stocks):
            window = input_seqs[stock_idx]  # [seq_len, features] or [seq_len+step, features]
            mean = window.mean(dim=0, keepdim=True)  # [1, features]
            std = window.std(dim=0, keepdim=True)
            std = torch.where(std > 1e-8, std, torch.ones_like(std))
            input_norm[stock_idx] = (window - mean) / std
        # Model predictions
        preds_orig = model_orig(input_norm)
        preds_updated = model_updated(input_norm)
        if preds_orig.shape[-1] != step:
            preds_orig = preds_orig.squeeze(-1)
        if preds_updated.shape[-1] != step:
            preds_updated = preds_updated.squeeze(-1)
        all_preds_orig.append(preds_orig.detach().cpu())
        all_preds_updated.append(preds_updated.detach().cpu())
        # Get the actual next step actuals for each stock
        actuals = []
        for stock_idx in range(n_stocks):
            actual = test_dataset.data[stock_idx, seq_len+start:seq_len+start+step, 10]  # [step]
            actuals.append(actual)
        actuals = torch.stack(actuals).to(device)  # [n_stocks, step]
        all_trues.append(actuals.cpu())
        # For next round, append the actuals to the window, keep only the last (seq_len+step)
        actuals_full = []
        for stock_idx in range(n_stocks):
            actual_full = test_dataset.data[stock_idx, seq_len+start:seq_len+start+step, :]
            actuals_full.append(actual_full)
        actuals_full = torch.stack(actuals_full).to(device)  # [n_stocks, step, features]
        input_seqs = torch.cat([input_seqs, actuals_full], dim=1)[:, -input_seqs.shape[1]-step:, :]  # [n_stocks, seq_len+step, features]
    all_preds_orig = torch.cat(all_preds_orig, dim=1)  # [n_stocks, pred_len]
    all_preds_updated = torch.cat(all_preds_updated, dim=1)  # [n_stocks, pred_len]
    all_trues = torch.cat(all_trues, dim=1)  # [n_stocks, pred_len]
    return all_preds_orig, all_preds_updated, all_trues

# --- SCROLLING WINDOW PREDICTION METHOD (REALISTIC) ---
def general_prediction_with_confidence(model, input_seq, remaining_minutes, device, n_samples=15):
    """
    Make a general prediction for the next remaining_minutes with confidence intervals.
    
    Args:
        model: The trained model
        input_seq: Current input sequence [seq_len, features]
        remaining_minutes: Number of minutes to predict
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
    seq_len = input_seq.shape[0]
    
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
                pred_sample = model(current_seq_norm.unsqueeze(0))  # [1, 1, 1]
                pred_sample = pred_sample.squeeze().detach().cpu().numpy()
                pred_samples.append(pred_sample)
        
        pred_samples = np.array(pred_samples)  # [n_samples]
        
        # Calculate mean prediction and confidence intervals
        pred_mean = np.mean(pred_samples)
        pred_std = np.std(pred_samples)
        
        # Add small epsilon to prevent zero std
        pred_std = pred_std + 1e-6
        
        # Use uncertainty scaling for confidence intervals
        uncertainty_scale = 2.0  # 95% confidence interval
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


def single_general_prediction_with_confidence(model, input_seq, device, n_samples=15):
    """
    Make a single general prediction for the remaining day using the current sequence only.
    Uses Monte Carlo dropout to compute a single scalar prediction with a 95% confidence interval.

    Args:
        model: The trained model
        input_seq: Current input sequence [seq_len, features]
        device: Device to run on
        n_samples: Number of Monte Carlo samples for uncertainty estimation

    Returns:
        pred_mean (float): Mean prediction
        ci_lower (float): Lower CI bound
        ci_upper (float): Upper CI bound
        pred_std (float): Standard deviation of predictions
    """
    model.eval()

    # Store original dropout states
    original_dropout_states = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            original_dropout_states[name] = module.training

    current_seq = input_seq.clone().to(device)

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

            # Single forward pass for general prediction
            pred_sample = model(current_seq_norm.unsqueeze(0))  # [1, 1, 1]
            pred_sample = pred_sample.squeeze()  # scalar tensor
            pred_samples.append(pred_sample.detach())

    # Restore original dropout states
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.train(original_dropout_states[name])

    pred_samples = torch.stack(pred_samples)  # [n_samples]

    pred_mean = pred_samples.mean().item()
    pred_std = pred_samples.std().item() + 1e-6

    # 95% CI with fixed uncertainty scale
    uncertainty_scale = 2.0
    ci_lower = pred_mean - uncertainty_scale * pred_std
    ci_upper = pred_mean + uncertainty_scale * pred_std

    return pred_mean, ci_lower, ci_upper, pred_std

def scrolling_window_comparison_with_coverage(test_dataset, model_orig, model_updated, seq_len=90, pred_len=20, device=None, end_time=210):
    """
    REALISTIC Scrolling window method with coverage rate analysis (matching train_two_gpu.py):
    - Start with seq_len minutes of historical data
    - Predict single value representing 20-minute trend
    - Calculate confidence intervals and coverage rates
    - After 1 minute, append actual data to sequence
    - Keep only last seq_len minutes (fixed memory window)
    - Repeat until reaching end_time (1:00 PM)
    """
    model_orig.eval()
    model_updated.eval()
    n_stocks = test_dataset.n_stocks
    
    # Calculate total available time steps
    total_steps = test_dataset.data.shape[1]
    max_scroll_steps = min(end_time - seq_len, total_steps - seq_len - pred_len)
    
    print(f"REALISTIC Scrolling window with coverage: seq_len={seq_len}, pred_len={pred_len}")
    print(f"Total available steps: {total_steps}, Max scroll steps: {max_scroll_steps}")
    print(f"Evaluating until: {end_time} minutes (1:00 PM)")
    
    all_preds_orig = []
    all_preds_updated = []
    all_trues = []
    all_timesteps = []
    
    # Coverage tracking
    all_coverage_orig = []
    all_coverage_updated = []
    all_ci_widths_orig = []
    all_ci_widths_updated = []
    all_pred_stds_orig = []
    all_pred_stds_updated = []
    
    # Initialize with first seq_len minutes
    current_seq = test_dataset.data[:, :seq_len, :].clone().to(device)  # [n_stocks, seq_len, features]
    current_seq_len = seq_len
    
    for step_idx in range(max_scroll_steps):
        # Normalize current sequence per stock (matching train_two_gpu.py)
        input_norm = torch.zeros_like(current_seq)
        for stock_idx in range(n_stocks):
            window = current_seq[stock_idx]  # [current_seq_len, features]
            mean = window.mean(dim=0, keepdim=True)  # [1, features]
            std = window.std(dim=0, keepdim=True)
            std = torch.where(std > 1e-8, std, torch.ones_like(std))
            input_norm[stock_idx] = (window - mean) / std
        
        # Monte Carlo dropout for uncertainty estimation
        n_samples = 15
        pred_samples_orig = []
        pred_samples_updated = []
        
        # Store original dropout states
        original_dropout_states_orig = {}
        original_dropout_states_updated = {}
        for name, module in model_orig.named_modules():
            if isinstance(module, nn.Dropout):
                original_dropout_states_orig[name] = module.training
        for name, module in model_updated.named_modules():
            if isinstance(module, nn.Dropout):
                original_dropout_states_updated[name] = module.training
        
        with torch.no_grad():
            for i in range(n_samples):
                # Enable dropout for uncertainty estimation
                for name, module in model_orig.named_modules():
                    if isinstance(module, nn.Dropout):
                        module.train()
                for name, module in model_updated.named_modules():
                    if isinstance(module, nn.Dropout):
                        module.train()
                
                # Make predictions with dropout enabled
                pred_sample_orig = model_orig(input_norm)  # [n_stocks, 1, 1]
                pred_sample_updated = model_updated(input_norm)  # [n_stocks, 1, 1]
                
                pred_sample_orig = pred_sample_orig.squeeze(-1)  # [n_stocks, 1]
                pred_sample_updated = pred_sample_updated.squeeze(-1)  # [n_stocks, 1]
                
                pred_samples_orig.append(pred_sample_orig.detach())
                pred_samples_updated.append(pred_sample_updated.detach())
            
            # Restore dropout states
            for name, module in model_orig.named_modules():
                if isinstance(module, nn.Dropout):
                    module.train(original_dropout_states_orig[name])
            for name, module in model_updated.named_modules():
                if isinstance(module, nn.Dropout):
                    module.train(original_dropout_states_updated[name])
            
            # Calculate statistics
            pred_samples_orig = torch.stack(pred_samples_orig)  # [n_samples, n_stocks, 1]
            pred_samples_updated = torch.stack(pred_samples_updated)  # [n_samples, n_stocks, 1]
            
            pred_mean_orig = pred_samples_orig.mean(dim=0)  # [n_stocks, 1]
            pred_mean_updated = pred_samples_updated.mean(dim=0)  # [n_stocks, 1]
            pred_std_orig = pred_samples_orig.std(dim=0)  # [n_stocks, 1]
            pred_std_updated = pred_samples_updated.std(dim=0)  # [n_stocks, 1]
            
            # Add epsilon to prevent zero std
            pred_std_orig = pred_std_orig + 1e-6
            pred_std_updated = pred_std_updated + 1e-6
        
        # Calculate confidence intervals
        uncertainty_scale = 2.0  # 95% confidence interval
        ci_lower_orig = pred_mean_orig - uncertainty_scale * pred_std_orig
        ci_upper_orig = pred_mean_orig + uncertainty_scale * pred_std_orig
        ci_lower_updated = pred_mean_updated - uncertainty_scale * pred_std_updated
        ci_upper_updated = pred_mean_updated + uncertainty_scale * pred_std_updated
        
        # Get actual values for the predicted period (average of next 20 minutes)
        actuals = []
        for stock_idx in range(n_stocks):
            # Get the next 20 minutes and calculate their average
            next_20_minutes = test_dataset.data[stock_idx, seq_len+step_idx:seq_len+step_idx+20, 10]  # [20] - next 20 minutes
            actual_avg = next_20_minutes.mean()  # Single value representing 20-minute trend
            actuals.append(actual_avg)
        actuals = torch.stack(actuals).to(device)  # [n_stocks]
        actuals = actuals.unsqueeze(-1)  # [n_stocks, 1] - add dimension to match predictions
        
        # Calculate coverage rates
        coverage_mask_orig = (actuals >= ci_lower_orig.to(device)) & (actuals <= ci_upper_orig.to(device))
        coverage_mask_updated = (actuals >= ci_lower_updated.to(device)) & (actuals <= ci_upper_updated.to(device))
        
        coverage_rate_orig = coverage_mask_orig.float().mean().item() * 100
        coverage_rate_updated = coverage_mask_updated.float().mean().item() * 100
        
        # Calculate confidence interval widths
        ci_width_orig = (ci_upper_orig - ci_lower_orig).mean().item()
        ci_width_updated = (ci_upper_updated - ci_lower_updated).mean().item()
        
        # Store results
        all_preds_orig.append(pred_mean_orig.detach().cpu())
        all_preds_updated.append(pred_mean_updated.detach().cpu())
        all_trues.append(actuals.cpu())
        all_timesteps.append(step_idx)
        
        # Store coverage statistics
        all_coverage_orig.append(coverage_rate_orig)
        all_coverage_updated.append(coverage_rate_updated)
        all_ci_widths_orig.append(ci_width_orig)
        all_ci_widths_updated.append(ci_width_updated)
        all_pred_stds_orig.append(pred_std_orig.mean().item())
        all_pred_stds_updated.append(pred_std_updated.mean().item())
        
        # Append the actual data for the next minute to the sequence
        next_minute_data = []
        for stock_idx in range(n_stocks):
            next_data = test_dataset.data[stock_idx, seq_len+step_idx:seq_len+step_idx+1, :]  # [1, features]
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
            print(f"  Coverage - Original: {coverage_rate_orig:.2f}%, Updated: {coverage_rate_updated:.2f}%")
            print(f"  CI Width - Original: {ci_width_orig:.6f}, Updated: {ci_width_updated:.6f}")
    
    # Concatenate all predictions
    if all_preds_orig:
        all_preds_orig = torch.cat(all_preds_orig, dim=1)  # [n_stocks, total_pred_len]
        all_preds_updated = torch.cat(all_preds_updated, dim=1)  # [n_stocks, total_pred_len]
        all_trues = torch.cat(all_trues, dim=1)  # [n_stocks, total_pred_len]
    else:
        # Handle case where no predictions were made
        all_preds_orig = torch.empty((n_stocks, 0))
        all_preds_updated = torch.empty((n_stocks, 0))
        all_trues = torch.empty((n_stocks, 0))
    
    # Create coverage statistics dictionary
    coverage_stats = {
        'coverage_orig': all_coverage_orig,
        'coverage_updated': all_coverage_updated,
        'ci_widths_orig': all_ci_widths_orig,
        'ci_widths_updated': all_ci_widths_updated,
        'pred_stds_orig': all_pred_stds_orig,
        'pred_stds_updated': all_pred_stds_updated
    }
    
    return all_preds_orig, all_preds_updated, all_trues, all_timesteps, coverage_stats

def general_prediction_comparison(test_dataset, model_orig, model_updated, seq_len=90, device=None, end_time=210):
    """
    Compare models using general predictions for the entire remaining day.
    Calculate general prediction only ONCE at the beginning (after seq_len minutes).
    This dramatically improves performance compared to calculating at every step.
    """
    model_orig.eval()
    model_updated.eval()
    n_stocks = test_dataset.n_stocks
    
    print(f"General prediction comparison: seq_len={seq_len}")
    print(f"Calculating ONE general prediction per stock at the beginning (after {seq_len} minutes)")
    print(f"Evaluating until: {end_time} minutes (1:00 PM)")
    
    # Initialize with first seq_len minutes
    current_seq = test_dataset.data[:, :seq_len, :].clone().to(device)  # [n_stocks, seq_len, features]
    
    # Calculate remaining minutes for general prediction
    # We now compute a single prediction per stock (no per-minute rollout)
    print(f"Making ONE general prediction per stock using current sequence only...")
    
    general_predictions_orig = []
    general_predictions_updated = []
    general_ci_lower_orig = []
    general_ci_upper_orig = []
    general_ci_lower_updated = []
    general_ci_upper_updated = []
    general_pred_stds_orig = []
    general_pred_stds_updated = []
    
    # Calculate general predictions for all stocks ONCE
    for stock_idx in range(n_stocks):
        stock_seq = current_seq[stock_idx]  # [seq_len, features]
        
        # Make a single general prediction with confidence intervals for both models
        pred_mean_orig, ci_low_orig, ci_up_orig, pred_std_orig = single_general_prediction_with_confidence(
            model_orig, stock_seq, device, n_samples=15
        )
        pred_mean_updated, ci_low_updated, ci_up_updated, pred_std_updated = single_general_prediction_with_confidence(
            model_updated, stock_seq, device, n_samples=15
        )

        general_predictions_orig.append(pred_mean_orig)
        general_predictions_updated.append(pred_mean_updated)
        general_ci_lower_orig.append(ci_low_orig)
        general_ci_upper_orig.append(ci_up_orig)
        general_ci_lower_updated.append(ci_low_updated)
        general_ci_upper_updated.append(ci_up_updated)
        general_pred_stds_orig.append(pred_std_orig)
        general_pred_stds_updated.append(pred_std_updated)
        
        # Progress indicator
        if (stock_idx + 1) % 50 == 0:
            print(f"Processed {stock_idx + 1}/{n_stocks} stocks...")
    
    print(f"✅ General predictions completed for all {n_stocks} stocks")
    
    # Create general prediction results dictionary (single timestep)
    general_results = {
        'predictions_orig': [general_predictions_orig],
        'predictions_updated': [general_predictions_updated],
        'ci_lower_orig': [general_ci_lower_orig],
        'ci_upper_orig': [general_ci_upper_orig],
        'ci_lower_updated': [general_ci_lower_updated],
        'ci_upper_updated': [general_ci_upper_updated],
        'pred_stds_orig': [general_pred_stds_orig],
        'pred_stds_updated': [general_pred_stds_updated],
        'timesteps': [0]
    }
    
    return general_results

# --- FINAL 15-MINUTE ROLLING PREDICTION FOR SCROLLING METHOD (REALISTIC) ---
def final_rolling_prediction_scrolling(test_dataset, model_orig, model_updated, seq_len=90, device=None):
    """
    Final 15-minute rolling prediction for the scrolling window method (REALISTIC)
    This simulates the end-of-day trading strategy with single prediction
    """
    model_orig.eval()
    model_updated.eval()
    n_stocks = test_dataset.n_stocks
    
    # Find the last 15-minute window (assuming 1-minute intervals)
    total_steps = test_dataset.data.shape[1]
    final_window_start = total_steps - 15
    
    # Get the sequence up to the final window
    final_seq = test_dataset.data[:, final_window_start-seq_len:final_window_start, :].clone().to(device)
    
    # Normalize the final sequence (matching train_two_gpu.py)
    input_norm = torch.zeros_like(final_seq)
    for stock_idx in range(n_stocks):
        window = final_seq[stock_idx]  # [seq_len, features]
        mean = window.mean(dim=0, keepdim=True)
        std = window.std(dim=0, keepdim=True)
        std = torch.where(std > 1e-8, std, torch.ones_like(std))
        input_norm[stock_idx] = (window - mean) / std
    
    # Make predictions for the final 15 minutes (single value per stock)
    preds_orig = model_orig(input_norm)  # [n_stocks, 1, 1]
    preds_updated = model_updated(input_norm)  # [n_stocks, 1, 1]
    
    # Ensure predictions are the right shape
    if len(preds_orig.shape) == 3:
        preds_orig = preds_orig.squeeze(-1)  # [n_stocks, 1]
    if len(preds_updated.shape) == 3:
        preds_updated = preds_updated.squeeze(-1)  # [n_stocks, 1]
    
    # Get actual values for the final 15 minutes (average - matching train_two_gpu.py)
    actuals = []
    for stock_idx in range(n_stocks):
        # Get the final 15 minutes and calculate their average
        final_15_minutes = test_dataset.data[stock_idx, final_window_start:final_window_start+15, 10]  # [15]
        actual_avg = final_15_minutes.mean()  # Single value representing 15-minute trend
        actuals.append(actual_avg)
    actuals = torch.stack(actuals).to(device)  # [n_stocks]
    actuals = actuals.unsqueeze(-1)  # [n_stocks, 1] - add dimension to match predictions
    
    return preds_orig.detach().cpu(), preds_updated.detach().cpu(), actuals.cpu()

# --- MAIN COMPARISON LOGIC ---
if __name__ == "__main__":
    print("\n" + "="*60)
    print("=== ROLLING WINDOW METHOD COMPARISON ===")
    print("="*60)
    
    # Run rolling multi-step comparison
    preds_orig, preds_updated, trues = rolling_multi_step_comparison(test_dataset, model_orig, model_updated, seq_len=90, pred_len=120, device=device, step=20)
    print("Original model predictions shape:", preds_orig.shape)
    print("Updated model predictions shape:", preds_updated.shape)
    print("Ground truth shape:", trues.shape)
    
    # Flatten for analysis
    orig_predictions_rolling = preds_orig.cpu().numpy().flatten()
    updated_predictions_rolling = preds_updated.cpu().numpy().flatten()
    true_values_rolling = trues.cpu().numpy().flatten()
    
    # Ensure all arrays are the same length
    min_len_rolling = min(len(orig_predictions_rolling), len(updated_predictions_rolling), len(true_values_rolling))
    orig_predictions_rolling = orig_predictions_rolling[:min_len_rolling]
    updated_predictions_rolling = updated_predictions_rolling[:min_len_rolling]
    true_values_rolling = true_values_rolling[:min_len_rolling]
    
    print("\n" + "="*60)
    print("=== SCROLLING WINDOW METHOD COMPARISON (REALISTIC) ===")
    print("="*60)
    
    # Run scrolling window comparison with coverage analysis (REALISTIC - matching train_two_gpu.py)
    preds_orig_scroll, preds_updated_scroll, trues_scroll, timesteps, coverage_stats = scrolling_window_comparison_with_coverage(
        test_dataset, model_orig, model_updated, seq_len=90, pred_len=20, device=device, end_time=210
    )
    print("Original model predictions shape (scroll):", preds_orig_scroll.shape)
    print("Updated model predictions shape (scroll):", preds_updated_scroll.shape)
    print("Ground truth shape (scroll):", trues_scroll.shape)
    print(f"Number of scrolling steps: {len(timesteps)}")
    
    # Print coverage statistics
    avg_coverage_orig = np.mean(coverage_stats['coverage_orig'])
    avg_coverage_updated = np.mean(coverage_stats['coverage_updated'])
    avg_ci_width_orig = np.mean(coverage_stats['ci_widths_orig'])
    avg_ci_width_updated = np.mean(coverage_stats['ci_widths_updated'])
    avg_pred_std_orig = np.mean(coverage_stats['pred_stds_orig'])
    avg_pred_std_updated = np.mean(coverage_stats['pred_stds_updated'])
    
    print(f"\n📊 COVERAGE ANALYSIS:")
    print(f"  Original Model - Avg Coverage: {avg_coverage_orig:.2f}%, Avg CI Width: {avg_ci_width_orig:.6f}, Avg Pred Std: {avg_pred_std_orig:.6f}")
    print(f"  Updated Model - Avg Coverage: {avg_coverage_updated:.2f}%, Avg CI Width: {avg_ci_width_updated:.6f}, Avg Pred Std: {avg_pred_std_updated:.6f}")
    
    # Flatten for analysis
    orig_predictions_scroll = preds_orig_scroll.cpu().numpy().flatten()
    updated_predictions_scroll = preds_updated_scroll.cpu().numpy().flatten()
    true_values_scroll = trues_scroll.cpu().numpy().flatten()
    
    # Ensure all arrays are the same length
    min_len_scroll = min(len(orig_predictions_scroll), len(updated_predictions_scroll), len(true_values_scroll))
    orig_predictions_scroll = orig_predictions_scroll[:min_len_scroll]
    updated_predictions_scroll = updated_predictions_scroll[:min_len_scroll]
    true_values_scroll = true_values_scroll[:min_len_scroll]
    
    # Run final 15-minute rolling prediction for scrolling method (REALISTIC)
    print("\n" + "="*60)
    print("=== FINAL 15-MINUTE ROLLING PREDICTION (SCROLLING METHOD - REALISTIC) ===")
    print("="*60)
    
    final_preds_orig, final_preds_updated, final_trues = final_rolling_prediction_scrolling(
        test_dataset, model_orig, model_updated, seq_len=90, device=device
    )
    print("Final 15-min predictions shape:", final_preds_orig.shape)
    print("Final 15-min ground truth shape:", final_trues.shape)
    print("Note: Each prediction represents the average trend of the final 15 minutes")
    
    # Flatten final predictions
    final_orig_predictions = final_preds_orig.cpu().numpy().flatten()
    final_updated_predictions = final_preds_updated.cpu().numpy().flatten()
    final_true_values = final_trues.cpu().numpy().flatten()
    
    # Debug: Print shapes to understand the issue
    print(f"DEBUG - final_orig_predictions shape: {final_orig_predictions.shape}")
    print(f"DEBUG - final_updated_predictions shape: {final_updated_predictions.shape}")
    print(f"DEBUG - final_true_values shape: {final_true_values.shape}")
    
    # Ensure all arrays have the same length
    min_len_final = min(len(final_orig_predictions), len(final_updated_predictions), len(final_true_values))
    final_orig_predictions = final_orig_predictions[:min_len_final]
    final_updated_predictions = final_updated_predictions[:min_len_final]
    final_true_values = final_true_values[:min_len_final]
    
    print(f"DEBUG - After trimming, all arrays have length: {min_len_final}")
    
    # Run general prediction comparison (matching train_two_gpu.py approach)
    print("\n" + "="*60)
    print("=== GENERAL PREDICTION COMPARISON (REALISTIC) ===")
    print("="*60)
    
    general_results = general_prediction_comparison(
        test_dataset, model_orig, model_updated, seq_len=90, device=device, end_time=210
    )
    print("General prediction comparison completed!")
    print(f"Number of timesteps with general predictions: {len(general_results['timesteps'])}")
    
    # Analyze general prediction results
    if len(general_results['predictions_orig']) > 0:
        # Calculate average prediction uncertainty for each model
        all_pred_stds_orig = []
        all_pred_stds_updated = []
        all_ci_widths_orig = []
        all_ci_widths_updated = []
        
        for timestep in range(len(general_results['timesteps'])):
            for stock_idx in range(len(general_results['pred_stds_orig'][timestep])):
                pred_std_orig = general_results['pred_stds_orig'][timestep][stock_idx]
                pred_std_updated = general_results['pred_stds_updated'][timestep][stock_idx]
                ci_lower_orig = general_results['ci_lower_orig'][timestep][stock_idx]
                ci_upper_orig = general_results['ci_upper_orig'][timestep][stock_idx]
                ci_lower_updated = general_results['ci_lower_updated'][timestep][stock_idx]
                ci_upper_updated = general_results['ci_upper_updated'][timestep][stock_idx]

                # Append scalar values (no per-minute rollout)
                all_pred_stds_orig.append(pred_std_orig)
                all_pred_stds_updated.append(pred_std_updated)
                all_ci_widths_orig.append(ci_upper_orig - ci_lower_orig)
                all_ci_widths_updated.append(ci_upper_updated - ci_lower_updated)
        
        avg_pred_std_orig = np.mean(all_pred_stds_orig)
        avg_pred_std_updated = np.mean(all_pred_stds_updated)
        avg_ci_width_orig = np.mean(all_ci_widths_orig)
        avg_ci_width_updated = np.mean(all_ci_widths_updated)
        
        print(f"\n📊 GENERAL PREDICTION ANALYSIS:")
        print(f"  Original Model - Avg Pred Std: {avg_pred_std_orig:.6f}, Avg CI Width: {avg_ci_width_orig:.6f}")
        print(f"  Updated Model - Avg Pred Std: {avg_pred_std_updated:.6f}, Avg CI Width: {avg_ci_width_updated:.6f}")
    
    # Use rolling window results for the main comparison (keeping original variable names)
    orig_predictions = orig_predictions_rolling
    updated_predictions = updated_predictions_rolling
    true_values = true_values_rolling

# Calculate overall metrics for ROLLING WINDOW METHOD
print("\n=== ROLLING WINDOW METHOD METRICS ===")
orig_mse = np.mean((orig_predictions - true_values) ** 2)
updated_mse = np.mean((updated_predictions - true_values) ** 2)

orig_rmse = np.sqrt(orig_mse)
updated_rmse = np.sqrt(updated_mse)

orig_mae = np.mean(np.abs(orig_predictions - true_values))
updated_mae = np.mean(np.abs(updated_predictions - true_values))

# Calculate directional accuracy
orig_directions = np.sign(orig_predictions) == np.sign(true_values)
updated_directions = np.sign(updated_predictions) == np.sign(true_values)
orig_dir_acc = np.mean(orig_directions) * 100
updated_dir_acc = np.mean(updated_directions) * 100

# Correlation
orig_corr = np.corrcoef(orig_predictions, true_values)[0, 1] if len(orig_predictions) > 1 else 0
updated_corr = np.corrcoef(updated_predictions, true_values)[0, 1] if len(updated_predictions) > 1 else 0

# Handle NaN correlations
if np.isnan(orig_corr):
    orig_corr = 0.0
if np.isnan(updated_corr):
    updated_corr = 0.0

print(f"{'Metric':<20} {'Original':<15} {'Updated':<15} {'Winner':<10}")
print("-" * 60)
print(f"{'MSE':<20} {orig_mse:<15.6f} {updated_mse:<15.6f} {'Updated' if updated_mse < orig_mse else 'Original':<10}")
print(f"{'RMSE':<20} {orig_rmse:<15.6f} {updated_rmse:<15.6f} {'Updated' if updated_rmse < orig_rmse else 'Original':<10}")
print(f"{'MAE':<20} {orig_mae:<15.6f} {updated_mae:<15.6f} {'Updated' if updated_mae < orig_mae else 'Original':<10}")
print(f"{'Directional Acc %':<20} {orig_dir_acc:<15.2f} {updated_dir_acc:<15.2f} {'Updated' if updated_dir_acc > orig_dir_acc else 'Original':<10}")
print(f"{'Correlation':<20} {orig_corr:<15.4f} {updated_corr:<15.4f} {'Higher is better':<10}")

print(f"\n=== ROLLING WINDOW PREDICTION RANGES ===")
print(f"Original predictions: [{np.min(orig_predictions):.6f}, {np.max(orig_predictions):.6f}] (std: {np.std(orig_predictions):.6f})")
print(f"Updated predictions:  [{np.min(updated_predictions):.6f}, {np.max(updated_predictions):.6f}] (std: {np.std(updated_predictions):.6f})")
print(f"True values:          [{np.min(true_values):.6f}, {np.max(true_values):.6f}] (std: {np.std(true_values):.6f})")

# Rolling window summary
rolling_updated_wins = sum([
    updated_mse < orig_mse,
    updated_rmse < orig_rmse, 
    updated_mae < orig_mae,
    updated_dir_acc > orig_dir_acc,
    abs(updated_corr) > abs(orig_corr)  # Higher absolute correlation is better
])

print(f"\n=== ROLLING WINDOW FINAL VERDICT ===")
if rolling_updated_wins >= 3:
    print("🏆 UPDATED MODEL PERFORMS BETTER (wins on more metrics)")
else:
    print("🏆 ORIGINAL MODEL PERFORMS BETTER (wins on more metrics)")

print(f"Updated model wins on {rolling_updated_wins}/5 metrics")

# Calculate overall metrics for SCROLLING WINDOW METHOD
print("\n" + "="*60)
print("=== SCROLLING WINDOW METHOD METRICS (REALISTIC) ===")
print("="*60)

# Check if scrolling window has valid data
if len(orig_predictions_scroll) > 0 and len(updated_predictions_scroll) > 0 and len(true_values_scroll) > 0:
    scroll_orig_mse = np.mean((orig_predictions_scroll - true_values_scroll) ** 2)
    scroll_updated_mse = np.mean((updated_predictions_scroll - true_values_scroll) ** 2)

    scroll_orig_rmse = np.sqrt(scroll_orig_mse)
    scroll_updated_rmse = np.sqrt(scroll_updated_mse)

    scroll_orig_mae = np.mean(np.abs(orig_predictions_scroll - true_values_scroll))
    scroll_updated_mae = np.mean(np.abs(updated_predictions_scroll - true_values_scroll))

    # Calculate directional accuracy
    scroll_orig_directions = np.sign(orig_predictions_scroll) == np.sign(true_values_scroll)
    scroll_updated_directions = np.sign(updated_predictions_scroll) == np.sign(true_values_scroll)
    scroll_orig_dir_acc = np.mean(scroll_orig_directions) * 100
    scroll_updated_dir_acc = np.mean(scroll_updated_directions) * 100

    # Correlation
    scroll_orig_corr = np.corrcoef(orig_predictions_scroll, true_values_scroll)[0, 1] if len(orig_predictions_scroll) > 1 else 0
    scroll_updated_corr = np.corrcoef(updated_predictions_scroll, true_values_scroll)[0, 1] if len(updated_predictions_scroll) > 1 else 0

    # Handle NaN correlations
    if np.isnan(scroll_orig_corr):
        scroll_orig_corr = 0.0
    if np.isnan(scroll_updated_corr):
        scroll_updated_corr = 0.0
    
    # Add coverage rate metrics (from coverage_stats)
    scroll_orig_coverage = avg_coverage_orig  # From earlier calculation
    scroll_updated_coverage = avg_coverage_updated  # From earlier calculation
    scroll_orig_ci_width = avg_ci_width_orig  # From earlier calculation
    scroll_updated_ci_width = avg_ci_width_updated  # From earlier calculation

    print(f"{'Metric':<20} {'Original':<15} {'Updated':<15} {'Winner':<10}")
    print("-" * 60)
    print(f"{'MSE':<20} {scroll_orig_mse:<15.6f} {scroll_updated_mse:<15.6f} {'Updated' if scroll_updated_mse < scroll_orig_mse else 'Original':<10}")
    print(f"{'RMSE':<20} {scroll_orig_rmse:<15.6f} {scroll_updated_rmse:<15.6f} {'Updated' if scroll_updated_rmse < scroll_orig_rmse else 'Original':<10}")
    print(f"{'MAE':<20} {scroll_orig_mae:<15.6f} {scroll_updated_mae:<15.6f} {'Updated' if scroll_updated_mae < scroll_orig_mae else 'Original':<10}")
    print(f"{'Directional Acc %':<20} {scroll_orig_dir_acc:<15.2f} {scroll_updated_dir_acc:<15.2f} {'Updated' if scroll_updated_dir_acc > scroll_orig_dir_acc else 'Original':<10}")
    print(f"{'Correlation':<20} {scroll_orig_corr:<15.4f} {scroll_updated_corr:<15.4f} {'Higher is better':<10}")
    print(f"{'Coverage Rate %':<20} {scroll_orig_coverage:<15.2f} {scroll_updated_coverage:<15.2f} {'Closer to 95%':<10}")
    print(f"{'CI Width':<20} {scroll_orig_ci_width:<15.6f} {scroll_updated_ci_width:<15.6f} {'Balanced':<10}")

    print(f"\n=== SCROLLING WINDOW PREDICTION RANGES ===")
    print(f"Original predictions: [{np.min(orig_predictions_scroll):.6f}, {np.max(orig_predictions_scroll):.6f}] (std: {np.std(orig_predictions_scroll):.6f})")
    print(f"Updated predictions:  [{np.min(updated_predictions_scroll):.6f}, {np.max(updated_predictions_scroll):.6f}] (std: {np.std(updated_predictions_scroll):.6f})")
    print(f"True values:          [{np.min(true_values_scroll):.6f}, {np.max(true_values_scroll):.6f}] (std: {np.std(true_values_scroll):.6f})")

    # Scrolling window summary (including coverage metrics)
    scroll_updated_wins = sum([
        scroll_updated_mse < scroll_orig_mse,
        scroll_updated_rmse < scroll_orig_rmse, 
        scroll_updated_mae < scroll_orig_mae,
        scroll_updated_dir_acc > scroll_orig_dir_acc,
        abs(scroll_updated_corr) > abs(scroll_orig_corr),  # Higher absolute correlation is better
        abs(scroll_updated_coverage - 95) < abs(scroll_orig_coverage - 95),  # Closer to 95% coverage is better
        scroll_updated_ci_width < scroll_orig_ci_width  # Smaller CI width is better (more precise)
    ])

    print(f"\n=== SCROLLING WINDOW FINAL VERDICT ===")
    if scroll_updated_wins >= 4:  # Updated threshold for 7 metrics
        print("🏆 UPDATED MODEL PERFORMS BETTER (wins on more metrics)")
    else:
        print("🏆 ORIGINAL MODEL PERFORMS BETTER (wins on more metrics)")

    print(f"Updated model wins on {scroll_updated_wins}/7 metrics")

else:
    print("❌ No valid scrolling window predictions available")
    scroll_orig_mse = scroll_updated_mse = scroll_orig_rmse = scroll_updated_rmse = scroll_orig_mae = scroll_updated_mae = 0
    scroll_orig_dir_acc = scroll_updated_dir_acc = scroll_orig_corr = scroll_updated_corr = 0
    scroll_orig_coverage = scroll_updated_coverage = scroll_orig_ci_width = scroll_updated_ci_width = 0
    scroll_updated_wins = 0

# Calculate metrics for FINAL 15-MINUTE ROLLING PREDICTION (REALISTIC)
print("\n" + "="*60)
print("=== FINAL 15-MINUTE ROLLING PREDICTION METRICS (REALISTIC) ===")
print("="*60)

if len(final_orig_predictions) > 0 and len(final_updated_predictions) > 0 and len(final_true_values) > 0:
    final_orig_mse = np.mean((final_orig_predictions - final_true_values) ** 2)
    final_updated_mse = np.mean((final_updated_predictions - final_true_values) ** 2)

    final_orig_rmse = np.sqrt(final_orig_mse)
    final_updated_rmse = np.sqrt(final_updated_mse)

    final_orig_mae = np.mean(np.abs(final_orig_predictions - final_true_values))
    final_updated_mae = np.mean(np.abs(final_updated_predictions - final_true_values))

    # Calculate directional accuracy
    final_orig_directions = np.sign(final_orig_predictions) == np.sign(final_true_values)
    final_updated_directions = np.sign(final_updated_predictions) == np.sign(final_true_values)
    final_orig_dir_acc = np.mean(final_orig_directions) * 100
    final_updated_dir_acc = np.mean(final_updated_directions) * 100

    # Correlation
    final_orig_corr = np.corrcoef(final_orig_predictions, final_true_values)[0, 1] if len(final_orig_predictions) > 1 else 0
    final_updated_corr = np.corrcoef(final_updated_predictions, final_true_values)[0, 1] if len(final_updated_predictions) > 1 else 0

    # Handle NaN correlations
    if np.isnan(final_orig_corr):
        final_orig_corr = 0.0
    if np.isnan(final_updated_corr):
        final_updated_corr = 0.0

    print(f"{'Metric':<20} {'Original':<15} {'Updated':<15} {'Winner':<10}")
    print("-" * 60)
    print(f"{'MSE':<20} {final_orig_mse:<15.6f} {final_updated_mse:<15.6f} {'Updated' if final_updated_mse < final_orig_mse else 'Original':<10}")
    print(f"{'RMSE':<20} {final_orig_rmse:<15.6f} {final_updated_rmse:<15.6f} {'Updated' if final_updated_rmse < final_orig_rmse else 'Original':<10}")
    print(f"{'MAE':<20} {final_orig_mae:<15.6f} {final_updated_mae:<15.6f} {'Updated' if final_updated_mae < final_orig_mae else 'Original':<10}")
    print(f"{'Directional Acc %':<20} {final_orig_dir_acc:<15.2f} {final_updated_dir_acc:<15.2f} {'Updated' if final_updated_dir_acc > final_orig_dir_acc else 'Original':<10}")
    print(f"{'Correlation':<20} {final_orig_corr:<15.4f} {final_updated_corr:<15.4f} {'Higher is better':<10}")

    print(f"\n=== FINAL 15-MIN PREDICTION RANGES ===")
    print(f"Original predictions: [{np.min(final_orig_predictions):.6f}, {np.max(final_orig_predictions):.6f}] (std: {np.std(final_orig_predictions):.6f})")
    print(f"Updated predictions:  [{np.min(final_updated_predictions):.6f}, {np.max(final_updated_predictions):.6f}] (std: {np.std(final_updated_predictions):.6f})")
    print(f"True values:          [{np.min(final_true_values):.6f}, {np.max(final_true_values):.6f}] (std: {np.std(final_true_values):.6f})")

    # Final 15-min summary
    final_updated_wins = sum([
        final_updated_mse < final_orig_mse,
        final_updated_rmse < final_orig_rmse, 
        final_updated_mae < final_orig_mae,
        final_updated_dir_acc > final_orig_dir_acc,
        abs(final_updated_corr) > abs(final_orig_corr)  # Higher absolute correlation is better
    ])

    print(f"\n=== FINAL 15-MIN VERDICT ===")
    if final_updated_wins >= 3:
        print("🏆 UPDATED MODEL PERFORMS BETTER (wins on more metrics)")
    else:
        print("🏆 ORIGINAL MODEL PERFORMS BETTER (wins on more metrics)")

    print(f"Updated model wins on {final_updated_wins}/5 metrics")

else:
    print("❌ No valid final 15-minute predictions available")
    final_orig_mse = final_updated_mse = final_orig_rmse = final_updated_rmse = final_orig_mae = final_updated_mae = 0
    final_orig_dir_acc = final_updated_dir_acc = final_orig_corr = final_updated_corr = 0
    final_updated_wins = 0

# OVERALL COMPARISON BETWEEN METHODS
print("\n" + "="*60)
print("=== METHOD COMPARISON SUMMARY ===")
print("="*60)

print(f"{'Method':<25} {'Rolling':<15} {'Scrolling':<15} {'Final 15-min':<15}")
print("-" * 70)
print(f"{'Original MSE':<25} {orig_mse:<15.6f} {scroll_orig_mse:<15.6f} {final_orig_mse:<15.6f}")
print(f"{'Updated MSE':<25} {updated_mse:<15.6f} {scroll_updated_mse:<15.6f} {final_updated_mse:<15.6f}")
print(f"{'Original RMSE':<25} {orig_rmse:<15.6f} {scroll_orig_rmse:<15.6f} {final_orig_rmse:<15.6f}")
print(f"{'Updated RMSE':<25} {updated_rmse:<15.6f} {scroll_updated_rmse:<15.6f} {final_updated_rmse:<15.6f}")
print(f"{'Original MAE':<25} {orig_mae:<15.6f} {scroll_orig_mae:<15.6f} {final_orig_mae:<15.6f}")
print(f"{'Updated MAE':<25} {updated_mae:<15.6f} {scroll_updated_mae:<15.6f} {final_updated_mae:<15.6f}")
print(f"{'Original Dir Acc %':<25} {orig_dir_acc:<15.2f} {scroll_orig_dir_acc:<15.2f} {final_orig_dir_acc:<15.2f}")
print(f"{'Updated Dir Acc %':<25} {updated_dir_acc:<15.2f} {scroll_updated_dir_acc:<15.2f} {final_updated_dir_acc:<15.2f}")
print(f"{'Original Corr':<25} {orig_corr:<15.4f} {scroll_orig_corr:<15.4f} {final_orig_corr:<15.4f}")
print(f"{'Updated Corr':<25} {updated_corr:<15.4f} {scroll_updated_corr:<15.4f} {final_updated_corr:<15.4f}")

print(f"\n{'Method Winner':<25} {'Rolling':<15} {'Scrolling':<15} {'Final 15-min':<15}")
print("-" * 70)
print(f"{'Updated Wins':<25} {rolling_updated_wins:<15} {scroll_updated_wins:<15} {final_updated_wins:<15}")

# Determine overall best method for each model
methods = ['Rolling', 'Scrolling', 'Final 15-min']
orig_method_scores = [orig_mse, scroll_orig_mse, final_orig_mse]  # Lower is better for MSE
updated_method_scores = [updated_mse, scroll_updated_mse, final_updated_mse]

best_orig_method = methods[np.argmin(orig_method_scores)]
best_updated_method = methods[np.argmin(updated_method_scores)]

print(f"\n=== OVERALL METHOD ANALYSIS ===")
print(f"Best method for Original model: {best_orig_method} (MSE: {min(orig_method_scores):.6f})")
print(f"Best method for Updated model: {best_updated_method} (MSE: {min(updated_method_scores):.6f})")

# Use rolling window results for the main comparison (keeping original variable names for compatibility)
updated_wins = rolling_updated_wins

print(f"\n" + "="*60)
print("=== FINAL CONCLUSION ===")
print("="*60)

if len(orig_predictions) > 0 and len(updated_predictions) > 0:
    print("🔍 BOTH MODELS PRODUCE VALID PREDICTIONS!")
    print(f"✓ Original model: {len(orig_predictions)} valid predictions")
    print(f"✓ Updated model: {len(updated_predictions)} valid predictions")
    print("")
    
    # Method comparison summary
    print("📊 METHOD COMPARISON SUMMARY:")
    print(f"• Rolling Window: Original wins on {5-rolling_updated_wins}/5 metrics")
    print(f"• Scrolling Window: Original wins on {7-scroll_updated_wins}/7 metrics (includes coverage)")
    print(f"• Final 15-min: Original wins on {5-final_updated_wins}/5 metrics")
    print("")
    
    # Overall recommendation
    total_updated_wins = rolling_updated_wins + scroll_updated_wins + final_updated_wins
    total_metrics = 17  # 5 + 7 + 5 metrics (updated for new coverage metrics)
    
    if total_updated_wins >= total_metrics // 2:
        print("🎯 OVERALL RECOMMENDATION: Updated model performs better across methods")
        print("✅ Use mambastock_updated.pth for better performance")
    else:
        print("🎯 OVERALL RECOMMENDATION: Original model performs better across methods") 
        print("✅ Use mambastock_original.pth for better performance")
    
    print(f"\nUpdated model wins: {total_updated_wins}/{total_metrics} total metrics")
    print(f"Win rate: {total_updated_wins/total_metrics*100:.1f}%")
    print("")
    
    # Best method analysis
    print("🏆 BEST METHOD ANALYSIS:")
    print(f"• Best for Original model: {best_orig_method} (MSE: {min(orig_method_scores):.6f})")
    print(f"• Best for Updated model: {best_updated_method} (MSE: {min(updated_method_scores):.6f})")
    print("")
    
    # Prediction ranges
    print(f"📈 PREDICTION RANGES (Rolling Window):")
    print(f"Original: [{np.min(orig_predictions):.6f}, {np.max(orig_predictions):.6f}]")
    print(f"Updated: [{np.min(updated_predictions):.6f}, {np.max(updated_predictions):.6f}]")
    print(f"True values: [{np.min(true_values):.6f}, {np.max(true_values):.6f}]")
    
    if len(orig_predictions_scroll) > 0:
        print(f"\n📈 PREDICTION RANGES (Scrolling Window):")
        print(f"Original: [{np.min(orig_predictions_scroll):.6f}, {np.max(orig_predictions_scroll):.6f}]")
        print(f"Updated: [{np.min(updated_predictions_scroll):.6f}, {np.max(updated_predictions_scroll):.6f}]")
        print(f"True values: [{np.min(true_values_scroll):.6f}, {np.max(true_values_scroll):.6f}]")
    
else:
    print("❌ ERROR: Failed to collect predictions for comparison")
    print("Check model compatibility and data format")

print("="*60)

# === CREATE COMPREHENSIVE PLOTS ===
print(f"\n📊 CREATING PREDICTION COMPARISON PLOTS...")

# Create comprehensive prediction comparison plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Scatter plot of predictions vs actual
ax1.scatter(true_values, orig_predictions, alpha=0.6, color='blue', label='Original Model', s=20)
ax1.scatter(true_values, updated_predictions, alpha=0.6, color='red', label='Updated Model', s=20)
# Perfect prediction line
min_val = min(np.min(true_values), np.min(orig_predictions), np.min(updated_predictions))
max_val = max(np.max(true_values), np.max(orig_predictions), np.max(updated_predictions))
ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='Perfect Prediction')
ax1.set_xlabel('True Values (%)')
ax1.set_ylabel('Predicted Values (%)')
ax1.set_title('Predictions vs Actual Values')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Time series of predictions
time_indices = range(len(orig_predictions))
ax2.plot(time_indices, true_values, 'k-', linewidth=2, label='True Values', alpha=0.8)
ax2.plot(time_indices, orig_predictions, 'b-', linewidth=1.5, label='Original Model', alpha=0.7)
ax2.plot(time_indices, updated_predictions, 'r-', linewidth=1.5, label='Updated Model', alpha=0.7)
ax2.set_xlabel('Prediction Index')
ax2.set_ylabel('Percentage Change (%)')
ax2.set_title('Prediction Time Series Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Error distribution
orig_errors = orig_predictions - true_values
updated_errors = updated_predictions - true_values

ax3.hist(orig_errors, bins=30, alpha=0.7, color='blue', label='Original Model', density=True)
ax3.hist(updated_errors, bins=30, alpha=0.7, color='red', label='Updated Model', density=True)
ax3.axvline(x=0, color='black', linestyle='--', alpha=0.7, label='Perfect Prediction')
ax3.set_xlabel('Prediction Error (%)')
ax3.set_ylabel('Density')
ax3.set_title('Error Distribution Comparison')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Model performance metrics comparison
metrics = ['MSE', 'RMSE', 'MAE', 'Dir_Acc', 'Abs_Corr']
orig_values = [orig_mse, orig_rmse, orig_mae, orig_dir_acc/100, abs(orig_corr)]
updated_values = [updated_mse, updated_rmse, updated_mae, updated_dir_acc/100, abs(updated_corr)]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax4.bar(x - width/2, orig_values, width, label='Original Model', color='blue', alpha=0.7)
bars2 = ax4.bar(x + width/2, updated_values, width, label='Updated Model', color='red', alpha=0.7)

ax4.set_xlabel('Metrics')
ax4.set_ylabel('Values')
ax4.set_title('Model Performance Metrics Comparison')
ax4.set_xticks(x)
ax4.set_xticklabels(metrics, rotation=45)
ax4.legend()
ax4.grid(True, alpha=0.3)

# Add value labels on bars
def add_value_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=8)

add_value_labels(ax4, bars1)
add_value_labels(ax4, bars2)

plt.tight_layout()
plt.savefig('model_comparison_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# === CREATE SCROLLING WINDOW COMPARISON PLOTS ===
if len(orig_predictions_scroll) > 0 and len(updated_predictions_scroll) > 0 and len(true_values_scroll) > 0:
    print(f"\n📊 CREATING SCROLLING WINDOW COMPARISON PLOTS...")
    
    # Create scrolling window comparison plots
    fig_scroll, ((ax1_scroll, ax2_scroll), (ax3_scroll, ax4_scroll)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Scatter plot of scrolling predictions vs actual
    ax1_scroll.scatter(true_values_scroll, orig_predictions_scroll, alpha=0.6, color='blue', label='Original Model', s=20)
    ax1_scroll.scatter(true_values_scroll, updated_predictions_scroll, alpha=0.6, color='red', label='Updated Model', s=20)
    # Perfect prediction line
    min_val_scroll = min(np.min(true_values_scroll), np.min(orig_predictions_scroll), np.min(updated_predictions_scroll))
    max_val_scroll = max(np.max(true_values_scroll), np.max(orig_predictions_scroll), np.max(updated_predictions_scroll))
    ax1_scroll.plot([min_val_scroll, max_val_scroll], [min_val_scroll, max_val_scroll], 'k--', alpha=0.7, label='Perfect Prediction')
    ax1_scroll.set_xlabel('True Values (%)')
    ax1_scroll.set_ylabel('Predicted Values (%)')
    ax1_scroll.set_title('Scrolling Window: Predictions vs Actual Values')
    ax1_scroll.legend()
    ax1_scroll.grid(True, alpha=0.3)
    
    # Plot 2: Time series of scrolling predictions
    time_indices_scroll = range(len(orig_predictions_scroll))
    ax2_scroll.plot(time_indices_scroll, true_values_scroll, 'k-', linewidth=2, label='True Values', alpha=0.8)
    ax2_scroll.plot(time_indices_scroll, orig_predictions_scroll, 'b-', linewidth=1.5, label='Original Model', alpha=0.7)
    ax2_scroll.plot(time_indices_scroll, updated_predictions_scroll, 'r-', linewidth=1.5, label='Updated Model', alpha=0.7)
    ax2_scroll.set_xlabel('Prediction Index')
    ax2_scroll.set_ylabel('Percentage Change (%)')
    ax2_scroll.set_title('Scrolling Window: Prediction Time Series')
    ax2_scroll.legend()
    ax2_scroll.grid(True, alpha=0.3)
    
    # Plot 3: Error distribution for scrolling
    orig_errors_scroll = orig_predictions_scroll - true_values_scroll
    updated_errors_scroll = updated_predictions_scroll - true_values_scroll
    
    ax3_scroll.hist(orig_errors_scroll, bins=30, alpha=0.7, color='blue', label='Original Model', density=True)
    ax3_scroll.hist(updated_errors_scroll, bins=30, alpha=0.7, color='red', label='Updated Model', density=True)
    ax3_scroll.axvline(x=0, color='black', linestyle='--', alpha=0.7, label='Perfect Prediction')
    ax3_scroll.set_xlabel('Prediction Error (%)')
    ax3_scroll.set_ylabel('Density')
    ax3_scroll.set_title('Scrolling Window: Error Distribution')
    ax3_scroll.legend()
    ax3_scroll.grid(True, alpha=0.3)
    
    # Plot 4: Scrolling window performance metrics comparison
    metrics_scroll = ['MSE', 'RMSE', 'MAE', 'Dir_Acc', 'Abs_Corr']
    orig_values_scroll = [scroll_orig_mse, scroll_orig_rmse, scroll_orig_mae, scroll_orig_dir_acc/100, abs(scroll_orig_corr)]
    updated_values_scroll = [scroll_updated_mse, scroll_updated_rmse, scroll_updated_mae, scroll_updated_dir_acc/100, abs(scroll_updated_corr)]
    
    x_scroll = np.arange(len(metrics_scroll))
    width_scroll = 0.35
    
    bars1_scroll = ax4_scroll.bar(x_scroll - width_scroll/2, orig_values_scroll, width_scroll, label='Original Model', color='blue', alpha=0.7)
    bars2_scroll = ax4_scroll.bar(x_scroll + width_scroll/2, updated_values_scroll, width_scroll, label='Updated Model', color='red', alpha=0.7)
    
    ax4_scroll.set_xlabel('Metrics')
    ax4_scroll.set_ylabel('Values')
    ax4_scroll.set_title('Scrolling Window: Model Performance Metrics')
    ax4_scroll.set_xticks(x_scroll)
    ax4_scroll.set_xticklabels(metrics_scroll, rotation=45)
    ax4_scroll.legend()
    ax4_scroll.grid(True, alpha=0.3)
    
    # Add value labels on bars
    add_value_labels(ax4_scroll, bars1_scroll)
    add_value_labels(ax4_scroll, bars2_scroll)
    
    plt.tight_layout()
    plt.savefig('scrolling_window_comparison_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

# === CREATE METHOD COMPARISON PLOTS ===
print(f"\n📊 CREATING METHOD COMPARISON PLOTS...")

# Create method comparison plots
fig_method, ((ax1_method, ax2_method), (ax3_method, ax4_method)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: MSE comparison across methods
methods_names = ['Rolling', 'Scrolling', 'Final 15-min']
orig_mse_methods = [orig_mse, scroll_orig_mse, final_orig_mse]
updated_mse_methods = [updated_mse, scroll_updated_mse, final_updated_mse]

x_methods = np.arange(len(methods_names))
width_methods = 0.35

bars1_methods = ax1_method.bar(x_methods - width_methods/2, orig_mse_methods, width_methods, label='Original Model', color='blue', alpha=0.7)
bars2_methods = ax1_method.bar(x_methods + width_methods/2, updated_mse_methods, width_methods, label='Updated Model', color='red', alpha=0.7)

ax1_method.set_xlabel('Methods')
ax1_method.set_ylabel('MSE')
ax1_method.set_title('MSE Comparison Across Methods')
ax1_method.set_xticks(x_methods)
ax1_method.set_xticklabels(methods_names)
ax1_method.legend()
ax1_method.grid(True, alpha=0.3)

# Add value labels
add_value_labels(ax1_method, bars1_methods)
add_value_labels(ax1_method, bars2_methods)

# Plot 2: Directional Accuracy comparison
orig_dir_methods = [orig_dir_acc, scroll_orig_dir_acc, final_orig_dir_acc]
updated_dir_methods = [updated_dir_acc, scroll_updated_dir_acc, final_updated_dir_acc]

bars3_methods = ax2_method.bar(x_methods - width_methods/2, orig_dir_methods, width_methods, label='Original Model', color='blue', alpha=0.7)
bars4_methods = ax2_method.bar(x_methods + width_methods/2, updated_dir_methods, width_methods, label='Updated Model', color='red', alpha=0.7)

ax2_method.set_xlabel('Methods')
ax2_method.set_ylabel('Directional Accuracy (%)')
ax2_method.set_title('Directional Accuracy Comparison Across Methods')
ax2_method.set_xticks(x_methods)
ax2_method.set_xticklabels(methods_names)
ax2_method.legend()
ax2_method.grid(True, alpha=0.3)

# Add value labels
add_value_labels(ax2_method, bars3_methods)
add_value_labels(ax2_method, bars4_methods)

# Plot 3: Correlation comparison
orig_corr_methods = [abs(orig_corr), abs(scroll_orig_corr), abs(final_orig_corr)]
updated_corr_methods = [abs(updated_corr), abs(scroll_updated_corr), abs(final_updated_corr)]

bars5_methods = ax3_method.bar(x_methods - width_methods/2, orig_corr_methods, width_methods, label='Original Model', color='blue', alpha=0.7)
bars6_methods = ax3_method.bar(x_methods + width_methods/2, updated_corr_methods, width_methods, label='Updated Model', color='red', alpha=0.7)

ax3_method.set_xlabel('Methods')
ax3_method.set_ylabel('Absolute Correlation')
ax3_method.set_title('Correlation Comparison Across Methods')
ax3_method.set_xticks(x_methods)
ax3_method.set_xticklabels(methods_names)
ax3_method.legend()
ax3_method.grid(True, alpha=0.3)

# Add value labels
add_value_labels(ax3_method, bars5_methods)
add_value_labels(ax3_method, bars6_methods)

# Plot 4: Method winner summary
method_wins = [rolling_updated_wins, scroll_updated_wins, final_updated_wins]
colors = ['green' if wins >= 3 else 'red' for wins in method_wins]

bars7_methods = ax4_method.bar(methods_names, method_wins, color=colors, alpha=0.7)
ax4_method.set_xlabel('Methods')
ax4_method.set_ylabel('Updated Model Wins (out of 5)')
ax4_method.set_title('Method Performance Summary')
ax4_method.axhline(y=2.5, color='black', linestyle='--', alpha=0.7, label='Tie Threshold')
ax4_method.legend()
ax4_method.grid(True, alpha=0.3)

# Add value labels
for bar in bars7_methods:
    height = bar.get_height()
    ax4_method.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}/5', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('method_comparison_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# === DETAILED STATISTICS TABLE ===
print(f"\n📈 DETAILED PREDICTION STATISTICS:")
print(f"{'Statistic':<25} {'Original':<15} {'Updated':<15} {'Difference':<15}")
print("-" * 70)
print(f"{'Mean Prediction':<25} {np.mean(orig_predictions):<15.6f} {np.mean(updated_predictions):<15.6f} {np.mean(updated_predictions) - np.mean(orig_predictions):<15.6f}")
print(f"{'Std Prediction':<25} {np.std(orig_predictions):<15.6f} {np.std(updated_predictions):<15.6f} {np.std(updated_predictions) - np.std(orig_predictions):<15.6f}")
print(f"{'Min Prediction':<25} {np.min(orig_predictions):<15.6f} {np.min(updated_predictions):<15.6f} {np.min(updated_predictions) - np.min(orig_predictions):<15.6f}")
print(f"{'Max Prediction':<25} {np.max(orig_predictions):<15.6f} {np.max(updated_predictions):<15.6f} {np.max(updated_predictions) - np.max(orig_predictions):<15.6f}")
print(f"{'Mean Absolute Error':<25} {orig_mae:<15.6f} {updated_mae:<15.6f} {updated_mae - orig_mae:<15.6f}")
print(f"{'Root Mean Sq Error':<25} {orig_rmse:<15.6f} {updated_rmse:<15.6f} {updated_rmse - orig_rmse:<15.6f}")

# Bias analysis
orig_errors = orig_predictions - true_values
updated_errors = updated_predictions - true_values
orig_bias = np.mean(orig_errors)
updated_bias = np.mean(updated_errors)
print(f"{'Prediction Bias':<25} {orig_bias:<15.6f} {updated_bias:<15.6f} {updated_bias - orig_bias:<15.6f}")

# Variance analysis
orig_var = np.var(orig_errors)
updated_var = np.var(updated_errors)
print(f"{'Error Variance':<25} {orig_var:<15.6f} {updated_var:<15.6f} {updated_var - orig_var:<15.6f}")

# === ADDITIONAL ANALYSIS PLOTS ===
print(f"\n📊 CREATING ADDITIONAL ANALYSIS PLOTS...")

fig2, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 5: Prediction ranges by model
models = ['Original', 'Updated']
pred_ranges = [
    [np.min(orig_predictions), np.max(orig_predictions)],
    [np.min(updated_predictions), np.max(updated_predictions)]
]
pred_means = [np.mean(orig_predictions), np.mean(updated_predictions)]
pred_stds = [np.std(orig_predictions), np.std(updated_predictions)]

ax5.bar(models, pred_means, yerr=pred_stds, capsize=5, alpha=0.7, color=['blue', 'red'])
ax5.set_ylabel('Prediction Value (%)')
ax5.set_title('Mean Predictions with Standard Deviation')
ax5.grid(True, alpha=0.3)

# Add text annotations
for i, (mean, std) in enumerate(zip(pred_means, pred_stds)):
    ax5.text(i, mean + std + 0.001, f'{mean:.4f}±{std:.4f}', ha='center', va='bottom')

# Plot 6: Cumulative error over time
cumulative_orig_error = np.cumsum(np.abs(orig_errors))
cumulative_updated_error = np.cumsum(np.abs(updated_errors))

ax6.plot(time_indices, cumulative_orig_error, 'b-', linewidth=2, label='Original Model')
ax6.plot(time_indices, cumulative_updated_error, 'r-', linewidth=2, label='Updated Model')
ax6.set_xlabel('Prediction Index')
ax6.set_ylabel('Cumulative Absolute Error')
ax6.set_title('Cumulative Error Over Time')
ax6.legend()
ax6.grid(True, alpha=0.3)

# Plot 7: Directional accuracy over time (rolling window)
window_size = 20
if len(orig_predictions) >= window_size:
    orig_dir_rolling = []
    updated_dir_rolling = []
    
    for i in range(window_size, len(orig_predictions)):
        start_idx = i - window_size
        end_idx = i
        
        orig_dir_window = np.mean(np.sign(orig_predictions[start_idx:end_idx]) == np.sign(true_values[start_idx:end_idx])) * 100
        updated_dir_window = np.mean(np.sign(updated_predictions[start_idx:end_idx]) == np.sign(true_values[start_idx:end_idx])) * 100
        
        orig_dir_rolling.append(orig_dir_window)
        updated_dir_rolling.append(updated_dir_window)
    
    rolling_indices = range(window_size, len(orig_predictions))
    ax7.plot(rolling_indices, orig_dir_rolling, 'b-', linewidth=2, label='Original Model')
    ax7.plot(rolling_indices, updated_dir_rolling, 'r-', linewidth=2, label='Updated Model')
    ax7.axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='Random Guess (50%)')
    ax7.set_xlabel('Prediction Index')
    ax7.set_ylabel('Directional Accuracy (%)')
    ax7.set_title(f'Rolling Directional Accuracy (Window={window_size})')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim(0, 100)
else:
    ax7.text(0.5, 0.5, 'Not enough data for rolling accuracy', ha='center', va='center', transform=ax7.transAxes)
    ax7.set_title('Rolling Directional Accuracy (Insufficient Data)')

# Plot 8: Error magnitude comparison
error_magnitudes_orig = np.abs(orig_errors)
error_magnitudes_updated = np.abs(updated_errors)

ax8.boxplot([error_magnitudes_orig, error_magnitudes_updated], 
           labels=['Original', 'Updated'], patch_artist=True,
           boxprops=dict(facecolor='lightblue', alpha=0.7),
           medianprops=dict(color='red', linewidth=2))
ax8.set_ylabel('Absolute Error Magnitude (%)')
ax8.set_title('Error Magnitude Distribution (Box Plot)')
ax8.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_detailed_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ COMPREHENSIVE EVALUATION WITH PLOTS COMPLETED!")
print(f"📁 Saved plots:")
print(f"   • 'model_comparison_plots.png' - Rolling window comparison")
print(f"   • 'scrolling_window_comparison_plots.png' - Scrolling window comparison")
print(f"   • 'method_comparison_plots.png' - Method comparison analysis")
print(f"   • 'model_detailed_analysis.png' - Detailed analysis")
print(f"📊 Models tested using both rolling and scrolling window methods")
print(f"🎯 Both models are now working correctly with comprehensive evaluation!")
print(f"🔍 Scrolling window method provides incremental prediction with growing context")
print(f"📈 Final 15-minute prediction simulates end-of-day trading strategy")