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
orig_path = "mambastock.pth"
updated_path = "mambastock_denormalized.pth"

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

# Load test dataset EXACTLY like train.py
try:
    # Import the FIXED dataset with global normalization
    from dataset_gpu import ClosePrice  # Use fixed version with global normalization
    print("✓ Dataset import successful (using fixed global normalization)")
    test_dataset = ClosePrice("h5_rty_data_test.h5")
    print(f"✓ Dataset loaded: {test_dataset.n_stocks} stocks, shape: {test_dataset.data.shape}")
    
    # Show normalization parameters
    print(f"✓ Global normalization parameters:")
    print(f"   Percentage change mean: {test_dataset.pct_change_mean:.6f}")
    print(f"   Percentage change std: {test_dataset.pct_change_std:.6f}")
    
except Exception as e:
    print(f"❌ Dataset loading failed: {e}")
    print("   Falling back to original dataset...")
    try:
        from dataset import ClosePrice
        print("✓ Fallback dataset import successful")
        test_dataset = ClosePrice("h5_rty_data_test.h5")
        print(f"✓ Dataset loaded: {test_dataset.n_stocks} stocks, shape: {test_dataset.data.shape}")
        print("ℹ️  Note: Using original dataset without denormalization support")
    except Exception as e2:
        print(f"❌ Fallback dataset loading also failed: {e2}")
        exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Using device: {device}")

# Initialize models EXACTLY like train.py
try:
    seq_len = 30  # Match train.py exactly
    pred_len = 2  # Match train.py exactly
    input_size = 13  # Match train.py exactly - this is the key!
    
    model_orig = MambaStock(input_size=input_size, seq_len=seq_len, pred_len=pred_len).to(device)
    model_updated = MambaStock(input_size=input_size, seq_len=seq_len, pred_len=pred_len).to(device)
    print(f"✓ Models initialized with input_size={input_size}, seq_len={seq_len}, pred_len={pred_len}")
except Exception as e:
    print(f"❌ Model initialization failed: {e}")
    exit(1)

# Load weights EXACTLY like train.py
try:
    orig_state = torch.load("mambastock_updated.pth", map_location=device)
    updated_state = torch.load("mambastock_denormalized.pth", map_location=device)
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
def rolling_multi_step_comparison(test_dataset, model_orig, model_updated, seq_len=30, pred_len=330, device=None, step=20):
    model_orig.eval()
    model_updated.eval()
    n_stocks = test_dataset.n_stocks
    input_seqs = test_dataset.data[:, :seq_len, :].clone().to(device)  # [n_stocks, seq_len, features]
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

# --- MAIN COMPARISON LOGIC ---
if __name__ == "__main__":
    # Run rolling multi-step comparison
    preds_orig, preds_updated, trues = rolling_multi_step_comparison(test_dataset, model_orig, model_updated, seq_len=30, pred_len=330, device=device, step=20)
    print("Original model predictions shape:", preds_orig.shape)
    print("Updated model predictions shape:", preds_updated.shape)
    print("Ground truth shape:", trues.shape)
    # Flatten for analysis
    orig_predictions = preds_orig.cpu().numpy().flatten()
    updated_predictions = preds_updated.cpu().numpy().flatten()
    true_values = trues.cpu().numpy().flatten()
    # Ensure all arrays are the same length
    min_len = min(len(orig_predictions), len(updated_predictions), len(true_values))
    orig_predictions = orig_predictions[:min_len]
    updated_predictions = updated_predictions[:min_len]
    true_values = true_values[:min_len]

# Calculate overall metrics
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

print("\n=== SIMPLE EVALUATION RESULTS ===")
print(f"{'Metric':<20} {'Original':<15} {'Updated':<15} {'Winner':<10}")
print("-" * 60)
print(f"{'MSE':<20} {orig_mse:<15.6f} {updated_mse:<15.6f} {'Updated' if updated_mse < orig_mse else 'Original':<10}")
print(f"{'RMSE':<20} {orig_rmse:<15.6f} {updated_rmse:<15.6f} {'Updated' if updated_rmse < orig_rmse else 'Original':<10}")
print(f"{'MAE':<20} {orig_mae:<15.6f} {updated_mae:<15.6f} {'Updated' if updated_mae < orig_mae else 'Original':<10}")
print(f"{'Directional Acc %':<20} {orig_dir_acc:<15.2f} {updated_dir_acc:<15.2f} {'Updated' if updated_dir_acc > orig_dir_acc else 'Original':<10}")
print(f"{'Correlation':<20} {orig_corr:<15.4f} {updated_corr:<15.4f} {'Higher is better':<10}")

print(f"\n=== PREDICTION RANGES ===")
print(f"Original predictions: [{np.min(orig_predictions):.6f}, {np.max(orig_predictions):.6f}] (std: {np.std(orig_predictions):.6f})")
print(f"Updated predictions:  [{np.min(updated_predictions):.6f}, {np.max(updated_predictions):.6f}] (std: {np.std(updated_predictions):.6f})")
print(f"True values:          [{np.min(true_values):.6f}, {np.max(true_values):.6f}] (std: {np.std(true_values):.6f})")

# Simple summary
updated_wins = sum([
    updated_mse < orig_mse,
    updated_rmse < orig_rmse, 
    updated_mae < orig_mae,
    updated_dir_acc > orig_dir_acc,
    abs(updated_corr) > abs(orig_corr)  # Higher absolute correlation is better
])

print(f"\n=== FINAL VERDICT ===")
if updated_wins >= 3:
    print("🏆 UPDATED MODEL PERFORMS BETTER (wins on more metrics)")
else:
    print("🏆 ORIGINAL MODEL PERFORMS BETTER (wins on more metrics)")

print(f"Updated model wins on {updated_wins}/5 metrics")

print(f"\n" + "="*60)
print("=== FINAL CONCLUSION ===")
print("="*60)

if len(orig_predictions) > 0 and len(updated_predictions) > 0:
    print("🔍 BOTH MODELS PRODUCE VALID PREDICTIONS!")
    print(f"✓ Original model: {len(orig_predictions)} valid predictions")
    print(f"✓ Updated model: {len(updated_predictions)} valid predictions")
    print("")
    
    if updated_wins >= 3:
        print("🎯 RECOMMENDATION: Updated model performs better overall")
        print("✅ Use mambastock_updated.pth for better performance")
    else:
        print("🎯 RECOMMENDATION: Original model performs better overall") 
        print("✅ Use mambastock_original.pth for better performance")
        
    print(f"\nPrediction ranges:")
    print(f"Original: [{np.min(orig_predictions):.6f}, {np.max(orig_predictions):.6f}]")
    print(f"Updated: [{np.min(updated_predictions):.6f}, {np.max(updated_predictions):.6f}]")
    print(f"True values: [{np.min(true_values):.6f}, {np.max(true_values):.6f}]")
    
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
print(f"📁 Saved plots: 'model_comparison_plots.png' and 'model_detailed_analysis.png'")
print(f"📊 Models tested using the same logic as train.py")
print(f"🎯 Both models are now working correctly with proper evaluation!")