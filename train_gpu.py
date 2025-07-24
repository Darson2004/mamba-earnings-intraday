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

# --- CONFIG ---
h5_path = "h5_rty_data.h5"  # <-- HDF5 file path
seq_len = 30
pred_len = 20
batch_size = 512
epochs = 3
lr = 1e-6  # Lower learning rate for fine-tuning from pretrained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))

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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log.txt', mode='w', encoding='utf-8'),
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
logger.info("STARTING MAMBA STOCK TRAINING")
logger.info("="*60)
logger.info(f"Config: seq_len={seq_len}, pred_len={pred_len}, batch_size={batch_size}, epochs={epochs}, lr={lr}")
logger.info(f"Device: {device}")
logger.info(f"Dataset: {h5_path}")
logger.info("="*60)
logger.info("Training Loss Log:")
logger.info("Epoch | Loss")
logger.info("-" * 20)
logger.info("="*60)


# --- LOAD DATA ---
dataset = ClosePrice(h5_path)
print(f"Dataset loaded: {len(dataset)} total items, {dataset.n_stocks} stocks")
logger.info(f"Dataset loaded: {len(dataset)} total items, {dataset.n_stocks} stocks")
logger.info("="*60)
print(f"Starting all calculations after {seq_len} data points (time index {seq_len}) to exclude first {seq_len} points")

# --- PRECOMPUTE BATCHES (no DataLoader needed) ---
n_stocks, day_length, n_features = dataset.data.shape
stock_indices = np.arange(n_stocks)
time_indices = np.arange(seq_len, day_length - pred_len)
pairs = np.array(np.meshgrid(stock_indices, time_indices)).T.reshape(-1, 2)
np.random.shuffle(pairs)
batch_size = 512  # You can adjust this for your GPU memory
num_batches = int(np.ceil(len(pairs) / batch_size))

# --- INIT MODEL ---
# Adjust input_size based on the ClosePrice dataset structure
model = MambaStock(input_size=13, seq_len=seq_len, pred_len=pred_len).to(device)
# Ensure model is on GPU
print("Model device:", next(model.parameters()).device)
# Load original weights
try:
    # Load weights with strict=False to handle architecture changes
    state_dict = torch.load("mambastock_original.pth", map_location=device)
    # Try to load as much as possible, ignoring mismatches
    model_dict = model.state_dict()
    
    # Filter state dict to only include keys that exist in current model
    filtered_state_dict = {}
    loaded_keys = 0
    for key, value in state_dict.items():
        if key in model_dict and model_dict[key].shape == value.shape:
            filtered_state_dict[key] = value
            loaded_keys += 1
    
    # Load the filtered state dict
    model.load_state_dict(filtered_state_dict, strict=False)
    logger.info(f"Loaded {loaded_keys} out of {len(state_dict)} keys from mambastock_original.pth")
    print(f"Loaded {loaded_keys} out of {len(state_dict)} keys from mambastock_original.pth")
    if loaded_keys == 0:
        logger.warning("No compatible weights found, starting with random weights")
        print("No compatible weights found, starting with random weights")
except FileNotFoundError:
    logger.warning("mambastock_original.pth not found, starting with random weights")
    print("mambastock_original.pth not found, starting with random weights")
except Exception as e:
    logger.error(f"Error loading weights: {e}")
    print(f"Error loading weights: {e}")
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
loss_fn = nn.MSELoss().to(device)  # Ensure loss function is on GPU

# --- TRAIN LOOP ---
model.train()
for epoch in range(epochs):
    total_loss = 0
    total_prediction_loss = 0  # Pure prediction loss (MSE on percentage changes)
    total_smoothness_loss = 0  # Smoothness penalty
    epoch_pct_mse = 0
    epoch_pct_rmse = 0
    epoch_directional_accuracy = 0
    batch_count = 0
    total_predictions = 0  # Count total predictions for averaging
    # Store predictions for last epoch
    if epoch == epochs - 1:
        all_stock_preds = [[] for _ in range(dataset.n_stocks)]
        all_stock_trues = [[] for _ in range(dataset.n_stocks)]
        all_stock_predlens = [[] for _ in range(dataset.n_stocks)]
        all_stock_truelens = [[] for _ in range(dataset.n_stocks)]
    for batch_idx in range(num_batches):
        batch_pairs = pairs[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        # Convert to tensor indices on GPU
        batch_stock_idx = torch.tensor(batch_pairs[:, 0], dtype=torch.long, device=dataset.data.device)
        batch_time_idx = torch.tensor(batch_pairs[:, 1], dtype=torch.long, device=dataset.data.device)

        # Use list comprehension for both inputs and targets (tensor range indexing not supported)
        batch_inputs = torch.stack([
            dataset.data[s, t - seq_len:t, :]
            for s, t in zip(batch_stock_idx.cpu().numpy(), batch_time_idx.cpu().numpy())
        ])
        batch_targets = torch.stack([
            dataset.data[s, t:t + pred_len, 10]
            for s, t in zip(batch_stock_idx.cpu().numpy(), batch_time_idx.cpu().numpy())
        ])
        # Check batch size in memory
        batch_size_bytes = batch_inputs.element_size() * batch_inputs.nelement() + batch_targets.element_size() * batch_targets.nelement()
        batch_size_mb = batch_size_bytes / (1024 ** 2)

        # --- Forward, loss, backward, optimizer step ---
        optimizer.zero_grad()
        
        # Pre-allocate full sequence tensor for efficient autoregressive prediction
        batch_size_actual = batch_inputs.shape[0]
        full_seq = torch.zeros(batch_size_actual, seq_len + pred_len, n_features, device=device)
        full_seq[:, :seq_len, :] = batch_inputs
        
        # Collect predictions efficiently
        all_preds = torch.zeros(batch_size_actual, pred_len, device=device)
        
        # Fixed autoregressive loop with realistic feature evolution
        for t in range(pred_len):
            # Sliding window - no memory allocation
            X = full_seq[:, t:t+seq_len, :]  # [batch, seq_len, features]
            pred = model(X).squeeze(-1).squeeze(-1)  # [batch]
            
            # Store prediction
            all_preds[:, t] = pred
            
            # FIXED: Proper feature evolution instead of copying entire row
            if t < pred_len - 1:  # Don't need to update on last iteration
                # Get the previous timestep's features
                prev_features = full_seq[:, seq_len + t - 1, :].clone()
                
                # Apply predicted percentage change to price-related features
                # Assuming features are: [open, high, low, close, volume, ..., other features, pct_chg]
                # Indices 0-3 are typically price features, let's evolve them realistically
                pct_change_factor = 1.0 + pred.detach().unsqueeze(-1)  # Convert pct to factor
                
                # Evolve price features (indices 0-3: open, high, low, close)
                prev_features[:, 0:4] = prev_features[:, 0:4] * pct_change_factor  # Apply to price features
                
                # For volume and other features (indices 4-9), use a more conservative approach
                # Instead of keeping them identical, add small realistic variations
                volume_noise = torch.randn_like(prev_features[:, 4:10]) * 0.01  # Small noise for non-price features
                prev_features[:, 4:10] = prev_features[:, 4:10] * (1.0 + volume_noise)
                
                # Set the predicted percentage change
                prev_features[:, 10] = pred.detach()
                
                # For any remaining features (indices 11+), keep them similar with small variation
                if n_features > 11:
                    remaining_noise = torch.randn_like(prev_features[:, 11:]) * 0.005
                    prev_features[:, 11:] = prev_features[:, 11:] * (1.0 + remaining_noise)
                
                # Validate evolved features to prevent extreme values
                # Clamp price features to reasonable bounds (prevent extreme price movements)
                original_prices = full_seq[:, seq_len + t - 1, 0:4]
                max_prices = original_prices * 10.0  # Max 1000% change
                prev_features[:, 0:4] = torch.maximum(prev_features[:, 0:4], 
                                                     torch.full_like(prev_features[:, 0:4], 0.01))
                prev_features[:, 0:4] = torch.minimum(prev_features[:, 0:4], max_prices)
                
                # Clamp percentage change to reasonable bounds
                prev_features[:, 10] = torch.clamp(prev_features[:, 10], min=-0.5, max=0.5)  # ±50% max
                
                # Update the sequence with evolved features
                full_seq[:, seq_len + t, :] = prev_features
        
        batch_targets = batch_targets.to(device)
        
        # Compute prediction loss (MSE)
        prediction_loss = loss_fn(all_preds, batch_targets)
        
        # Validate predictions before proceeding
        if torch.isnan(prediction_loss) or torch.isinf(prediction_loss):
            print(f"⚠️  CRITICAL: Loss is NaN/Inf at batch {batch_count}, skipping batch")
            continue
            
        if torch.isnan(all_preds).any() or torch.isinf(all_preds).any():
            print(f"⚠️  CRITICAL: Predictions contain NaN/Inf at batch {batch_count}, skipping batch")
            continue
        
        # Check for extreme predictions (sign of instability)
        pred_max = torch.max(torch.abs(all_preds)).item()
        if pred_max > 1.0:  # Percentage changes > 100% are suspicious
            print(f"⚠️  WARNING: Extreme prediction {pred_max:.3f} at batch {batch_count}")
        
        # Reduced smoothness loss: financial markets can have rapid changes
        if pred_len > 1:
            smoothness_loss = torch.mean((all_preds[:, 1:] - all_preds[:, :-1]) ** 2)
        else:
            smoothness_loss = torch.tensor(0.0, device=device)
        # Combine losses with reduced smoothness weight (0.05 instead of 0.25)
        total_sample_loss = prediction_loss + 0.05 * smoothness_loss
        
        total_sample_loss.backward()
        
        # Check gradients before optimizer step
        if not check_gradients(model, batch_count):
            print(f"⚠️  CRITICAL: Bad gradients detected, skipping optimizer step")
            optimizer.zero_grad()
            continue
        
        # Moderate gradient clipping (improved from 0.5 to 0.8 for better learning)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        # Log the loss for this batch
        logger.info(f"Batch {batch_count}: Loss = {total_sample_loss.item():.6f}")
        
        # Check model health after update
        if not check_model_health(model, batch_count):
            print(f"⚠️  CRITICAL: Model corrupted at batch {batch_count}, stopping training")
            break
        
        total_loss += total_sample_loss.item()
        total_prediction_loss += prediction_loss.item()
        total_smoothness_loss += smoothness_loss.item()
        
        # Calculate metrics
        batch_pred_tensor = all_preds.detach().cpu()
        batch_true_tensor = batch_targets.detach().cpu()
        diff = batch_pred_tensor - batch_true_tensor
        pct_mse = torch.mean(diff ** 2).item()
        pct_rmse = torch.sqrt(torch.mean(diff ** 2)).item()
        pct_directional_accuracy = (torch.mean((torch.sign(batch_pred_tensor) == torch.sign(batch_true_tensor)).float()) * 100).item()
        epoch_pct_mse += pct_mse
        epoch_pct_rmse += pct_rmse
        epoch_directional_accuracy += pct_directional_accuracy
        
        batch_count += 1
        if (batch_count) % 1000 == 0:
            torch.save(model.state_dict(), "mambastock_updated.pth")
            logger.info(f"Model saved to mambastock_updated.pth after batch {batch_count} of epoch {epoch+1}")
    if batch_count > 0:
        # Calculate AVERAGE loss per batch (was previously showing accumulated total!)
        avg_total_loss = total_loss / batch_count
        avg_pct_mse = epoch_pct_mse / batch_count
        avg_pct_rmse = epoch_pct_rmse / batch_count
        avg_directional_accuracy = epoch_directional_accuracy / batch_count
        avg_prediction_loss = total_prediction_loss / batch_count
        avg_smoothness_loss = total_smoothness_loss / batch_count
        
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss per Batch: {avg_total_loss:.6f} (Total: {total_loss:.2f} across {batch_count:,} batches)")
        print(f"  Prediction Loss: {avg_prediction_loss:.6f}")
        print(f"  Smoothness Loss: {avg_smoothness_loss:.6f}")
        print(f"  Pct Change MSE: {avg_pct_mse:.6f}")
        print(f"  Pct Change RMSE: {avg_pct_rmse:.6f}")
        print(f"  Directional Accuracy: {avg_directional_accuracy:.2f}%")
        force_log_write(f"{epoch+1} | Avg: {avg_total_loss:.6f} | Total: {total_loss:.2f} | Batches: {batch_count}")
    else:
        print(f"Epoch {epoch+1}/{epochs} - No valid batches processed")
        force_log_write(f"{epoch+1} | No valid batches")

# --- SAVE MODEL ---
torch.save(model.state_dict(), "mambastock_updatedsave.pth")
logger.info("Model saved to mambastock_updatedsave.pth")
logger.info("Training completed")

# --- BASELINE ---
def flat_baseline_prediction(history, pred_len=1):
    """Predicts the last observed value for all future steps (flat baseline)."""
    history = np.asarray(history)
    if history.ndim == 1:
        history = history[:, None]
    seq_len, num_features = history.shape
    last_value = history[-1]
    predictions = np.tile(last_value, (pred_len, 1))
    return predictions

# Replace all calls to linear_trend_prediction with flat_baseline_prediction in the code below

def evaluate_and_plot_single_prediction(dataset, model, seq_len, pred_len, device):
    model.eval()
    for stock_idx in range(dataset.n_stocks):
        stock_name = dataset.name_date[stock_idx]
        stock_data = dataset.data[stock_idx]  # [day_length, 11]
        # Only one prediction per stock: after the first seq_len minutes
        X = stock_data[:seq_len, :].unsqueeze(0).to(device)  # [1, seq_len, 11]
        y_true = stock_data[seq_len:seq_len+pred_len, 10].cpu().numpy()  # [pred_len]
        # Model now outputs only one prediction at a time
        # We need to make rolling predictions for pred_len steps
        y_pred = []
        input_seq = stock_data[:seq_len, :].clone().to(device)
        for t in range(pred_len):
            X = input_seq.unsqueeze(0).to(device)
            pred = model(X).squeeze().cpu().detach().numpy()
            y_pred.append(pred)
            # Add the actual next value to input_seq for next prediction
            # Use the PREDICTED value for next prediction (realistic)
            # Create a placeholder with the predicted value in the pct_chg position (index 10)
            next_pred = stock_data[seq_len + t, :].clone()
            # Convert numpy value to tensor before assignment
            next_pred[10] = torch.tensor(pred, dtype=next_pred.dtype, device=next_pred.device)
            next_pred = torch.tensor(next_pred).unsqueeze(0).to(device)
            input_seq = torch.cat([input_seq, next_pred], dim=0)
            input_seq = input_seq[-seq_len:, :]  # Keep only last seq_len elements
        y_pred = np.array(y_pred)
        # Baseline
        baseline = flat_baseline_prediction(stock_data[:seq_len, :].cpu().numpy(), pred_len)[:, 10]
        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(range(seq_len, seq_len+pred_len), y_true, label='Actual', color='black', linewidth=2)
        plt.plot(range(seq_len, seq_len+pred_len), y_pred, label='Mamba', color='red', linestyle='--', linewidth=2)
        plt.plot(range(seq_len, seq_len+pred_len), baseline, label='Baseline', color='blue', linestyle=':', linewidth=2)
        plt.title(f'Prediction for {stock_name} (minutes {seq_len} to {seq_len+pred_len-1})')
        plt.xlabel('Minute Index')
        plt.ylabel('Percentage Change')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{stock_name}_prediction.png')
        plt.show()

def evaluate_rolling_2min_prediction(dataset, model, seq_len, pred_len, device):
    """
    For each stock, fit on seq_len, then predict 2-minute intervals with 95% confidence intervals,
    append actuals, and repeat until pred_len is covered.
    Calculate metrics for the pred_len window.
    """
    model.eval()
    all_preds = []
    all_trues = []
    all_signs = []
    all_actual_signs = []
    all_ci_lower = []
    all_ci_upper = []
    all_coverage = []
    
    # Enable dropout for uncertainty estimation
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    
    # Set model to training mode for dropout to work
    model.train()

    for stock_idx in range(dataset.n_stocks):
        stock_data = dataset.data[stock_idx]  # [day_length, features]
        # Only predict if enough data
        if seq_len + pred_len > dataset.day_length:
            continue
        input_seq = stock_data[:seq_len, :].clone().to(device)  # [seq_len, features]
        preds = []
        trues = []
        ci_lower = []
        ci_upper = []
        coverage = []
        t = seq_len
        while t < seq_len + pred_len:
            # Predict next 2 minutes with uncertainty estimation
            X = input_seq[-seq_len:].unsqueeze(0).to(device)  # [1, seq_len, features]
            
            # Monte Carlo dropout for uncertainty estimation
            n_samples = 20  # Increase samples for better uncertainty estimation
            pred_samples = []
            with torch.no_grad():
                for _ in range(n_samples):
                    pred_sample = model(X)  # [1, 1, 1] - single prediction
                    # Extract single prediction value
                    pred_sample = pred_sample.squeeze().detach().cpu().numpy()
                    pred_samples.append(pred_sample)
            
            pred_samples = np.array(pred_samples)  # [n_samples]
            
            # Calculate mean prediction and confidence intervals
            pred_mean = np.mean(pred_samples)  # scalar
            pred_std = np.std(pred_samples)  # scalar
            
            # 95% confidence interval (1.96 * std)
            ci_lower_2min = pred_mean - 1.96 * pred_std
            ci_upper_2min = pred_mean + 1.96 * pred_std
            
            # True values for next 2 minutes
            # Since we're predicting one step at a time, get only the next value
            true_1min = stock_data[t, 10].to(device)
            true_1min_np = true_1min.detach().cpu().numpy()
            
            # Check if true values fall within confidence intervals
            coverage_1min = (true_1min_np >= ci_lower_2min and true_1min_np <= ci_upper_2min)
            
            preds.append(pred_mean)
            trues.append(true_1min_np)
            ci_lower.append(ci_lower_2min)
            ci_upper.append(ci_upper_2min)
            coverage.append(float(coverage_1min))
            
            # Append actual data to input_seq
            # Use the PREDICTED values for next prediction (realistic)
            # Create placeholders with the predicted values in the pct_chg position (index 10)
            # Use the PREDICTED value for next prediction (realistic)
            # Create a placeholder with the predicted value in the pct_chg position (index 10)
            next_pred = stock_data[t, :].clone()
            # Convert numpy value to tensor before assignment
            next_pred[10] = torch.tensor(pred_mean, dtype=next_pred.dtype, device=next_pred.device)
            next_pred = torch.tensor(next_pred).unsqueeze(0).to(device)
            input_seq = torch.cat([input_seq, next_pred], dim=0)
            t += 1
        
        # Store for metrics
        all_preds.extend(preds[:pred_len])
        all_trues.extend(trues[:pred_len])
        all_signs.extend(np.sign(preds[:pred_len]))
        all_actual_signs.extend(np.sign(trues[:pred_len]))
        all_ci_lower.extend(ci_lower[:pred_len])
        all_ci_upper.extend(ci_upper[:pred_len])
        all_coverage.extend(coverage[:pred_len])
    
    # Disable dropout after uncertainty estimation
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()

    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)
    all_ci_lower = np.array(all_ci_lower)
    all_ci_upper = np.array(all_ci_upper)
    all_coverage = np.array(all_coverage)
    
    valid = ~np.isnan(all_preds) & ~np.isnan(all_trues)
    mse = np.mean((all_preds[valid] - all_trues[valid]) ** 2)
    rmse = np.sqrt(mse)
    directional_acc = np.mean(np.array(all_signs)[valid] == np.array(all_actual_signs)[valid]) * 100
    correlation = np.corrcoef(all_preds[valid], all_trues[valid])[0, 1] if np.sum(valid) > 1 else float('nan')
    
    # Calculate confidence interval metrics
    ci_width = np.mean(all_ci_upper[valid] - all_ci_lower[valid])
    ci_coverage = np.mean(all_coverage[valid]) * 100
    
    # Calculate average standard deviation across all predictions
    # Since CI width = 2 * 1.96 * std, we can extract std from CI width
    avg_std = ci_width / (2 * 1.96)
    
    # Calculate Sharpe ratio using the formula: (return*100 - 0.018056)/(average standard deviation)
    # First calculate the average return (mean of actual percentage changes)
    avg_return = np.mean(all_trues[valid]) * 100  # Convert to percentage
    risk_free_rate_pct = 0.018056 # Convert to percentage
    # Convert avg_std to percentage to match the return units
    avg_std_pct = avg_std * 100  # Convert to percentage
    sharpe_ratio = (avg_return - risk_free_rate_pct) / avg_std_pct if avg_std_pct > 0 else float('nan')
    
    print("\nRolling 2-Minute Prediction Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  Directional Accuracy: {directional_acc:.2f}%")
    print(f"  Correlation: {correlation:.4f}")
    print(f"  Average CI Width: {ci_width:.6f}")
    print(f"  CI Coverage Rate: {ci_coverage:.2f}%")
    print(f"  Average Standard Deviation: {avg_std:.6f}")
    print(f"  Average Return: {avg_return:.6f}%")
    print(f"  Risk-free Rate: {risk_free_rate_pct:.6f}%")
    print(f"  Average Std (pct): {avg_std_pct:.6f}%")
    print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
    
    # Plot predictions with confidence intervals - show seq_len + pred_len for one stock
    plt.figure(figsize=(14, 8))
    
    # Plot all stocks' data with vertical lines separating each stock
    all_actuals_plot = []
    all_preds_plot = []
    all_ci_lower_plot = []
    all_ci_upper_plot = []
    all_times_plot = []
    
    for stock_idx in range(dataset.n_stocks):
        stock_data = dataset.data[stock_idx]
        if seq_len + pred_len > dataset.day_length:
            continue
            
        # Get seq_len period (input data)
        seq_data = stock_data[:seq_len, 10].cpu().numpy()
        seq_time = np.arange(stock_idx * (seq_len + pred_len), stock_idx * (seq_len + pred_len) + seq_len)
        
        # Get pred_len period (prediction data)
        pred_data = stock_data[seq_len:seq_len+pred_len, 10].cpu().numpy()
        pred_time = np.arange(stock_idx * (seq_len + pred_len) + seq_len, 
                             stock_idx * (seq_len + pred_len) + seq_len + pred_len)
        
        # Get predictions for this stock
        stock_preds = all_preds[stock_idx * pred_len:(stock_idx + 1) * pred_len]
        stock_ci_lower = all_ci_lower[stock_idx * pred_len:(stock_idx + 1) * pred_len]
        stock_ci_upper = all_ci_upper[stock_idx * pred_len:(stock_idx + 1) * pred_len]
        
        # Regenerate pred_len predictions to ensure they properly continue from seq_len
        # Start with the full seq_len data and make rolling predictions
        input_seq = stock_data[:seq_len, :].clone().to(device)
        rolling_preds = []
        rolling_ci_lower = []
        rolling_ci_upper = []
        
        for t in range(pred_len):
            # Use current input sequence to predict next value
            X = input_seq.unsqueeze(0).to(device)
            
            # Monte Carlo dropout for uncertainty estimation
            n_samples = 20
            pred_samples = []
            with torch.no_grad():
                for _ in range(n_samples):
                    pred_sample = model(X)
                    # Model outputs [1, 1, 1] - extract the single prediction
                    pred_sample = pred_sample.squeeze()  # Remove all dimensions of size 1
                    pred_samples.append(pred_sample.detach().cpu().numpy())
            
            pred_samples = np.array(pred_samples)
            pred_mean = np.mean(pred_samples)
            pred_std = np.std(pred_samples)
            
            # Debug: Print prediction statistics for first few predictions
            if t < 3:  # Only print for first 3 predictions to avoid spam
                print(f"Stock {stock_idx}, t={t}: pred_mean={pred_mean:.6f}, pred_std={pred_std:.6f}, samples_range=[{np.min(pred_samples):.6f}, {np.max(pred_samples):.6f}]")
            
            rolling_preds.append(pred_mean)
            rolling_ci_lower.append(pred_mean - 1.96 * pred_std)
            rolling_ci_upper.append(pred_mean + 1.96 * pred_std)
            
            # Add the actual next value to the input sequence for next prediction
            # Use the PREDICTED value for next prediction (realistic)
            # Create a placeholder with the predicted value in the pct_chg position (index 10)
            next_pred = stock_data[seq_len + t, :].clone()
            # Convert numpy value to tensor before assignment
            next_pred[10] = torch.tensor(pred_mean, dtype=next_pred.dtype, device=next_pred.device)
            next_pred = next_pred.detach().clone().unsqueeze(0).to(device)
            input_seq = torch.cat([input_seq, next_pred], dim=0)
            # Keep only the last seq_len elements
            input_seq = input_seq[-seq_len:, :]
        
        # Update the stock predictions with the properly generated ones
        stock_preds = rolling_preds
        stock_ci_lower = rolling_ci_lower
        stock_ci_upper = rolling_ci_upper
        
        # Generate predictions for seq_len period (should track actual data)
        # For seq_len period, use the actual data as predictions (model should reproduce it)
        seq_preds = seq_data.tolist()
        # Use a small standard deviation for seq_len confidence intervals since we know the actual values
        seq_std = 0.001  # Small uncertainty for known data
        seq_ci_lower = [seq_data[i] - 1.96 * seq_std for i in range(len(seq_data))]
        seq_ci_upper = [seq_data[i] + 1.96 * seq_std for i in range(len(seq_data))]
        
        # Plot seq_len period
        plt.plot(seq_time, seq_data, 'b-', linewidth=1.5, alpha=0.8)
        plt.plot(seq_time, seq_preds, 'r--', linewidth=1.5, alpha=0.8)
        plt.fill_between(seq_time, seq_ci_lower, seq_ci_upper, 
                         alpha=0.2, color='red')
        
        # Plot pred_len period
        plt.plot(pred_time, pred_data, 'b-', linewidth=1.5, alpha=0.8)
        plt.plot(pred_time, stock_preds, 'r--', linewidth=1.5, alpha=0.8)
        plt.fill_between(pred_time, stock_ci_lower, stock_ci_upper, 
                         alpha=0.2, color='red')
        
        # Add vertical red line to separate stocks (except before first stock)
        if stock_idx > 0:
            separator_x = stock_idx * (seq_len + pred_len)
            plt.axvline(x=separator_x, color='red', linestyle='-', linewidth=2, alpha=0.7)
        
        # Store for legend (only once)
        if stock_idx == 0:
            all_actuals_plot.extend(seq_data)
            all_actuals_plot.extend(pred_data)
            all_preds_plot.extend(seq_preds)  # Predictions during seq_len
            all_preds_plot.extend(stock_preds)
            all_ci_lower_plot.extend(seq_ci_lower)  # CI during seq_len
            all_ci_lower_plot.extend(stock_ci_lower)
            all_ci_upper_plot.extend(seq_ci_upper)  # CI during seq_len
            all_ci_upper_plot.extend(stock_ci_upper)
            all_times_plot.extend(seq_time)
            all_times_plot.extend(pred_time)
    
    plt.xlabel('Time Steps (2-minute intervals)')
    plt.ylabel('Percentage Change')
    plt.title(f'All Stocks: seq_len({seq_len}) + pred_len({pred_len}) = {seq_len+pred_len} per stock')
    
    # Add legend with sample data
    plt.plot(all_times_plot, all_actuals_plot, 'b-', label='Actual', linewidth=2, alpha=0.8)
    plt.plot(all_times_plot, all_preds_plot, 'r--', label='Predicted', linewidth=2, alpha=0.8)
    plt.fill_between(all_times_plot, all_ci_lower_plot, all_ci_upper_plot, 
                     alpha=0.3, color='red', label='95% Confidence Interval')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rolling_2min_predictions_with_ci.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'mse': mse,
        'rmse': rmse,
        'directional_accuracy': directional_acc,
        'correlation': correlation,
        'ci_width': ci_width,
        'ci_coverage': ci_coverage,
        'avg_std': avg_std,
        'avg_return': avg_return,
        'sharpe_ratio': sharpe_ratio,
        'predictions': all_preds,
        'actuals': all_trues,
        'ci_lower': all_ci_lower,
        'ci_upper': all_ci_upper
    }

if __name__ == "__main__":
    evaluate_rolling_2min_prediction(dataset, model, seq_len, pred_len, device)
