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

# --- CONFIG ---
h5_path = "h5_rty_data_test.h5"  # <-- HDF5 file path
seq_len = 30
pred_len = 10
batch_size = 8
epochs = 3
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            # Force close and reopen to ensure write
            handler.close()
            handler._open()

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

# Create sampler and dataloader
# Start at seq_len to truly exclude the first seq_len points from all calculations
sampler = TimeAlignedSampler(dataset, batch_size=batch_size, start_time=seq_len, end_time=dataset.day_length-pred_len, shuffle=True)
loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)

# --- INIT MODEL ---
# Adjust input_size based on the ClosePrice dataset structure
model = MambaStock(input_size=13, seq_len=seq_len, pred_len=pred_len).to(device)
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
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

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
    for batch in loader:
        close_prices = batch['close_prices']  # [batch_size, time_idx, 11]
        time_index = batch['time_index']
        batch_size_actual = close_prices.shape[0]
        batch_loss = 0
        batch_prediction_loss = 0
        batch_smoothness_loss = 0
        batch_pred_list = []
        batch_true_list = []
        for i in range(batch_size_actual):
            stock_idx = i
            stock_data = close_prices[i]  # [time_idx, 11]
            if time_index + pred_len < dataset.day_length:
                full_stock_data = dataset.data[stock_idx]
                input_seq = full_stock_data[time_index-seq_len:time_index, :].clone().to(device)  # [seq_len, 11]
                prediction_loss = 0  # Pure MSE loss on predictions
                rolling_preds = []
                rolling_trues = []
                for t in range(pred_len):
                    X = input_seq[-seq_len:].unsqueeze(0).to(device)
                    pred = model(X)
                    # Model now outputs [1, 1, 1] - extract the single prediction
                    pred_1min = pred.squeeze()  # Remove all dimensions of size 1
                    true_1min = full_stock_data[time_index + t, 10].to(device)
                    prediction_loss += loss_fn(pred_1min, true_1min)
                    rolling_preds.append(pred_1min)
                    rolling_trues.append(true_1min)
                    # Use the PREDICTED value for next prediction (realistic)
                    # Create a placeholder with the predicted value in the pct_chg position (index 10)
                    next_pred = full_stock_data[time_index + t, :].clone()
                    # Convert tensor to scalar before assignment
                    pred_val = pred_1min.detach().item()
                    next_pred[10] = pred_val  # Replace actual with predicted
                    next_pred = next_pred.unsqueeze(0).to(device)
                    input_seq = torch.cat([input_seq, next_pred], dim=0)
                # Calculate smoothness loss
                smoothness_loss = 0
                if len(rolling_preds) > 1:
                    rolling_preds_tensor = torch.stack(rolling_preds)
                    smoothness_loss = torch.mean((rolling_preds_tensor[1:] - rolling_preds_tensor[:-1])**2)
                
                # Combine losses
                total_sample_loss = (prediction_loss / pred_len) + 0.25 * smoothness_loss
                
                # Track individual loss components
                batch_prediction_loss += prediction_loss / pred_len
                batch_smoothness_loss += smoothness_loss
                batch_loss += total_sample_loss
                total_predictions += pred_len
                
                batch_pred_list.extend([p.detach().cpu().item() for p in rolling_preds])
                batch_true_list.extend([t.detach().cpu().item() for t in rolling_trues])
                # Store for plotting (last epoch only)
                if epoch == epochs - 1:
                    all_stock_preds[stock_idx].extend([p.detach().cpu().item() for p in rolling_preds])
                    all_stock_trues[stock_idx].extend([t.detach().cpu().item() for t in rolling_trues])
                    all_stock_predlens[stock_idx].append([p.detach().cpu().item() for p in rolling_preds])
                    all_stock_truelens[stock_idx].append([t.detach().cpu().item() for t in rolling_trues])
            else:
                continue
        if batch_size_actual > 0:
            batch_loss = batch_loss / batch_size_actual
            batch_prediction_loss = batch_prediction_loss / batch_size_actual
            batch_smoothness_loss = batch_smoothness_loss / batch_size_actual
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
            total_prediction_loss += batch_prediction_loss.item()
            total_smoothness_loss += batch_smoothness_loss.item()
            batch_pred_arr = np.array(batch_pred_list)
            batch_true_arr = np.array(batch_true_list)
            if len(batch_pred_arr) > 0:
                pct_mse = np.mean((batch_pred_arr - batch_true_arr) ** 2)
                pct_rmse = np.sqrt(pct_mse)
                pct_directional_accuracy = np.mean(np.sign(batch_pred_arr) == np.sign(batch_true_arr)) * 100
                epoch_pct_mse += pct_mse
                epoch_pct_rmse += pct_rmse
                epoch_directional_accuracy += pct_directional_accuracy
                batch_count += 1
    if batch_count > 0:
        avg_pct_mse = epoch_pct_mse / batch_count
        avg_pct_rmse = epoch_pct_rmse / batch_count
        avg_directional_accuracy = epoch_directional_accuracy / batch_count
        avg_prediction_loss = total_prediction_loss / batch_count
        avg_smoothness_loss = total_smoothness_loss / batch_count
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")
        print(f"  Prediction Loss: {avg_prediction_loss:.6f}")
        print(f"  Smoothness Loss: {avg_smoothness_loss:.6f}")
        print(f"  Pct Change MSE: {avg_pct_mse:.6f}")
        print(f"  Pct Change RMSE: {avg_pct_rmse:.6f}")
        print(f"  Directional Accuracy: {avg_directional_accuracy:.2f}%")
        force_log_write(f"{epoch+1} | {total_loss:.6f}")
    else:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} (No valid batches)")
        force_log_write(f"{epoch+1} | {total_loss:.6f} (No valid batches)")

# --- PLOTTING: All stocks, full day, last epoch, subplots ---
import matplotlib.pyplot as plt
n_stocks = len(all_stock_preds)
fig, axes = plt.subplots(n_stocks, 1, figsize=(14, 3*n_stocks), sharex=True)
if n_stocks == 1:
    axes = [axes]
for stock_idx, (preds, trues) in enumerate(zip(all_stock_preds, all_stock_trues)):
    x = np.arange(len(trues))
    axes[stock_idx].plot(x, trues, label='Actual', linestyle='-', linewidth=1)
    axes[stock_idx].plot(x, preds, label='Predicted', linestyle='--', linewidth=1)
    axes[stock_idx].set_ylabel('Pct Change')
    axes[stock_idx].set_title(f'Stock {stock_idx}')
    axes[stock_idx].set_xlim(0, dataset.day_length)
    if stock_idx == 0:
        axes[stock_idx].legend()
axes[-1].set_xlabel('Minute Index (0-389)')
fig.suptitle('Predictions vs Actuals for Each Stock (Last Epoch, Full Day)', fontsize=16)
plt.tight_layout(rect=(0, 0, 1, 0.97))
plt.show()

# --- SAVE MODEL ---
torch.save(model.state_dict(), "mambastock_updated.pth")
logger.info("Model saved to mambastock_updated.pth")
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
            n_samples = 100
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

def compare_predictions():
    """
    Compare Mamba model vs a constant flat line baseline (e.g., zero).
    """
    print("Evaluating Mamba model vs flat line baseline...")

    # Create a single test dataloader to ensure both models see the same data
    test_sampler = TimeAlignedSampler(dataset, batch_size=batch_size, start_time=seq_len, end_time=dataset.day_length - pred_len, shuffle=False)
    test_loader = DataLoader(dataset, batch_sampler=test_sampler, collate_fn=collate_fn)

    all_mamba_predictions = []
    all_actuals = []
    all_times = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            close_prices = batch['close_prices']  # shape: [batch_size, seq_len, features]
            time_index = batch['time_index']

            # Get actual values
            if time_index + pred_len < dataset.day_length:
                targets = []
                for i in range(len(close_prices)):
                    stock_idx = batch['stock_index'][i] if 'stock_index' in batch else i
                    next_values = dataset.data[stock_idx, time_index + 1:time_index + 1 + pred_len, :]
                    targets.append(torch.tensor(next_values, dtype=torch.float32))
                y = torch.stack(targets)
            else:
                y = torch.zeros(len(close_prices), pred_len, 13)

            # Generate Mamba predictions
            X = close_prices[:, -seq_len:, :].to(device)
            # Model now outputs only one prediction at a time
            # We need to make rolling predictions for each sample
            mamba_pred = []
            for i in range(len(close_prices)):
                input_seq = close_prices[i, -seq_len:, :].to(device)
                sample_preds = []
                for t in range(pred_len):
                    X = input_seq.unsqueeze(0).to(device)
                    pred = model(X).squeeze()
                    sample_preds.append(pred)
                    # For comparison, we don't have actual future values to append
                    # So we'll use the last known value as a placeholder
                    if t < pred_len - 1:  # Don't append for the last prediction
                        last_known = input_seq[-1:, :]
                        input_seq = torch.cat([input_seq, last_known], dim=0)
                        input_seq = input_seq[-seq_len:, :]
                mamba_pred.append(torch.stack(sample_preds))
            mamba_pred = torch.stack(mamba_pred)  # [batch_size, pred_len]

            # Only append to all_times when appending a prediction/target
            mamba_pred_np = mamba_pred.cpu().numpy()
            y_np = y.numpy()
            for i in range(mamba_pred_np.shape[0]):
                for t in range(mamba_pred_np.shape[1]):
                    all_mamba_predictions.append(mamba_pred_np[i, t])  # Now just [batch, pred_len]
                    all_actuals.append(y_np[i, t, 10])
                    all_times.append(time_index + t)

    all_mamba_predictions = np.array(all_mamba_predictions)
    all_actuals = np.array(all_actuals)
    times = np.array(all_times)

    # Use a flat line (e.g., zero) as the baseline
    flat_value = 0.0
    flat_baseline = np.full_like(all_actuals, flat_value)
    actuals = np.atleast_1d(all_actuals)
    mamba_preds = np.atleast_1d(all_mamba_predictions)

    # Calculate metrics for flat baseline
    baseline_errors = flat_baseline - actuals
    baseline_mse = np.mean(baseline_errors**2)
    baseline_rmse = np.sqrt(baseline_mse)
    valid_baseline = ~np.isnan(actuals) & ~np.isnan(flat_baseline)
    baseline_corr = np.corrcoef(actuals[valid_baseline], flat_baseline[valid_baseline])[0, 1] if np.sum(valid_baseline) > 1 else float('nan')
    baseline_directional = np.mean(np.sign(flat_baseline) == np.sign(actuals)) * 100

    # Calculate metrics for Mamba
    mamba_errors = mamba_preds - actuals
    mamba_mse = np.mean(mamba_errors**2)
    mamba_rmse = np.sqrt(mamba_mse)
    valid_mamba = ~np.isnan(actuals) & ~np.isnan(mamba_preds)
    mamba_corr = np.corrcoef(actuals[valid_mamba], mamba_preds[valid_mamba])[0, 1] if np.sum(valid_mamba) > 1 else float('nan')
    mamba_directional = np.mean(np.sign(mamba_preds) == np.sign(actuals)) * 100

    # Debug: Print array lengths to confirm alignment
    print(f"all_times: {len(all_times)}, all_mamba_predictions: {len(all_mamba_predictions)}, all_actuals: {len(all_actuals)}")

    print(f"\n{'='*60}")
    print(f"{'METRIC':<20} {'FLAT BASELINE':<15} {'MAMBA':<15} {'IMPROVEMENT':<15}")
    print(f"{'='*60}")
    print(f"{'Mean Square Error':<20} {baseline_mse:<15.6f} {mamba_mse:<15.6f} {(baseline_mse - mamba_mse) / baseline_mse * 100 if baseline_mse != 0 else float('nan'):<15.2f}%")
    print(f"{'Root Mean Square Error':<20} {baseline_rmse:<15.6f} {mamba_rmse:<15.6f} {(baseline_rmse - mamba_rmse) / baseline_rmse * 100 if baseline_rmse != 0 else float('nan'):<15.2f}%")
    print(f"{'Correlation':<20} {baseline_corr:<15.4f} {mamba_corr:<15.4f} {(mamba_corr - baseline_corr) * 100 if not np.isnan(baseline_corr) and not np.isnan(mamba_corr) else float('nan'):<15.2f}%")
    print(f"{'Directional Accuracy':<20} {baseline_directional:<15.2f}% {mamba_directional:<15.2f}% {(mamba_directional - baseline_directional):<15.2f}%")
    print(f"{'='*60}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Mamba vs Flat Line Baseline Prediction Comparison (Excluding First seq_len Points)', fontsize=16, fontweight='bold')

    min_length = min(len(actuals), len(flat_baseline), len(mamba_preds))
    axes[0, 0].scatter(actuals[:min_length], flat_baseline[:min_length], alpha=0.6, s=20, label='Flat Baseline', color='blue')
    axes[0, 0].scatter(actuals[:min_length], mamba_preds[:min_length], alpha=0.6, s=20, label='Mamba', color='red')
    perfect = [min(actuals[:min_length].min(), mamba_preds[:min_length].min()),
               max(actuals[:min_length].max(), mamba_preds[:min_length].max())]
    axes[0, 0].plot(perfect, perfect, 'k--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Percentage Change')
    axes[0, 0].set_ylabel('Predicted Percentage Change')
    axes[0, 0].set_title('Predicted vs Actual Values')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # --- FIX: Only plot histograms if there are valid (finite) values ---
    valid_baseline_errors = baseline_errors[np.isfinite(baseline_errors)]
    valid_mamba_errors = mamba_errors[np.isfinite(mamba_errors)]
    if valid_baseline_errors.size > 0 and valid_mamba_errors.size > 0:
        axes[0, 1].hist(valid_baseline_errors.flatten(), bins=30, alpha=0.7, label='Flat Baseline', color='blue', density=True)
        axes[0, 1].hist(valid_mamba_errors.flatten(), bins=30, alpha=0.7, label='Mamba', color='red', density=True)
        axes[0, 1].axvline(0, color='black', linestyle='--', alpha=0.8)
        axes[0, 1].set_xlabel('Prediction Error')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Error Distribution Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No valid error data to plot', ha='center', va='center')

    min_length = min(50, len(flat_baseline), len(actuals), len(mamba_preds))
    axes[1, 0].plot(range(min_length), actuals[:min_length], 'b-', label='Actual', alpha=0.7, linewidth=1)
    axes[1, 0].plot(range(min_length), flat_baseline[:min_length], 'b--', label='Flat Baseline', alpha=0.7, linewidth=1)
    axes[1, 0].plot(range(min_length), mamba_preds[:min_length], 'r--', label='Mamba', alpha=0.7, linewidth=1)
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Percentage Change')
    axes[1, 0].set_title('Time Series: First 50 Predictions')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Ensure all arrays are the same length for plotting
    min_scatter_length = min(len(times), len(baseline_errors), len(mamba_errors))
    times_plot = times[:min_scatter_length]
    baseline_errors_plot = baseline_errors[:min_scatter_length]
    mamba_errors_plot = mamba_errors[:min_scatter_length]

    axes[1, 1].scatter(times_plot, baseline_errors_plot, alpha=0.6, s=20, label='Flat Baseline', color='blue')
    axes[1, 1].scatter(times_plot, mamba_errors_plot, alpha=0.6, s=20, label='Mamba', color='red')
    axes[1, 1].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Time Index (minutes from 9:30)')
    axes[1, 1].set_ylabel('Prediction Error')
    axes[1, 1].set_title('Error vs Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mamba_vs_flatline_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nComparison plot saved as 'mamba_vs_flatline_comparison.png'")

    print("mamba_preds shape:", mamba_preds.shape)
    print("actuals shape:", actuals.shape)
    print("Number of finite mamba_preds:", np.isfinite(mamba_preds).sum())
    print("Number of finite actuals:", np.isfinite(actuals).sum())
    print("First 10 mamba_preds:", mamba_preds[:10])
    print("First 10 actuals:", actuals[:10])


# Run comparison
compare_predictions()



