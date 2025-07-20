import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from mambastock_hdf5_dataset import MambaHDF5Dataset
import matplotlib.pyplot as plt
import argparse

def calculate_metrics(forecasts, ground_truth):
    """
    Calculate MSE and confidence interval coverage for predictions.
    
    Args:
        forecasts: [batch_size, num_samples, forecast_length] torch tensor
        ground_truth: [batch_size, forecast_length] torch tensor
    
    Returns:
        squared_error: [forecast_length] torch tensor - squared error for each forecast step
        confidence_coverage: [forecast_length] torch tensor - ratio of ground truth within 95% CI
    """
    # Calculate mean forecast
    mean_forecast = torch.mean(forecasts, dim=1)
    
    # Calculate MSE for each forecast step
    squared_error = (mean_forecast - ground_truth) ** 2
    
    # Calculate 95% confidence intervals
    lower_bound = torch.quantile(forecasts, 0.025, dim=1)
    upper_bound = torch.quantile(forecasts, 0.975, dim=1)
    
    # Check if ground truth falls within confidence interval for each step
    within_ci = (ground_truth >= lower_bound) & (ground_truth <= upper_bound)
    confidence_coverage = within_ci.float()
    
    return squared_error.sum(dim=0), confidence_coverage.sum(dim=0)

def test_model_performance(input_file, min_time, max_time, forecast_length, n_samples, batch_size):
    # Initialize results arrays
    time_span = max_time - min_time
    all_mse = np.zeros((time_span, forecast_length))
    all_confidence = np.zeros((time_span, forecast_length))
    prediction_counts = np.zeros((time_span, forecast_length))
    
    # Load model
    from mambastock_model import MambaStock
    model = MambaStock(input_size=1, seq_len=60, pred_len=1)
    model.load_state_dict(torch.load('mambastock_ohlcv.pth', map_location='cpu'))
    model.eval()
    
    # Load dataset
    dataset = ClosePrice(input_file)
    print(f"Dataset loaded: {len(dataset)} total items, {dataset.n_stocks} stocks")
    
    # Create sampler and dataloader
    sampler = TimeAlignedSampler(dataset, batch_size=batch_size, start_time=min_time, end_time=max_time, shuffle=False)
    dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)
    
    print(f"Processing {len(dataloader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            close_prices = batch['close_prices']  # [batch_size, time_idx]
            history = close_prices[:, :-forecast_length]
            ground_truth = close_prices[:, -forecast_length:]
            time_index = batch['time_index']  # int
            
            # Run batch inference
            forecasts = model.generate(
                history, 
                max_new_tokens=forecast_length, 
                num_samples=n_samples,
            )  # [batch_size, num_samples, forecast_length]
            
            # Get ground truth for each item in batch
            mse, confidence_coverage = calculate_metrics(forecasts, ground_truth)
            i = time_index - min_time
            all_mse[i] += mse.detach().cpu().numpy()
            all_confidence[i] += confidence_coverage.detach().cpu().numpy()
            prediction_counts[i] += forecasts.shape[0]
    
    # Average the results across all predictions
    # Avoid division by zero
    valid_predictions = prediction_counts > 0
    all_mse[valid_predictions] /= prediction_counts[valid_predictions]
    all_confidence[valid_predictions] /= prediction_counts[valid_predictions]
    
    # Set invalid predictions to NaN
    all_mse[~valid_predictions] = np.nan
    all_confidence[~valid_predictions] = np.nan
    
    print(f"Completed processing. Total predictions: {prediction_counts.sum()}")
    
    return all_mse, all_confidence

def plot_results(data, label):
    """
    Plot the given data as a 3D plot with time_idx, forecast_length, and the data
    Args: 
        data: [time_span, forecast_length] numpy array 
    """
    save_path = f"plots/{label}.png"
    time_span, forecast_length = data.shape
    time_idx = np.arange(time_span)
    forecast_idx = np.arange(forecast_length)
    T, F = np.meshgrid(time_idx, forecast_idx, indexing='ij')

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Flatten for plotting
    X = T.flatten()
    Y = F.flatten()
    Z = data.flatten()

    # Mask NaNs for plotting
    mask = ~np.isnan(Z)
    X = X[mask]
    Y = Y[mask]
    Z = Z[mask]

    surf = ax.plot_trisurf(X, Y, Z, cmap='viridis', linewidth=0.2, antialiased=True)
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Forecast Horizon')
    ax.set_zlabel(label)
    fig.colorbar(surf, shrink=0.5, aspect=10, label='Value')
    # plt.show()
    plt.savefig(save_path)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input hdf5 file")
    parser.add_argument("--min_time", type=int, default=30, help="Minutes after 9:30 to start predicting")
    parser.add_argument("--max_time", type=int, default=60, help="Minutes after 9:30 to stop predicting")
    parser.add_argument("--forecast_length", type=int, default=30, help="Minutes to predict for")
    parser.add_argument("--n_samples", type=int, default=20, help="Number of samples Sundial generates for probability distribution")
    parser.add_argument("--batch_size", type=int, default=40, help="Number of stocks to process at once")
    args = parser.parse_args()

    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # Run the test
    mse, confidence = test_model_performance(args.input_file, args.min_time + args.forecast_length, args.max_time + args.forecast_length, args.forecast_length, args.n_samples, args.batch_size) 
    plot_results(mse, "mse")
    plot_results(confidence, "confidence")