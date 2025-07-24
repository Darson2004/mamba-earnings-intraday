# From train.py
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