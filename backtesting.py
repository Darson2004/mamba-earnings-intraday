import argparse
import re
import random
from collections import defaultdict
import csv
import os

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid popup windows
import matplotlib.pyplot as plt

from dataset import ClosePrice
from mambastock_model import MambaStock
from bare_strategy import TradingStrategy


def extract_date(name: str) -> str:
    """Extract YYYY-MM-DD date from an HDF5 key or return 'unknown'."""
    # Try multiple patterns; return normalized YYYY-MM-DD
    patterns = [
        r"(\d{4})-(\d{1,2})-(\d{1,2})",
        r"(\d{4})/(\d{1,2})/(\d{1,2})",
        r"(\d{4})\.(\d{1,2})\.(\d{1,2})",
        r"(\d{4})(\d{2})(\d{2})",
        r"(\d{1,2})/(\d{1,2})/(\d{4})",
        r"(\d{4})[^0-9]*(\d{1,2})[^0-9]*(\d{1,2})",
    ]
    for pat in patterns:
        m = re.search(pat, name)
        if m:
            g = m.groups()
            if len(g) == 3:
                if len(g[0]) == 4:  # YYYY first
                    y, mo, d = g[0], g[1].zfill(2), g[2].zfill(2)
                else:  # MM/DD/YYYY
                    mo, d, y = g[0].zfill(2), g[1].zfill(2), g[2]
                return f"{y}-{mo}-{d}"
    return "unknown"


def build_date_batches(name_date_list):
    date_to_indices = defaultdict(list)
    for i, name in enumerate(name_date_list):
        date = extract_date(name)
        date_to_indices[date].append(i)
    if "unknown" in date_to_indices:
        print(f"⚠️  {len(date_to_indices['unknown'])} stocks with unknown date; skipping those batches")
        del date_to_indices["unknown"]
    # Randomize order if desired; keep deterministic for now
    batches = list(date_to_indices.items())
    return batches


def load_model(device: torch.device, input_size: int, seq_len: int, pred_len: int) -> MambaStock:
    print("🔧 Creating model...")
    model = MambaStock(input_size=input_size, seq_len=seq_len, pred_len=pred_len)
    model = model.to(device)
    model.eval()
    # GPU perf tweaks
    if device.type == 'cuda':
        try:
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("✅ Enabled cuDNN benchmark and TF32")
        except Exception:
            pass
    # Prefer weights from train_two_gpu pipeline
    ckpts = ["mambastock_scrolling_eight.pth", "mambastock_scrolling_two.pth"]
    loaded = False
    for ck in ckpts:
        try:
            state = torch.load(ck, map_location=device)
            model.load_state_dict(state, strict=False)
            print(f"✅ Loaded weights from {ck}")
            loaded = True
            break
        except Exception as e:
            print(f"⚠️  Could not load {ck}: {e}")
    if not loaded:
        print("⚠️  Proceeding with randomly initialized weights")
    return model


def plot_predictions_vs_actuals(stock_predictions, dataset, stock_indices, date, sample_size=2):
    """Plot predictions vs actual results for randomly selected stocks."""
    try:
        # Randomly select sample_size stocks from the batch
        available_stocks = list(stock_predictions.keys())
        if len(available_stocks) < sample_size:
            sample_size = len(available_stocks)
        
        selected_stocks = random.sample(available_stocks, sample_size)
        
        # Create subplot for each selected stock
        fig, axes = plt.subplots(sample_size, 1, figsize=(15, 6 * sample_size))
        if sample_size == 1:
            axes = [axes]
        
        fig.suptitle(f'Predictions vs Actual Results - {date}', fontsize=16, fontweight='bold')
        
        for i, stock_name in enumerate(selected_stocks):
            if stock_predictions[stock_name]:
                preds = stock_predictions[stock_name]
                
                # Extract data
                timestamps = [p['timestamp'] for p in preds]
                predictions = [p['prediction'] for p in preds]
                pred_prices = [p['price'] for p in preds]
                
                # Get actual prices for the same timestamps
                stock_idx = None
                for idx in stock_indices:
                    if dataset.name_date[idx] == stock_name:
                        stock_idx = idx
                        break
                
                if stock_idx is not None:
                    actual_prices = []
                    for t in timestamps:
                        if t < len(dataset.data[stock_idx]):
                            actual_prices.append(dataset.data[stock_idx][t, 3].item())  # Close price
                        else:
                            actual_prices.append(actual_prices[-1] if actual_prices else 0.0)
                    
                    # Calculate actual returns (percentage change from first price)
                    if actual_prices and actual_prices[0] > 0:
                        first_price = actual_prices[0]
                        actual_returns = [(price - first_price) / first_price * 100 for price in actual_prices]
                        
                        # Convert predictions to percentage and align with actual baseline
                        # The model predicts absolute returns, so we need to align the starting point
                        pred_returns = [pred * 100 for pred in predictions]
                        
                        # CRITICAL FIX: Align prediction baseline with actual data starting from t=30 (10:00 AM)
                        # Predictions begin at t=30 (seq_len), so align them with actual returns at that exact time
                        if len(actual_returns) > 0 and len(pred_returns) > 0:
                            # Find the first prediction timestamp (should be t=30, 10:00 AM)
                            first_pred_idx = 0
                            for j, t in enumerate(timestamps):
                                if t >= 30:  # Start aligning from t=30 when predictions begin
                                    first_pred_idx = j
                                    break
                            
                            if first_pred_idx < len(pred_returns) and first_pred_idx < len(actual_returns):
                                # Calculate the offset needed to align predictions with actuals at t=30
                                actual_at_pred_start = actual_returns[first_pred_idx]
                                pred_at_start = pred_returns[first_pred_idx]
                                
                                # Adjust all predictions by this offset to align baselines
                                baseline_offset = actual_at_pred_start - pred_at_start
                                pred_returns = [pred + baseline_offset for pred in pred_returns]
                                
                                # Debug info
                                if i == 0:  # Only print for first stock to avoid spam
                                    print(f"        🔧 Baseline alignment at t={timestamps[first_pred_idx]} (10:00 AM): actual={actual_at_pred_start:.2f}%, pred={pred_at_start:.2f}%, offset={baseline_offset:.2f}%")
                                    
                                    # Show alignment quality for first few predictions
                                    if len(pred_returns) > 3:
                                        print(f"        📊 Alignment check: t={timestamps[0]}: pred={pred_returns[0]:.2f}%, actual={actual_returns[0]:.2f}%")
                                        print(f"        📊 Alignment check: t={timestamps[1]}: pred={pred_returns[1]:.2f}%, actual={actual_returns[1]:.2f}%")
                                        print(f"        📊 Alignment check: t={timestamps[2]}: pred={pred_returns[2]:.2f}%, actual={actual_returns[2]:.2f}%")
                        
                        # Create the plot
                        ax = axes[i]
                        
                        # Plot actual returns
                        ax.plot(timestamps, actual_returns, 'b-', linewidth=2, label='Actual Returns (%)', alpha=0.8)
                        
                        # Plot predictions
                        ax.plot(timestamps, pred_returns, 'r--', linewidth=2, label='Predicted Returns (%)', alpha=0.8)
                        
                        # Add horizontal lines for thresholds - SHORTING DISABLED
                        ax.axhline(y=0.1, color='g', linestyle=':', alpha=0.5, label='BUY Threshold (0.1%)')
                        # ax.axhline(y=-0.3, color='orange', linestyle=':', alpha=0.5, label='SHORT Threshold (-0.3%)')  # SHORTING DISABLED
                        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, label='Zero Line')
                        
                        # Mark trading decisions - SHORTING DISABLED
                        for j, (t, pred, price) in enumerate(zip(timestamps, predictions, pred_prices)):
                            if pred >= 0.001:  # BUY signal
                                ax.scatter(t, pred_returns[j], color='green', s=50, marker='^', alpha=0.7)
                            # elif pred <= -0.003:  # SHORT signal - SHORTING DISABLED
                            #     ax.scatter(t, pred_returns[j], color='red', s=50, marker='v', alpha=0.7)
                        
                        # Customize the plot
                        ax.set_title(f'{stock_name} - Predictions vs Actual Returns', fontsize=14, fontweight='bold')
                        ax.set_xlabel('Time (minutes from 9:30 AM)', fontsize=12)
                        ax.set_ylabel('Returns (%)', fontsize=12)
                        ax.legend(fontsize=10)
                        ax.grid(True, alpha=0.3)
                        
                        # Add prediction statistics
                        pred_std = np.std(predictions) * 100
                        pred_mean = np.mean(predictions) * 100
                        ax.text(0.02, 0.98, f'Pred Mean: {pred_mean:.2f}%\nPred Std: {pred_std:.2f}%', 
                               transform=ax.transAxes, verticalalignment='top', 
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                        
                        # Add correlation info
                        if len(actual_returns) > 1:
                            correlation = np.corrcoef(pred_returns, actual_returns)[0, 1]
                            ax.text(0.98, 0.98, f'Correlation: {correlation:.3f}', 
                                   transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                    else:
                        ax = axes[i]
                        ax.text(0.5, 0.5, f'No valid price data for {stock_name}', 
                               transform=ax.transAxes, ha='center', va='center', fontsize=14)
                        ax.set_title(f'{stock_name} - No Data Available', fontsize=14)
                else:
                    ax = axes[i]
                    ax.text(0.5, 0.5, f'Stock index not found for {stock_name}', 
                           transform=ax.transAxes, ha='center', va='center', fontsize=14)
                    ax.set_title(f'{stock_name} - Index Not Found', fontsize=14)
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f'predictions_vs_actuals_{date}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"    📊 Prediction vs Actual plots saved to: {plot_filename}")
        
        # Show the plot
        plt.show()
        
    except Exception as e:
        print(f"    ⚠️  Failed to create prediction plots: {e}")


def save_predictions_to_csv(stock_predictions, filename):
    """Save all stock predictions to a CSV file for detailed analysis."""
    try:
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['stock', 'timestamp', 'prediction', 'price', 'action']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for stock_name, preds in stock_predictions.items():
                for pred in preds:
                    action = 'BUY' if pred['prediction'] >= 0.001 else 'HOLD'  # SHORTING DISABLED
                    writer.writerow({
                        'stock': stock_name,
                        'timestamp': pred['timestamp'],
                        'prediction': pred['prediction'],
                        'price': pred['price'],
                        'action': action
                    })
        print(f"✅ Predictions saved to {filename}")
    except Exception as e:
        print(f"⚠️  Failed to save predictions to {filename}: {e}")


@torch.inference_mode()
def predict_mu20_like_train_two(model: MambaStock, device: torch.device, current_seq: torch.Tensor, seq_len: int) -> float:
    """
    Predict using the EXACT normalization approach matching train_two_gpu.py:
    - Normalize over the last seq_len window (no chaining)
    - Feed ONLY the last window to the model
    - Model input shape: [1, seq_len, features]
    - Output: scalar prediction representing 20-min trend
    """
    # current_seq: [T, features] on device
    window = current_seq[-seq_len:, :]
    mean = window.mean(dim=0, keepdim=True)
    std = window.std(dim=0, keepdim=True)
    std = torch.where(std > 1e-8, std, torch.ones_like(std))
    seq_norm = (window - mean) / std
    
    # Feed the entire normalized sequence to the model (EXACTLY like training)
    model_input = seq_norm.unsqueeze(0)  # [1, seq_len, features]
    out = model(model_input)
    out = out.squeeze()
    return float(out.item())


def run_backtest(
    h5_path: str,
    initial_portfolio: float = 100000.0,
    seq_len: int = 30,           # Predictions start at 10:00 AM (t=30)
    pred_len: int = 20,          # 20-minute trend target
    predict_end_minute: int = 360,  # 3:30 PM (from 9:30 AM)
    trade_start_minute: int = 90,   # 11:00 AM
    trade_end_minute: int = 210,    # 1:00 PM
    stock_batch_size: int = 512,    # GPU chunking for large universes
):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Data
    dataset = ClosePrice(h5_path)
    print(f"Loaded dataset: {dataset.n_stocks} stocks, {dataset.day_length} minutes/day")

    # Model (infer input_size from dataset features)
    inferred_input_size = int(dataset.data[0].shape[1])
    model = load_model(device, input_size=inferred_input_size, seq_len=seq_len, pred_len=1)

    # Group by date in names
    date_batches = build_date_batches(dataset.name_date)
    print(f"📅 Found {len(date_batches)} unique trading dates")

    # Portfolio state across days
    current_portfolio = float(initial_portfolio)
    total_realized_pnl = 0.0

    # Iterate days
    for day_idx, (date, stock_indices) in enumerate(date_batches, start=1):
        print("".ljust(60, "="))
        print(f"📊 TRADING DAY {day_idx}/{len(date_batches)}: {date}")
        print(f"💰 Starting Portfolio: ${current_portfolio:,.2f}")
        print(f"📈 Stocks in batch: {len(stock_indices)}")
        print("".ljust(60, "="))

        # Strategy for this day; match constraints and fees
        strategy = TradingStrategy(
            model=model,
            device=device,
            dataset=dataset,
            considered_stocks=stock_indices,
            portfolio_value=current_portfolio,
            seq_len=seq_len,
            pred_len=pred_len,
            max_position_frac=1.0 / max(1, len(stock_indices)),
            start_trading_minute=trade_start_minute,
            max_trading_minutes=(trade_end_minute - trade_start_minute),
        )

        # Per-symbol FIFO queues for realized P&L - SHORTING DISABLED
        buy_queues = {dataset.name_date[idx]: [] for idx in stock_indices}
        # short_queues = {dataset.name_date[idx]: [] for idx in stock_indices}  # SHORTING DISABLED
        batch_realized = 0.0

        # Run predictions from open until predict_end_minute (batched on GPU)
        max_t = min(predict_end_minute, dataset.day_length - 1)

        # Prepare batched rolling sequence: [N, 0, F] grow minute-by-minute
        N = len(stock_indices)
        F = dataset.data[stock_indices[0]].shape[1]
        batch_seq = torch.empty((N, 0, F), device=device)
        names = [dataset.name_date[idx] for idx in stock_indices]
        
        # Track predictions for each stock throughout the day
        stock_predictions = {name: [] for name in names}
        prediction_timestamps = []

        for t in range(0, max_t + 1):
            # Accumulate raw rows for each stock in batch tensor
            rows = [dataset.data[idx][t : t + 1, :].clone().to(device) for idx in stock_indices]
            # rows: list of [1, F] -> [N, 1, F]
            new_rows = torch.stack([r.squeeze(0) for r in rows], dim=0).unsqueeze(1)
            batch_seq = torch.cat([batch_seq, new_rows], dim=1)

            # Only predict once we have at least seq_len minutes (EXACTLY like train_two_gpu.py)
            if t < seq_len:
                continue

            # Compute predictions for all stocks at this minute (GPU parallel)
            minute_metrics = []  # (idx, name, mu20)
            T = batch_seq.shape[1]
            
            # Print prediction summary every 30 minutes for monitoring
            if t % 30 == 0:
                print(f"  📊 Minute {t}: Computing predictions for {N} stocks...")
            
            # Process in chunks to control VRAM
            for start in range(0, N, max(1, stock_batch_size)):
                end = min(N, start + stock_batch_size)
                sub_seq = batch_seq[start:end, :, :]  # [B, T, F]
                # Normalize per stock over the last seq_len window (no chaining)
                sub_window = sub_seq[:, -seq_len:, :]  # [B, seq_len, F]
                mean = sub_window.mean(dim=1, keepdim=True)
                std = sub_window.std(dim=1, keepdim=True)
                std = torch.where(std > 1e-8, std, torch.ones_like(std))
                sub_norm = (sub_window - mean) / std
                
                # Use only the last-window normalized sequence for model input
                sub_input = sub_norm  # [B, seq_len, F]
                
                # Debug: Print the normalization approach (only once per run)
                if t == seq_len and start == 0:
                    print(f"    🔧 NORMALIZATION: Using last window={seq_len} timesteps for normalization and model input")
                
                # Inference with autocast on CUDA
                if device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        sub_out = model(sub_input)
                else:
                    sub_out = model(sub_input)
                
                # Handle tuple output (mean, log_var) from trained model
                if isinstance(sub_out, tuple):
                    sub_out = sub_out[0]  # Take the mean prediction
                
                sub_out = sub_out.squeeze()
                # Ensure shape [B]
                if sub_out.dim() == 0:
                    sub_out = sub_out.unsqueeze(0)
                elif sub_out.dim() > 1:
                    sub_out = sub_out.view(sub_out.shape[0])
                sub_out_np = sub_out.detach().float().cpu().numpy().tolist()
                for i, mu in enumerate(sub_out_np):
                    global_idx = stock_indices[start + i]
                    name = names[start + i]
                    minute_metrics.append((global_idx, name, float(mu)))
                    
                    # Store prediction for this stock at this timestamp
                    stock_predictions[name].append({
                        'timestamp': t,
                        'prediction': float(mu),
                        'price': dataset.data[global_idx][t, 3].item() if t < len(dataset.data[global_idx]) else 0.0
                    })
            
            # Store timestamp for this prediction round
            if minute_metrics:
                prediction_timestamps.append(t)
            
            # Print prediction statistics every 30 minutes
            if t % 30 == 0 and minute_metrics:
                mus = [m[2] for m in minute_metrics]
                print(f"    📈 Predictions: min={min(mus):.4f}, max={max(mus):.4f}, mean={np.mean(mus):.4f}")
                buy_signals = len([m for m in minute_metrics if m[2] >= 0.001])
                # short_signals = len([m for m in minute_metrics if m[2] <= -0.003])  # SHORTING DISABLED
                hold_signals = len([m for m in minute_metrics if m[2] < 0.001])  # All non-buy are holds
                print(f"    🎯 Signals: BUY={buy_signals}, HOLD={hold_signals}")  # No more SHORT

            # During trading window [trade_start, trade_end), execute bare_strategy decisions USING SAME PREDICTIONS
            if trade_start_minute <= t < trade_end_minute:
                print(f"  🚀 TRADING WINDOW: Minute {t} (11:00 AM - 1:00 PM)")
                
                # Sort by mu descending to process strongest longs first (optional)
                minute_metrics.sort(key=lambda x: x[2], reverse=True)

                for idx, name, mu20 in minute_metrics:
                    price_mid = dataset.data[idx][t, 3].item()  # close/last col=3

                    # Exits first
                    if name in strategy.positions:
                        if mu20 <= 0.0:
                            pos = strategy.positions[name]
                            print(f"    🔴 EXIT LONG: {name} at t={t}, mu={mu20:.4f}, price=${price_mid:.2f}")
                            strategy.sell(name, price_mid, pos['shares'] * price_mid, t)
                            # Record SELL in FIFO
                            sell_price = price_mid * (1 - getattr(strategy, 'spread_fee', 0.0))
                            shares = pos['shares']
                            # Consume from buy queue for realized P&L
                            shares_left = shares
                            while shares_left > 0 and buy_queues[name]:
                                lot = buy_queues[name][0]
                                match_qty = min(lot['shares'], shares_left)
                                gain = (sell_price - lot['price']) * match_qty
                                batch_realized += gain
                                lot['shares'] -= match_qty
                                shares_left -= match_qty
                                if lot['shares'] <= 1e-8:
                                    buy_queues[name].pop(0)
                            continue

                                        # SHORTING DISABLED - Comment out short position exits
                    # if name in strategy.short_positions:
                    #     if mu20 >= 0.0:
                    #         print(f"    🔵 EXIT SHORT: {name} at t={t}, mu={mu20:.4f}, price=${price_mid:.2f}")
                    #         strategy.short_cover(name, price_mid, t)
                    #         # Record COVER in FIFO
                    #         cover_price = price_mid * (1 + getattr(strategy, 'short_spread_fee', 0.003))
                    #         shares = short_queues[name][0]['shares'] if short_queues[name] else 0.0
                    #         shares_left = shares
                    #         while shares_left > 0 and short_queues[name]:
                    #             lot = short_queues[name][0]
                    #             match_qty = min(lot['shares'], shares_left)
                    #             gain = (lot['price'] - cover_price) * match_qty
                    #             batch_realized += gain
                    #             lot['shares'] -= match_qty
                    #             shares_left -= match_qty
                    #             if lot['shares'] <= 1e-8:
                    #                 short_queues[name].pop(0)
                    #         continue

                    # Entries
                    if t <= (trade_end_minute - 30):  # leave time to manage position
                        # Long entries
                        if name not in strategy.positions and mu20 >= strategy.buy_threshold:
                            strength = min(1.0, max(0.0, mu20) / 0.03)
                            cap = strategy.portfolio_value * 0.01
                            amount = min(cap * strength, strategy.cash)
                            if amount > 0:
                                print(f"    🟢 ENTER LONG: {name} at t={t}, mu={mu20:.4f}, price=${price_mid:.2f}, amount=${amount:.0f}")
                                entry_cap = strategy.buy(name, price_mid, amount, t=t, snr=0.0)
                                if entry_cap > 0:
                                    ask = price_mid * (1 + getattr(strategy, 'spread_fee', 0.0))
                                    shares = entry_cap / ask
                                    buy_queues[name].append({'shares': shares, 'price': ask, 'time': t, 'snr': 0.0})

                        # SHORTING DISABLED - Comment out short entries
                        # if strategy.enable_shorts and name not in strategy.short_positions and mu20 <= -0.003:
                        #     # Ensure no long remains
                        #     if name in strategy.positions and strategy.positions[name]['shares'] > 0:
                        #         pos = strategy.positions[name]
                        #         print(f"    ⚠️  LIQUIDATE LONG BEFORE SHORT: {name} at t={t}")
                        #         strategy.sell(name, price_mid, pos['shares'] * price_mid, t)
                        #     strength = min(1.0, abs(mu20) / 0.03)
                        #     cap = strategy.portfolio_value * 0.01
                        #     amount = min(cap * strength, strategy.cash)
                        #     if amount > 0:
                        #         print(f"    🔻 ENTER SHORT: {name} at t={t}, mu={mu20:.4f}, price=${price_mid:.2f}, amount=${amount:.0f}")
                        #         proceeds = strategy.short_open(name, price_mid, amount, t=t, snr=0.0)
                        #         if proceeds > 0:
                        #             bid = price_mid * (1 - getattr(strategy, 'short_spread_fee', 0.003))
                        #             shares = proceeds / bid
                        #             short_queues[name].append({'shares': shares, 'price': bid, 'time': t, 'snr': 0.0})

        # EOD liquidation at predict_end_minute
        eod_t = max_t
        for idx in stock_indices:
            name = dataset.name_date[idx]
            price_mid = dataset.data[idx][eod_t, 3].item()

            # Liquidate longs
            # Sell whatever shares remain in inventory for P&L FIFO
            if buy_queues[name]:
                bid = price_mid * (1 - getattr(strategy, 'spread_fee', 0.0))
                while buy_queues[name]:
                    lot = buy_queues[name].pop(0)
                    gain = (bid - lot['price']) * lot['shares']
                    batch_realized += gain
                # Close strategy position if still present
                if name in strategy.positions and strategy.positions[name]['shares'] > 0:
                    pos = strategy.positions[name]
                    strategy.sell(name, price_mid, pos['shares'] * price_mid, eod_t)

            # SHORTING DISABLED - Comment out short covering
            # if short_queues[name]:
            #     ask = price_mid * (1 + getattr(strategy, 'short_spread_fee', 0.003))
            #     while short_queues[name]:
            #         lot = short_queues[name].pop(0)
            #         gain = (lot['price'] - ask) * lot['shares']
            #         batch_realized += gain
            #     if name in strategy.short_positions and strategy.short_positions[name]['shares'] > 0:
            #         strategy.short_cover(name, price_mid, eod_t)

        # Update portfolio across days
        current_portfolio += batch_realized
        total_realized_pnl += batch_realized

        # Print trading summary for this day
        total_trades = len([t for t in strategy.trade_log if t[1] in ['BUY', 'SELL']])  # Only BUY/SELL now
        buy_trades = len([t for t in strategy.trade_log if t[1] == 'BUY'])
        sell_trades = len([t for t in strategy.trade_log if t[1] == 'SELL'])
        # short_trades = len([t for t in strategy.trade_log if t[1] == 'SHORT'])  # SHORTING DISABLED
        # cover_trades = len([t for t in strategy.trade_log if t[1] == 'COVER'])  # SHORTING DISABLED
        
        print(f"📄 Day {day_idx} {date} | Realized P&L: ${batch_realized:,.2f} | Ending: ${current_portfolio:,.2f}")
        print(f"    📊 Trading Summary: {total_trades} total trades")
        print(f"    🟢 Long entries: {buy_trades}, exits: {sell_trades}")
        # print(f"    🔻 Short entries: {short_trades}, covers: {cover_trades}")  # SHORTING DISABLED
        print(f"    💰 Final positions: {len(strategy.positions)} longs")  # No more shorts
        
        # Display prediction history for each stock
        print(f"\n📊 PREDICTION HISTORY FOR EACH STOCK:")
        print(f"    Prediction period: {min(prediction_timestamps) if prediction_timestamps else 'N/A'} to {max(prediction_timestamps) if prediction_timestamps else 'N/A'} minutes")
        print(f"    Total prediction rounds: {len(prediction_timestamps)}")
        
        # Show detailed predictions for first 5 stocks (to avoid overwhelming output)
        sample_stocks = list(stock_predictions.keys())[:5]
        for stock_name in sample_stocks:
            if stock_predictions[stock_name]:
                preds = stock_predictions[stock_name]
                print(f"\n    📈 {stock_name}:")
                
                # Show prediction statistics
                predictions = [p['prediction'] for p in preds]
                prices = [p['price'] for p in preds]
                timestamps = [p['timestamp'] for p in preds]
                
                print(f"      Predictions: {len(preds)} total")
                print(f"      Range: [{min(predictions):.4f}, {max(predictions):.4f}]")
                print(f"      Mean: {np.mean(predictions):.4f}")
                print(f"      Std: {np.std(predictions):.4f}")
                
                # Show prediction evolution over time
                print(f"      Evolution:")
                for i in range(0, len(preds), max(1, len(preds)//5)):  # Show ~5 key points
                    p = preds[i]
                    action = 'BUY' if p['prediction'] >= 0.001 else ('SHORT' if p['prediction'] <= -0.003 else 'HOLD')
                    print(f"        t={p['timestamp']}: {p['prediction']:.4f} ({action}) @ ${p['price']:.2f}")
                
                # Show final prediction
                if preds:
                    final_pred = preds[-1]
                    final_action = 'BUY' if final_pred['prediction'] >= 0.001 else ('SHORT' if final_pred['prediction'] <= -0.003 else 'HOLD')
                    print(f"      Final: t={final_pred['timestamp']}: {final_pred['prediction']:.4f} ({final_action}) @ ${final_pred['price']:.2f}")
        
        # Show summary statistics for all stocks
        if stock_predictions:
            all_predictions = []
            for stock_name, preds in stock_predictions.items():
                if preds:
                    all_predictions.extend([p['prediction'] for p in preds])
            
            if all_predictions:
                print(f"\n    📊 OVERALL PREDICTION STATISTICS:")
                print(f"      Total predictions: {len(all_predictions)}")
                print(f"      Global range: [{min(all_predictions):.4f}, {max(all_predictions):.4f}]")
                print(f"      Global mean: {np.mean(all_predictions):.4f}")
                print(f"      Global std: {np.std(all_predictions):.4f}")
                
                # Count actions across all predictions - SHORTING DISABLED
                buy_count = sum(1 for p in all_predictions if p >= 0.001)
                # short_count = sum(1 for p in all_predictions if p <= -0.003)  # SHORTING DISABLED
                hold_count = sum(1 for p in all_predictions if p < 0.001)  # All non-buy are holds
                
                print(f"      Action distribution: BUY={buy_count}, HOLD={hold_count}")  # No more SHORT
                print(f"      BUY rate: {buy_count/len(all_predictions)*100:.1f}%")
                # print(f"      SHORT rate: {short_count/len(all_predictions)*100:.1f}%")  # SHORTING DISABLED
                print(f"      HOLD rate: {hold_count/len(all_predictions)*100:.1f}%")
        
        # Save predictions to CSV for detailed analysis
        csv_filename = f"predictions_{date}.csv"
        save_predictions_to_csv(stock_predictions, csv_filename)
        print(f"    💾 Predictions saved to: {csv_filename}")
        
        # Create prediction vs actual plots for 2 random stocks
        print(f"    📊 Creating prediction vs actual plots...")
        plot_predictions_vs_actuals(stock_predictions, dataset, stock_indices, date, sample_size=2)

    print("".ljust(60, "="))
    print("🎯 FINAL SUMMARY")
    print(f"Initial: ${initial_portfolio:,.2f}")
    print(f"Final:   ${current_portfolio:,.2f}")
    print(f"Return:  ${total_realized_pnl:,.2f} ({(total_realized_pnl/initial_portfolio)*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Backtest with continuous predictions (train_two style) and bare strategy execution window")
    parser.add_argument('--h5', type=str, default='h5_test_training_two.h5', help='Path to HDF5 file')
    parser.add_argument('--init', type=float, default=100000.0, help='Initial portfolio value')
    parser.add_argument('--seq_len', type=int, default=30, help='Initial sequence length (predictions start at 10:00 AM)')
    parser.add_argument('--pred_len', type=int, default=20, help='Prediction horizon length (minutes)')
    parser.add_argument('--trade_start', type=int, default=90, help='Start trading minute (11:00 AM)')
    parser.add_argument('--trade_end', type=int, default=210, help='End trading minute (1:00 PM)')
    parser.add_argument('--pred_end', type=int, default=360, help='End of prediction (3:30 PM)')
    parser.add_argument('--stock_batch_size', type=int, default=512, help='Per-GPU inference batch for stocks')
    
    try:
        args = parser.parse_known_args()[0]
    except SystemExit:
        # Handle case when running in notebook (no command line args)
        print("⚠️  No command line arguments found, using defaults for notebook execution")
        args = argparse.Namespace(
            h5='h5_test_training_two.h5',
            init=100000.0,
            seq_len=30,
            pred_len=20,
            trade_start=90,
            trade_end=210,
            pred_end=360,
            stock_batch_size=512
        )

    # Note: With seq_len=30, predictions begin at 10:00 AM (t=30).
    # Trading starts at 11:00 AM (t=90) after 1 hour of prediction calibration.
    run_backtest(
        h5_path=args.h5,
        initial_portfolio=args.init,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        predict_end_minute=args.pred_end,
        trade_start_minute=args.trade_start,
        trade_end_minute=args.trade_end,
        stock_batch_size=args.stock_batch_size,
    )


if __name__ == "__main__":
	main() 
