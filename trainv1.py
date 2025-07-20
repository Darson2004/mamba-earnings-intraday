from mambastock_model import MambaStock
from dataset import ClosePrice, TimeAlignedSampler, collate_fn
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

# --- CONFIG ---
h5_path = "h5_EST_earnings_ohlcv_data_test.h5"
seq_len = 30
pred_len = 25
batch_size = 32
epochs = 1
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LOAD DATA ---
dataset = ClosePrice(h5_path)
print(f"Dataset loaded: {len(dataset)} total items, {dataset.n_stocks} stocks")
print(f"Starting all calculations after {seq_len} data points (time index {seq_len}) to exclude first {seq_len} points")

# Optimized DataLoader
sampler = TimeAlignedSampler(dataset, batch_size=batch_size, start_time=seq_len, end_time=dataset.day_length-pred_len, shuffle=True)
loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn, pin_memory=True, num_workers=2)

# --- INIT MODEL ---
model = MambaStock(input_size=7, seq_len=seq_len, pred_len=pred_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# --- STRATEGY WITH DEBUGGING ---
def run_strategy(dataset, seq_len, pred_len, predict_fn, initial_portfolio=10000, stock_indices=None, verbose=True, label="MODEL"):
    import time
    start_time = time.time()
    portfolio_value = initial_portfolio
    buy_threshold = 0.05
    sell_threshold = 0.02
    sharpe_buy_threshold = 0.3  # lowered for debug
    sharpe_sell_threshold = 0.1
    final_sell_start = 330
    final_sell_end = 345

    def calculate_sharpe_ratio(predictions, risk_free_rate=0.043/252):
        if len(predictions) < 2:
            return 0.0
        returns = np.diff(predictions)
        if len(returns) == 0:
            return 0.0
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return 0.0
        return (mean_return - risk_free_rate) / std_return

    def buy_stock(amount, price):
        nonlocal portfolio_value
        shares = amount / price
        portfolio_value -= amount
        return shares

    def sell_stock(shares, price):
        nonlocal portfolio_value
        portfolio_value += shares * price

    if stock_indices is None:
        stock_indices = range(dataset.n_stocks)

    for stock_idx in stock_indices:
        stock_data = dataset.data[stock_idx]
        stock_name = dataset.name_date[stock_idx]
        shares_held = 0
        for t in range(seq_len, dataset.day_length - pred_len):
            history = stock_data[t-seq_len:t, :].cpu().numpy()
            preds = predict_fn(history, pred_len)
            if len(preds) < pred_len:
                continue
            window = preds[:pred_len]
            sharpe = calculate_sharpe_ratio(window)
            projected_return = window[-1] - window[0]
            returns = np.diff(window)
            price = stock_data[t, 3].item()

            if verbose:
                print(f"[{stock_name} @ t={t}] Sharpe={sharpe:.4f}, Return={projected_return:.4f}, Price={price:.2f}")
                print(f"    Predictions: {window}")
                print(f"    Returns: {returns}")

            if projected_return > 0 and sharpe > sharpe_buy_threshold:
                buy_amt = portfolio_value * buy_threshold
                if buy_amt > 0:
                    shares_held += buy_stock(buy_amt, price)
                    print(f"  ✅ BUY at t={t}, price={price:.2f}, Sharpe={sharpe:.2f}, Return={projected_return:.2f}")

            if sharpe < sharpe_sell_threshold and projected_return < 0 and shares_held > 0:
                sell_amt = min(shares_held, (portfolio_value + shares_held * price) * sell_threshold / price)
                sell_stock(sell_amt, price)
                shares_held -= sell_amt
                print(f"  ❌ SELL at t={t}, price={price:.2f}, Sharpe={sharpe:.2f}, Return={projected_return:.2f}")

        final_prices = stock_data[final_sell_start:final_sell_end+1, 3].cpu().numpy()
        if shares_held > 0 and len(final_prices) > 0:
            max_price = np.max(final_prices)
            sell_stock(shares_held, max_price)
            shares_held = 0

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"{label} STRATEGY RESULTS")
    print(f"{'='*60}")
    print(f"Final Portfolio: ${portfolio_value:,.2f}")
    print(f"Total Return: {((portfolio_value - initial_portfolio) / initial_portfolio * 100):.2f}%")
    print(f"Strategy run time: {elapsed:.2f} seconds")
    return portfolio_value
