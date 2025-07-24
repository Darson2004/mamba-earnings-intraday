import argparse
import h5py
import numpy as np
import torch
from dataset_gpu import ClosePrice
from strategy_normalized import TradingStrategy
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Backtest trading strategy with optional model.")
    parser.add_argument('--model', type=str, default=None, help='Path to model weights (e.g., mambastock_original.pth)')
    args = parser.parse_args()

    # --- CONFIG ---
    h5_path = "h5_rty_data_test.h5"
    seq_len = 30
    pred_len = 2
    initial_portfolio = 100000

    # Load dataset
    dataset = ClosePrice(h5_path)
    print(f"Loaded dataset: {dataset.n_stocks} stocks, {dataset.day_length} minutes per day")
    
    # Display close price and pct_chg for each minute for each stock
    for stock_idx, name in enumerate(dataset.name_date):
        print(f"\n=== {name} ===")
        stock_data = dataset.data[stock_idx].cpu().numpy()
        print("Minute | Close | Pct_Change")
        for t in range(dataset.day_length):
            close = stock_data[t, 3]
            pct_chg = stock_data[t, 10]
            print(f"{t:3d} | {close:8.3f} | {pct_chg:8.4f}")
        if stock_idx > 2:
            print("... (truncated)")
            break
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model:
        print(f"Loading model weights from {args.model}")
        from mambastock_model import MambaStock
        model = MambaStock(input_size=13, seq_len=seq_len, pred_len=pred_len)
        model.load_state_dict(torch.load(args.model, map_location=device), strict=False)
        model = model.to(device)
        model.eval()
        print("Using real model for predictions.")
    else:
        print("No model specified, using dummy model (all predictions will be zero).")
        class DummyModel(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(x.shape[0], 1, 1)
        model = DummyModel().to(device)
    considered_stocks = list(range(dataset.n_stocks))
    strategy = TradingStrategy(
        model=model,
        device=device,
        dataset=dataset,
        considered_stocks=considered_stocks,
        portfolio_value=initial_portfolio,
        seq_len=seq_len,
        pred_len=pred_len
    )
    
    # Run the strategy to get the trade log
    trade_log, final_positions = strategy.run_day()
    # Remove trade log and FIFO realized P&L prints
    # print("\n=== TRADE LOG ===")
    # for entry in trade_log:
    #     print(entry)
    # FIFO matching for realized P&L
    # print("\n=== FIFO REALIZED P&L ===")
    # For each stock, maintain a FIFO queue of buys
    buy_queues = {name: [] for name in dataset.name_date}
    realized_pnl = 0.0
    trade_details = []
    for entry in trade_log:
        symbol, action, price, shares, capital, trade_time = entry
        if action == 'BUY':
            buy_queues[symbol].append({'shares': shares, 'price': price, 'capital': capital, 'time': trade_time})
        elif action == 'SELL':
            shares_to_sell = shares
            while shares_to_sell > 0 and buy_queues[symbol]:
                buy = buy_queues[symbol][0]
                matched_shares = min(buy['shares'], shares_to_sell)
                buy_price = buy['price']
                sell_price = price
                gain = (sell_price - buy_price) * matched_shares
                realized_pnl += gain
                trade_details.append({
                    'symbol': symbol,
                    'buy_time': buy['time'],
                    'buy_price': buy_price,
                    'sell_time': trade_time,
                    'sell_price': sell_price,
                    'shares': matched_shares,
                    'gain': gain,
                    'return_pct': 100 * (sell_price - buy_price) / buy_price
                })
                buy['shares'] -= matched_shares
                shares_to_sell -= matched_shares
                if buy['shares'] <= 1e-8:
                    buy_queues[symbol].pop(0)
    # At end of day, sell all remaining positions at the highest price between 3:30 and 3:45 (minutes 360 to 375)
    # Get normalization params for all features
    normalization_params = dataset.normalization_params
    # When using close price for trading logic, normalize it on-the-fly
    # Example for EOD sell:
    for symbol, queue in buy_queues.items():
        if not queue:
            continue
        if symbol in dataset.name_date:
            idx = dataset.name_date.index(symbol)
        else:
            try:
                idx = int(symbol)
            except:
                continue
        # Find the highest close price between 3:30pm and 3:45pm (minutes 360 to 375)
        price_window = dataset.data[idx][360:376, 3].cpu().numpy()  # 376 is exclusive, so includes 375
        # Normalize price_window
        col_mean = normalization_params[3]['mean']
        col_std = normalization_params[3]['std']
        if col_std > 1e-8:
            price_window_norm = (price_window - col_mean) / col_std
        else:
            price_window_norm = price_window * 0.0
        if len(price_window_norm) == 0:
            print(f"WARNING: No price data for {symbol} in 3:30-3:45 window, using last available close.")
            sell_price = dataset.data[idx][-1, 3].item()
            sell_price_norm = (sell_price - col_mean) / col_std if col_std > 1e-8 else 0.0
        else:
            sell_price_norm = np.max(price_window_norm)
        for buy in queue:
            matched_shares = buy['shares']
            buy_price = buy['price']
            gain = (sell_price_norm - buy_price) * matched_shares
            realized_pnl += gain
            trade_details.append({
                'symbol': symbol,
                'buy_time': buy['time'],
                'buy_price': buy_price,
                'sell_time': 'EOD_MAX_3:30-3:45',
                'sell_price': sell_price_norm,
                'shares': matched_shares,
                'gain': gain,
                'return_pct': 100 * (sell_price_norm - buy_price) / buy_price if buy_price != 0 else 0.0
            })
    # Print trade details
    # for td in trade_details:
    #     print(f"{td['symbol']} | Buy@{td['buy_time']} {td['buy_price']:.2f} -> Sell@{td['sell_time']} {td['sell_price']:.2f} | Shares: {td['shares']:.2f} | Gain: ${td['gain']:.2f} | Return: {td['return_pct']:.2f}%")
    # Final portfolio value
    final_portfolio = initial_portfolio + realized_pnl
    print(f"\nInitial portfolio: ${initial_portfolio:,.2f}")
    print(f"Final portfolio:   ${final_portfolio:,.2f}")
    print(f"Total return:      ${realized_pnl:,.2f} ({100*realized_pnl/initial_portfolio:.2f}%)")

    # Track portfolio value at every minute
    portfolio_values = []
    cash = float(initial_portfolio)
    positions = {name: [] for name in dataset.name_date}  # FIFO queues for each stock
    time_to_trades = {t: [] for t in range(dataset.day_length)}
    for entry in trade_log:
        symbol, action, price, shares, capital, trade_time = entry
        if trade_time is not None and 0 <= trade_time < dataset.day_length:
            time_to_trades[int(trade_time)].append(entry)
    for t in range(dataset.day_length):
        # Apply trades at this minute
        for entry in time_to_trades[t]:
            symbol, action, price, shares, capital, trade_time = entry
            if action == 'BUY':
                cash -= capital
                positions[symbol].append({'shares': shares, 'price': price})
            elif action == 'SELL':
                cash += capital
                shares_to_sell = shares
                while shares_to_sell > 0 and positions[symbol]:
                    lot = positions[symbol][0]
                    matched_shares = min(lot['shares'], shares_to_sell)
                    lot['shares'] -= matched_shares
                    shares_to_sell -= matched_shares
                    if lot['shares'] <= 1e-8:
                        positions[symbol].pop(0)
        # Calculate current portfolio value at this minute
        total_position_value = 0.0
        for symbol, lots in positions.items():
            if not lots:
                continue
            idx = dataset.name_date.index(symbol)
            current_price = dataset.data[idx][t, 3].item()
            for lot in lots:
                total_position_value += lot['shares'] * current_price
        portfolio_values.append(float(cash + total_position_value))
    # Plot portfolio value over time
    plt.figure(figsize=(12, 6))
    plt.plot(range(dataset.day_length), portfolio_values, label='Portfolio Value')
    plt.xlabel('Minute of Day')
    plt.ylabel('Portfolio Value ($)')
    plt.title('Total Portfolio Value Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # Per-stock net gain/loss and percent change (as text only)
    print("\n=== PER-STOCK NET GAIN/LOSS ===")
    print("DEBUG: trade_details length:", len(trade_details))
    print("DEBUG: trade_details sample:", trade_details[:3])
    stock_invested = {}
    stock_realized = {}
    for td in trade_details:
        symbol = td['symbol']
        stock_invested.setdefault(symbol, 0.0)
        stock_realized.setdefault(symbol, 0.0)
        stock_invested[symbol] += td['buy_price'] * td['shares']
        stock_realized[symbol] += td['sell_price'] * td['shares']
    print("DEBUG: stock_invested keys:", list(stock_invested.keys()))
    print("DEBUG: stock_realized keys:", list(stock_realized.keys()))
    try:
        for symbol in stock_invested:
            invested = stock_invested[symbol]
            realized = stock_realized[symbol]
            net = realized - invested
            pct = (net / invested * 100) if invested > 0 else 0.0
            print(f"DEBUG: {symbol}: Invested={invested}, Realized={realized}, Net={net}, Pct={pct}")
            print(f"{symbol}: Invested ${invested:.2f}, Realized ${realized:.2f}, Net Gain/Loss ${net:.2f} ({pct:.2f}%)")
    except Exception as e:
        print(f"EXCEPTION in per-stock net gain/loss loop: {e}")

if __name__ == "__main__":
    main() 