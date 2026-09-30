import torch
import numpy as np

# === Synthetic Dataset ===
class SyntheticDataset:
    def __init__(self, n_stocks=1, day_length=390, seq_len=30):
        self.n_stocks = n_stocks
        self.day_length = day_length
        self.seq_len = seq_len
        self.pred_len = 25
        self.data = self._generate_random_data()
        self.name_date = [f"Stock{i}" for i in range(n_stocks)]

    def _generate_random_data(self):
        data = []
        for _ in range(self.n_stocks):
            stock_data = torch.zeros(self.day_length, 7)
            price = 100.0
            for t in range(self.day_length):
                delta = np.random.normal(0, 0.5)
                price += delta
                pct_chg = delta / price
                volume = np.random.randint(1000, 5000)
                stock_data[t, 0] = price       # open
                stock_data[t, 1] = price + 0.5 # high
                stock_data[t, 2] = price - 0.5 # low
                stock_data[t, 3] = price       # close
                stock_data[t, 4] = pct_chg     # change
                stock_data[t, 5] = volume      # volume
                stock_data[t, 6] = pct_chg     # duplicated for compatibility
            data.append(stock_data)
        return data

# === Dummy Model ===
class DummyModel(torch.nn.Module):
    def __init__(self, pred_len=25):
        super().__init__()
        self.pred_len = pred_len

    def forward(self, x):
        batch_size = x.shape[0]
        return torch.rand(batch_size, self.pred_len, 7) * 0.01  # random small % changes

# === Strategy Function ===
def strategy(model, dataset, seq_len, pred_len, device, initial_portfolio=10000, stock_indices=None, verbose=True):
    model.eval()
    portfolio_value = initial_portfolio
    holdings = {}
    daily_trades = {}

    buy_threshold = 0.1
    extra_buy_threshold = 0.05
    sell_threshold = 0.05
    sharpe_buy_threshold = 0.1
    sharpe_sell_threshold = 0.05
    interval_minutes = 15
    final_sell_start = 375
    final_sell_end = 389

    def calculate_sharpe_ratio(predictions):
        if len(predictions) < 2:
            return 0.0
        returns = np.diff(predictions)
        std_return = np.std(returns)
        if std_return == 0:
            return 0.0
        return np.mean(returns) / std_return

    def get_price_in_interval(stock_data, start_time, end_time, fn=torch.min):
        if start_time >= len(stock_data) or end_time > len(stock_data):
            return float('inf') if fn == torch.min else 0.0
        return fn(stock_data[start_time:end_time, 3]).item()

    def buy_stock(stock_idx, amount, price):
        nonlocal portfolio_value
        shares = amount / price
        holdings.setdefault(stock_idx, {'shares': 0, 'avg_price': 0})
        pos = holdings[stock_idx]
        total = pos['shares'] + shares
        pos['avg_price'] = (pos['avg_price'] * pos['shares'] + price * shares) / total
        pos['shares'] = total
        daily_trades.setdefault(stock_idx, {'buys': 0, 'sells': 0})['buys'] += 1
        portfolio_value -= amount
        return shares

    def sell_stock(stock_idx, amount, price):
        nonlocal portfolio_value
        if stock_idx not in holdings or holdings[stock_idx]['shares'] <= 0:
            return 0
        shares = min(amount / price, holdings[stock_idx]['shares'])
        holdings[stock_idx]['shares'] -= shares
        daily_trades.setdefault(stock_idx, {'buys': 0, 'sells': 0})['sells'] += 1
        portfolio_value += shares * price
        return shares

    if stock_indices is None:
        stock_indices = range(dataset.n_stocks)

    for stock_idx in stock_indices:
        stock_data = dataset.data[stock_idx]
        print(f"\nRunning strategy on {dataset.name_date[stock_idx]}")
        for time_idx in range(seq_len, dataset.day_length - pred_len):
            with torch.no_grad():
                X = stock_data[time_idx - seq_len:time_idx, :].unsqueeze(0).to(device)
                pred = model(X)
                predictions = pred[0, :, 6].cpu().numpy()

            projected_return = predictions[-1] - predictions[0]
            sharpe_ratio = calculate_sharpe_ratio(predictions)
            interval_start = (time_idx // interval_minutes) * interval_minutes
            interval_end = min(interval_start + interval_minutes, dataset.day_length)

            if projected_return > 0 and sharpe_ratio > sharpe_buy_threshold:
                price = get_price_in_interval(stock_data, interval_start, interval_end, fn=torch.min)
                if price < float('inf') and portfolio_value > 0:
                    shares = buy_stock(stock_idx, portfolio_value * buy_threshold, price)
                    print(f"  {time_idx}: BUY {shares:.2f} @ {price:.2f} (Sharpe={sharpe_ratio:.2f})")

            elif sharpe_ratio < sharpe_sell_threshold:
                price = get_price_in_interval(stock_data, interval_start, interval_end, fn=torch.max)
                if price > 0:
                    shares = sell_stock(stock_idx, portfolio_value * sell_threshold, price)
                    if shares > 0:
                        print(f"  {time_idx}: SELL {shares:.2f} @ {price:.2f} (Sharpe={sharpe_ratio:.2f})")

            if final_sell_start <= time_idx <= final_sell_end:
                if stock_idx in holdings and holdings[stock_idx]['shares'] > 0:
                    price = stock_data[time_idx, 3].item()
                    shares = sell_stock(stock_idx, holdings[stock_idx]['shares'] * price, price)
                    if shares > 0:
                        print(f"  {time_idx}: FINAL SELL {shares:.2f} @ {price:.2f}")

    # Portfolio final value
    final_value = portfolio_value
    for stock_idx, pos in holdings.items():
        final_value += pos['shares'] * dataset.data[stock_idx][-1, 3].item()

    print("\n=== STRATEGY SUMMARY ===")
    print(f"Initial portfolio: ${initial_portfolio:.2f}")
    print(f"Final portfolio:   ${final_value:.2f}")
    print(f"Total return:      {((final_value - initial_portfolio) / initial_portfolio * 100):.2f}%")

# === Run It ===
if __name__ == "__main__":
    device = torch.device("cpu")
    dataset = SyntheticDataset(n_stocks=1)
    model = DummyModel(pred_len=25).to(device)
    strategy(model, dataset, seq_len=30, pred_len=25, device=device)
