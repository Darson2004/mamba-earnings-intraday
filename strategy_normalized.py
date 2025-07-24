# trading_strategy.py
import torch
import numpy as np
from collections import defaultdict

class TradingStrategy:
    """
    Batch-compatible trading strategy for post-training evaluation.
    Operates on a batch of stocks, using rolling predictions and portfolio management logic.
    Compatible with train.py and dataset.py.
    """
    def __init__(self, model, device, dataset, considered_stocks, portfolio_value, seq_len=30, pred_len=2, confidence_level=0.95, min_hold_minutes=1):
        self.model = model
        self.device = device
        self.dataset = dataset
        self.considered_stocks = considered_stocks  # list of indices or names
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.confidence_level = confidence_level
        self.portfolio_value = portfolio_value
        self.cash = portfolio_value  # Track available cash
        self.positions = {}  # {symbol: {'shares': float, 'avg_price': float, 'capital': float, 'entry_price': float}}
        self.distinct_stocks_bought = set()
        self.trade_log = []  # List of trade actions
        self.stock_returns = defaultdict(float)  # Track net returns per stock
        self.stock_prediction_mode = defaultdict(lambda: 2)  # 2 or 10 min rolling
        self.stock_last_action_time = defaultdict(lambda: 0)
        self.stock_last_entry_price = {}  # For trimming logic
        self.min_hold_minutes = min_hold_minutes
        self.last_buy_time = {}  # Track last buy time for each stock
        
        # NEW: Track price history for stagnation detection
        self.price_history = defaultdict(list)  # {symbol: [(time, price), ...]}
        self.stagnation_threshold = 15  # 15 minutes of staying in range
        self.min_profit_for_stagnation_sell = 0.01  # 1% minimum profit to trigger stagnation sell
        self.stagnation_range = 0.002  # ±0.2% range for stagnation
        self.stagnation_start_price = {}  # Track starting price when entering stagnation range
        self.stagnation_start_time = {}  # Track when stagnation period began
        
        # NEW: Stop loss functionality
        self.stop_loss_threshold = -0.015  # -1.5% stop loss
        self.stop_loss_sold = set()  # Track stocks sold due to stop loss
        self.continue_monitoring_after_stop_loss = True  # Continue monitoring after stop loss

    def compute_sharpe(self, delta_future, returns):
        # returns: torch tensor, delta_future: float
        # Compute std of 2-minute changes in pct_chg over the last seq_len
        changes = []
        arr = returns.cpu().numpy() if hasattr(returns, 'cpu') else np.array(returns)
        for i in range(len(arr) - 2):
            changes.append(arr[i+2] - arr[i])
        if len(changes) == 0:
            std_2min = 1e-8  # Avoid div by zero
        else:
            std_2min = np.std(changes, ddof=0)
        return ((delta_future - 0.000180556) / std_2min) if std_2min > 0 else 0.0

    def compute_confidence_interval(self, samples):
        # samples: torch tensor
        mu = samples.mean()
        sigma = samples.std(unbiased=False)
        z = 1.96  # for 95% CI
        lower = mu - z * sigma
        upper = mu + z * sigma
        return mu, lower, upper, sigma

    def cap_check(self, symbol, amount):
        max_cap = self.portfolio_value / max(len(self.distinct_stocks_bought), 1)
        current_cap = self.positions.get(symbol, {}).get('capital', 0.0)
        return min(amount, max(0.0, max_cap - current_cap))

    def buy(self, symbol, price, amount, t=None):
        if price <= 0:
            print(f"WARNING: Invalid buy price for {symbol} at t={t}")
            return 0.0
        # Prevent multiple buys at the same timestamp
        if t is not None and symbol in self.last_buy_time and self.last_buy_time[symbol] == t:
            return 0.0
        capital = min(self.cap_check(symbol, amount), self.cash)
        if capital <= 0:
            return 0.0
        shares = capital / price
        if shares <= 0:
            return 0.0
        self.cash -= capital
        self.distinct_stocks_bought.add(symbol)
        if symbol in self.positions:
            pos = self.positions[symbol]
            total_shares = pos['shares'] + shares
            avg_price = (pos['avg_price'] * pos['shares'] + price * shares) / total_shares
            self.positions[symbol].update({'shares': total_shares, 'avg_price': avg_price, 'capital': pos['capital'] + capital})
        else:
            self.positions[symbol] = {'shares': shares, 'avg_price': price, 'capital': capital, 'entry_price': price}
        self.trade_log.append((symbol, 'BUY', price, shares, capital, t))
        if t is not None:
            self.last_buy_time[symbol] = t
        return capital

    def sell(self, symbol, price, amount, t=None):
        if price <= 0:
            print(f"WARNING: Invalid sell price for {symbol} at t={t}")
            return 0.0
        # Enforce minimum holding period
        if symbol in self.last_buy_time and t is not None:
            if t - self.last_buy_time[symbol] < self.min_hold_minutes:
                return 0.0
        if symbol not in self.positions:
            return 0.0
        pos = self.positions[symbol]
        max_shares_to_sell = pos['shares']
        shares_to_sell = min(max_shares_to_sell, amount / price)
        if shares_to_sell <= 0:
            return 0.0
        sell_cap = shares_to_sell * price
        self.cash += sell_cap
        # Realized P&L is (sell_price - avg_price) * shares_to_sell
        pos['shares'] -= shares_to_sell
        pos['capital'] -= shares_to_sell * pos['avg_price']
        if pos['shares'] <= 1e-8:
            del self.positions[symbol]
        self.trade_log.append((symbol, 'SELL', price, shares_to_sell, sell_cap, t))
        return sell_cap

    def predict_with_ci(self, input_seq, pred_len):
        """
        Monte Carlo dropout prediction for pred_len steps, returns mean, lower, upper, std for each step.
        Uses torch tensors for all calculations.
        """
        self.model.train()  # Enable dropout
        preds = []
        lowers = []
        uppers = []
        stds = []
        n_samples = 20
        seq = input_seq.clone()
        for t in range(pred_len):
            samples = []
            X = seq[-self.seq_len:].unsqueeze(0).to(self.device)
            with torch.no_grad():
                for _ in range(n_samples):
                    pred = self.model(X).squeeze()
                    samples.append(pred)
            samples_tensor = torch.stack(samples)
            mu, lower, upper, sigma = self.compute_confidence_interval(samples_tensor)
            # Check for NaN values and handle them
            if torch.isnan(mu) or torch.isnan(lower) or torch.isnan(upper):
                print(f"WARNING: NaN detected in predictions at step {t}")
                mu = torch.tensor(0.0, device=self.device)
                lower = torch.tensor(-0.01, device=self.device)
                upper = torch.tensor(0.01, device=self.device)
                sigma = torch.tensor(0.005, device=self.device)
            preds.append(mu)
            lowers.append(lower)
            uppers.append(upper)
            stds.append(sigma)
            # For next step, append predicted value to sequence
            next_pred = seq[-1, :].clone()
            next_pred[10] = mu  # pct_chg index
            seq = torch.cat([seq, next_pred.unsqueeze(0)], dim=0)
        self.model.eval()
        # Return as torch tensors
        return torch.stack(preds), torch.stack(lowers), torch.stack(uppers), torch.stack(stds)

    def run_day(self):
        """
        Simulate a trading day for all considered stocks, applying the strategy logic in batch mode.
        All stocks are processed in parallel at each time step.
        """
        n_stocks = len(self.considered_stocks)
        stock_indices = self.considered_stocks
        stock_names = [self.dataset.name_date[idx] if hasattr(self.dataset, 'name_date') else str(idx) for idx in stock_indices]
        day_length = self.dataset.day_length
        # DEBUG: Print mapping and data info for each stock
        for i, name in enumerate(stock_names):
            idx = stock_indices[i]
            print(f"DEBUG INIT: stock_names[{i}] = {name}, stock_indices[{i}] = {idx}, data shape = {self.dataset.data[idx].shape}")
            print(f"DEBUG INIT: {name} close price min={self.dataset.data[idx][:, 3].min().item()}, max={self.dataset.data[idx][:, 3].max().item()}")
        # Initialize per-stock state
        seqs = [self.dataset.data[idx][:self.seq_len, :].clone().to(self.device) for idx in stock_indices]
        t = self.seq_len
        pred_modes = [20 for _ in stock_indices]  # Always 20-minute rolling prediction
        done = [False for _ in stock_indices]  # Track if stock is 'done' (e.g., all sold)
        sold_eod = set()  # Track stocks sold in 3:30-3:45 window
        # Main loop: process all stocks in parallel
        while t < min(day_length, self.seq_len + 330):
            # Prepare batch for prediction
            batch_X = []
            for i, seq in enumerate(seqs):
                if done[i]:
                    batch_X.append(torch.zeros_like(seq))
                else:
                    batch_X.append(seq)
            # Per-stock rolling normalization for each window
            batch_X_norm = []
            for i, seq in enumerate(batch_X):
                mean = seq.mean(dim=0, keepdim=True)
                std = seq.std(dim=0, keepdim=True)
                std = torch.where(std > 1e-8, std, torch.ones_like(std))
                seq_norm = (seq - mean) / std
                batch_X_norm.append(seq_norm[-self.seq_len:])  # Always use last seq_len for model
            batch_X_norm = torch.stack(batch_X_norm).to(self.device)  # [batch, seq_len, features]
            # Model prediction for next 20 minutes
            with torch.no_grad():
                batch_preds = self.model(batch_X_norm)  # [batch, 20]
            # Now process each stock's logic (simplified for rolling prediction)
            for i, name in enumerate(stock_names):
                if done[i]:
                    continue
                # Use normalized close price for trading logic
                current_price_norm = batch_X_norm[i, -1, 3].item()
                delta_future = batch_preds[i, 0].item()  # Use first prediction as signal
                print(f"Stock {name} at t={t}: delta_future={delta_future:.6f}, price={current_price_norm:.2f}")
                # Example buy logic (customize as needed)
                if delta_future > 0:
                    buy_amount = self.portfolio_value * 0.01
                    actual_bought = self.buy(name, current_price_norm, buy_amount, t=t)
                    print(f"  → BUY attempted: {buy_amount:.2f}, actual: {actual_bought:.2f}")
                else:
                    print(f"  → NO BUY: delta_future <= 0")
            # Advance all sequences by 20 (append actuals, grow window)
            for i in range(n_stocks):
                if done[i]:
                    continue
                advance = 20
                if t + advance < day_length:
                    seqs[i] = torch.cat([seqs[i], self.dataset.data[stock_indices[i]][t:t+advance, :].clone().to(self.device)], dim=0)
                else:
                    done[i] = True
            t += 20
        return self.trade_log, self.positions

    def check_price_stagnation(self, symbol, current_price, current_time):
        """
        Check if the stock price has stayed within ±0.2% range for 15+ minutes
        Returns True if stagnant (within range for 15+ min), False otherwise
        Resets if price ever exits the range
        """
        # Check if price is outside the current stagnation range
        if symbol in self.stagnation_start_price:
            start_price = self.stagnation_start_price[symbol]
            price_change = abs(current_price - start_price) / start_price
            
            # If price exits the ±0.2% range, reset stagnation tracking
            if price_change > self.stagnation_range:
                print(f"  🔄 {symbol} exited stagnation range: {price_change*100:.2f}% from start price")
                del self.stagnation_start_price[symbol]
                del self.stagnation_start_time[symbol]
                return False
        
        # If not currently tracking stagnation, start tracking
        if symbol not in self.stagnation_start_price:
            self.stagnation_start_price[symbol] = current_price
            self.stagnation_start_time[symbol] = current_time
            return False
        
        # Check if we've been in the range for 15+ minutes
        time_in_range = current_time - self.stagnation_start_time[symbol]
        
        if time_in_range >= self.stagnation_threshold:
            start_price = self.stagnation_start_price[symbol]
            price_change = abs(current_price - start_price) / start_price
            print(f"  📊 {symbol} stagnant: {time_in_range} min in ±{self.stagnation_range*100:.1f}% range, current change: {price_change*100:.2f}%")
            return True
        
        return False

    def update_price_history(self, symbol, price, time):
        """Update price history for stagnation tracking"""
        self.price_history[symbol].append((time, price))
