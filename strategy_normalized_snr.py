import torch
import numpy as np
from collections import defaultdict

class TradingStrategySNR:
    """
    Trading strategy using SNR-based thresholding, position sizing, and ensemble averaging (MC dropout).
    - Only acts when signal-to-noise ratio (SNR) of prediction exceeds a threshold.
    - Position size scales with SNR (confidence).
    - Uses MC dropout for uncertainty estimation.
    """
    def __init__(self, model, device, dataset, considered_stocks, portfolio_value, seq_len=30, pred_len=2,
                 snr_threshold=1.0, max_position_frac=0.05, min_hold_minutes=1, n_mc_samples=20):
        self.model = model
        self.device = device
        self.dataset = dataset
        self.considered_stocks = considered_stocks
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.snr_threshold = snr_threshold
        self.max_position_frac = max_position_frac
        self.n_mc_samples = n_mc_samples
        self.portfolio_value = portfolio_value
        self.cash = portfolio_value
        self.positions = {}
        self.distinct_stocks_bought = set()
        self.trade_log = []
        self.min_hold_minutes = min_hold_minutes
        self.last_buy_time = {}

    def cap_check(self, symbol, amount):
        max_cap = self.portfolio_value * self.max_position_frac
        current_cap = self.positions.get(symbol, {}).get('capital', 0.0)
        return min(amount, max(0.0, max_cap - current_cap))

    def buy(self, symbol, price, amount, t=None):
        if price <= 0:
            print(f"WARNING: Invalid buy price for {symbol} at t={t}")
            return 0.0
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
        pos['shares'] -= shares_to_sell
        pos['capital'] -= shares_to_sell * pos['avg_price']
        if pos['shares'] <= 1e-8:
            del self.positions[symbol]
        self.trade_log.append((symbol, 'SELL', price, shares_to_sell, sell_cap, t))
        return sell_cap

    def predict_with_ci(self, input_seq, pred_len):
        """
        MC dropout prediction for pred_len steps, returns mean, std for each step.
        """
        self.model.train()
        preds = []
        stds = []
        seq = input_seq.clone()
        for t in range(pred_len):
            samples = []
            X = seq[-self.seq_len:].unsqueeze(0).to(self.device)
            with torch.no_grad():
                for _ in range(self.n_mc_samples):
                    pred = self.model(X).squeeze()
                    samples.append(pred)
            samples_tensor = torch.stack(samples)
            mu = samples_tensor.mean()
            sigma = samples_tensor.std(unbiased=False)
            preds.append(mu)
            stds.append(sigma)
            # For next step, append predicted value to sequence
            next_pred = seq[-1, :].clone()
            next_pred[10] = mu  # pct_chg index
            seq = torch.cat([seq, next_pred.unsqueeze(0)], dim=0)
        self.model.eval()
        return torch.stack(preds), torch.stack(stds)

    def run_day(self):
        n_stocks = len(self.considered_stocks)
        stock_indices = self.considered_stocks
        stock_names = [self.dataset.name_date[idx] if hasattr(self.dataset, 'name_date') else str(idx) for idx in stock_indices]
        day_length = self.dataset.day_length
        seqs = [self.dataset.data[idx][:self.seq_len, :].clone().to(self.device) for idx in stock_indices]
        t = self.seq_len
        done = [False for _ in stock_indices]
        while t < min(day_length, self.seq_len + 330):
            batch_X = []
            for i, seq in enumerate(seqs):
                if done[i]:
                    batch_X.append(torch.zeros_like(seq))
                else:
                    batch_X.append(seq)
            batch_X_norm = []
            for i, seq in enumerate(batch_X):
                mean = seq.mean(dim=0, keepdim=True)
                std = seq.std(dim=0, keepdim=True)
                std = torch.where(std > 1e-8, std, torch.ones_like(std))
                seq_norm = (seq - mean) / std
                batch_X_norm.append(seq_norm[-self.seq_len:])
            batch_X_norm = torch.stack(batch_X_norm).to(self.device)
            # Use MC dropout for each stock
            for i, name in enumerate(stock_names):
                if done[i]:
                    continue
                preds, stds = self.predict_with_ci(batch_X_norm[i], self.pred_len)
                mu = preds[0].item()
                sigma = stds[0].item()
                snr = mu / (sigma + 1e-8)
                print(f"Stock {name} at t={t}: mu={mu:.6f}, sigma={sigma:.6f}, SNR={snr:.2f}")
                if snr > self.snr_threshold and mu > 0:
                    # Position size scales with SNR, up to max_position_frac
                    base_amount = self.portfolio_value * 0.01
                    scale = min(snr / self.snr_threshold, self.max_position_frac / 0.01)
                    buy_amount = base_amount * scale
                    actual_bought = self.buy(name, batch_X_norm[i, -1, 3].item(), buy_amount, t=t)
                    print(f"  → BUY attempted: {buy_amount:.2f}, actual: {actual_bought:.2f}")
                else:
                    print(f"  → NO BUY: SNR <= threshold or mu <= 0")
            # Advance all sequences by pred_len
            for i in range(n_stocks):
                if done[i]:
                    continue
                advance = self.pred_len
                if t + advance < day_length:
                    seqs[i] = torch.cat([seqs[i], self.dataset.data[stock_indices[i]][t:t+advance, :].clone().to(self.device)], dim=0)
                else:
                    done[i] = True
            t += self.pred_len
        return self.trade_log, self.positions 