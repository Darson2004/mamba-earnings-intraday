import math
import numpy as np
import torch
from collections import defaultdict
from typing import NamedTuple, List, Dict, Tuple, Optional


class Trade(NamedTuple):
    """Unified trade record.
    cash_flow > 0 for BUY (cash spent), > 0 for SELL (cash received).
    """
    symbol: str
    side: str               # "BUY" | "SELL"
    exec_px: float          # executed price (ask for BUY, bid for SELL)
    mid_px: float           # mid price observed at execution
    shares: float
    cash_flow: float        # dollars spent or received for the leg
    t: int                  # minute since 9:30
    reason: str             # "ENTRY" | "SCALE" | "STOP" | "TIME" | "EOD"
    mu: float               # mu20 used at decision time (if applicable)
    spread_paid: float      # dollars paid vs mid (ask-mid for buy, mid-bid for sell) * shares


class TradingStrategy:
    """
    Selective, unified intraday strategy.

    Design rules (numbering kept from the development checklist):
    1) Correct SELL semantics (operate in SHARES, not dollars) and full exits.
    3) Unified clock: PREDICT_START, TRADE_START, TRADE_END, LAST_ENTRY.
    4) top_k actually enforced on entries (never restrict exits).
    5) Hold horizon & stop-loss enforced using bid/ask microstructure.
    6) Spread analytics recorded as mid vs exec; no double counting.
    7) Predict during calibration; only trade inside trading window.
    8) Backtester/CLI will control model path & args; strategy stays parameterized.
    9) (Seeding handled in backtester.)
    11) EOD liquidation is handled *inside* the strategy when liquidate_eod=True.
    12) Unified trade log schema via Trade NamedTuple.
    13) Edge > transaction-cost hurdle enforced at decision time.
    14) Feature contract: use dataset close column (col=3) for prices only; no magic feature rewrites.
    15) Interaction of top_k and caps: per-minute gross add cap enforced.
    16) Tight, data-driven clamp: volatility-scaled clamp of mu20.
    17) Do NOT chain predictions: fresh mu20 from latest observed window each minute (renormalize each time).

    Notes
    -----
    • We intentionally *do not* truncate the internal per-stock sequence (growing window by design),
      but inference always uses the last `seq_len` slice and re-normalizes per minute.
    • Dataset API assumed:
        - dataset.data[idx] -> Tensor [T, F]
        - dataset.day_length -> int
        - dataset.name_date -> List[str]
        - close price at column 3
    """

    def __init__(
        self,
        model,
        device,
        dataset,
        considered_stocks: List[int],
        portfolio_value: float,
        *,
        seq_len: int = 50,
        pred_len: int = 20,
        max_position_frac: float = 0.02,      # per-name cap vs initial portfolio
        # microstructure / costs (bps per side)
        spread_bps: float = 5.0,              # half-spread per side
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
        buffer_bps: float = 0.0,
        # unified clock (minutes since 9:30)
        predict_start_minute: int = 0,
        start_trading_minute: int = 90,
        trade_end_minute: int = 240,
        # risk rules
        stop_loss: float = -0.005,            # -0.50%
        buy_threshold: float = 0.001,         # +0.10% (raw before hurdle)
        top_k: Optional[int] = None,          # max new entries per minute
        gross_add_cap: float = 0.02,          # ≤ this fraction of portfolio added per minute
        liquidate_eod: bool = True,
        clamp_k: float = 2.0,                 # k * sigma_20m clamp
        vol_lookback: int = 60,               # minutes for sigma estimation
    ):
        self.model = model
        self.device = device
        self.dataset = dataset
        self.considered_stocks = considered_stocks
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.max_position_frac = max_position_frac
        self.portfolio_value = float(portfolio_value)
        self.cash = float(portfolio_value)
        self.positions: Dict[str, Dict[str, float]] = {}
        self.entry_times: Dict[str, int] = {}
        self.trade_log: List[Trade] = []
        self.all_predictions: List[Dict[str, float]] = []

        # Prices & costs
        self.spread_fee = spread_bps / 10000.0
        self.commission_fee = commission_bps / 10000.0
        self.slippage_fee = slippage_bps / 10000.0
        self.extra_buffer = buffer_bps / 10000.0
        # Round-trip hurdle in decimal (BUY+SELL)
        self.hurdle = 2 * (self.spread_fee + self.commission_fee + self.slippage_fee) + self.extra_buffer

        # Unified window
        self.PREDICT_START = predict_start_minute
        self.TRADE_START = start_trading_minute
        self.TRADE_END = trade_end_minute
        self.LAST_ENTRY = self.TRADE_END - self.pred_len
        assert 0 <= self.PREDICT_START <= self.TRADE_START < self.TRADE_END

        # Risk & decisions
        self.stop_loss = stop_loss
        self.buy_threshold = buy_threshold
        self.top_k = top_k
        self.gross_add_cap = gross_add_cap
        self.liquidate_eod = liquidate_eod
        self.clamp_k = clamp_k
        self.vol_lookback = vol_lookback

        # Convenience maps
        self.stock_indices = self.considered_stocks
        self.names = [self.dataset.name_date[idx] if hasattr(self.dataset, 'name_date') else str(idx)
                      for idx in self.stock_indices]
        self.idx_by_name = {self.names[i]: self.stock_indices[i] for i in range(len(self.names))}
        self.day_len = int(self.dataset.day_length)

    # ---------------- Model inference helpers ----------------
    def _predict_mu20(self, seq_tensor: torch.Tensor) -> float:
        """Predict mu20 using *last seq_len rows*, per-feature z-score across time.
        Fresh infer each minute; do **not** chain prior predictions.
        """
        if seq_tensor.shape[0] < self.seq_len:
            return 0.0
        X = seq_tensor[-self.seq_len:].unsqueeze(0).to(self.device)  # [1, L, F]
        mean = X.mean(dim=1, keepdim=True)
        std = X.std(dim=1, keepdim=True)
        std = torch.where(std > 1e-8, std, torch.ones_like(std))
        Xn = (X - mean) / std
        with torch.no_grad():
            out = self.model(Xn)
            mu = out[0] if isinstance(out, tuple) else out
            mu = float(mu.squeeze().item())
        return mu

    def _vol_scaled_clamp(self, mu20_raw: float, close_series: List[float]) -> float:
        """Clamp mu20 by k * sigma_20m, where sigma_20m is estimated from recent 1m returns.
        Uses robust MAD-based std for stability.
        """
        if len(close_series) < max(2, self.vol_lookback):
            return mu20_raw  # not enough data to clamp intelligently
        # 1-min returns over lookback window
        arr = np.asarray(close_series[-self.vol_lookback:], dtype=np.float64)
        rets = np.diff(arr) / arr[:-1]
        if not np.isfinite(rets).all() or len(rets) < 3:
            return mu20_raw
        med = np.median(rets)
        mad = np.median(np.abs(rets - med))
        sigma_1m = 1.4826 * mad if mad > 0 else np.std(rets)
        if not np.isfinite(sigma_1m) or sigma_1m <= 0:
            return mu20_raw
        sigma_20m = sigma_1m * math.sqrt(self.pred_len)
        bound = self.clamp_k * sigma_20m
        return float(np.clip(mu20_raw, -bound, +bound))

    # ---------------- Risk & sizing ----------------
    def _per_name_cap_room(self, symbol: str, dollars: float) -> float:
        per_name_cap = self.portfolio_value * self.max_position_frac
        current_cap = self.positions.get(symbol, {}).get('capital', 0.0)
        allowed = max(0.0, per_name_cap - current_cap)
        return min(dollars, allowed, self.cash)

    # ---------------- Execution primitives ----------------
    def buy_dollars(self, symbol: str, mid: float, dollars: float, t: int, reason: str, mu: float) -> float:
        if mid <= 0 or dollars <= 0:
            return 0.0
        allowed = self._per_name_cap_room(symbol, dollars)
        if allowed <= 0:
            return 0.0
        ask = mid * (1 + self.spread_fee)
        shares = allowed / ask
        if shares <= 0:
            return 0.0
        exec_px = ask
        spread_paid = (ask - mid) * shares
        self.cash -= allowed
        if symbol in self.positions:
            pos = self.positions[symbol]
            new_sh = pos['shares'] + shares
            avg_px = (pos['avg_exec_price'] * pos['shares'] + exec_px * shares) / new_sh
            pos.update({'shares': new_sh, 'avg_exec_price': avg_px, 'capital': pos['capital'] + allowed})
        else:
            self.positions[symbol] = {
                'shares': shares,
                'avg_exec_price': exec_px,
                'capital': allowed,
                'entry_time': t,
            }
            self.entry_times[symbol] = t
        self.trade_log.append(Trade(symbol, 'BUY', exec_px, mid, shares, allowed, t, reason, mu, spread_paid))
        return allowed

    def sell_shares(self, symbol: str, mid: float, shares: float, t: int, reason: str, mu: float) -> float:
        if symbol not in self.positions or mid <= 0 or shares <= 0:
            return 0.0
        pos = self.positions[symbol]
        shares = min(shares, pos['shares'])
        bid = mid * (1 - self.spread_fee)
        cash_out = shares * bid
        exec_px = bid
        spread_paid = (mid - bid) * shares
        self.cash += cash_out
        pos['shares'] -= shares
        # reduce cost basis proportionally by avg exec price paid
        pos['capital'] -= shares * pos['avg_exec_price']
        if pos['shares'] <= 1e-8:
            del self.positions[symbol]
            self.entry_times.pop(symbol, None)
        self.trade_log.append(Trade(symbol, 'SELL', exec_px, mid, shares, cash_out, t, reason, mu, spread_paid))
        return cash_out

    # ---------------- Main loop ----------------
    def run_day(self) -> Tuple[List[Trade], Dict[str, Dict[str, float]]]:
        F = self.dataset.data[self.stock_indices[0]].shape[1]
        seqs: List[torch.Tensor] = [torch.empty((0, F), device=self.device) for _ in self.stock_indices]
        done = [False] * len(self.stock_indices)

        print(
            f"🔁 Predict≥t={max(self.seq_len, self.PREDICT_START)}, "
            f"Trade t∈[{self.TRADE_START},{self.TRADE_END}), LastEntry={self.LAST_ENTRY} (horizon={self.pred_len})"
        )

        t = 0
        while t < min(self.day_len, self.TRADE_END):
            # 1) append this minute's row (growing window; intentionally not truncated)
            for i, idx in enumerate(self.stock_indices):
                if done[i]:
                    continue
                if t < len(self.dataset.data[idx]):
                    row = self.dataset.data[idx][t:t+1, :].clone().to(self.device)
                    seqs[i] = torch.cat([seqs[i], row], dim=0)
                else:
                    done[i] = True

            # guard: need seq_len to start predicting
            if t < self.seq_len:
                t += 1
                continue

            # 2) predictions (fresh each minute, renormalized)
            minute_metrics: List[Tuple[int, str, float, float]] = []  # (i, name, mu, mid)
            if t >= self.PREDICT_START:
                for i, name in enumerate(self.names):
                    if done[i] or seqs[i].shape[0] < self.seq_len:
                        continue
                    mu_raw = self._predict_mu20(seqs[i])
                    # data-driven clamp using recent closes
                    idx = self.idx_by_name[name]
                    # collect close prices for clamp
                    closes = self.dataset.data[idx][:min(t+1, len(self.dataset.data[idx])), 3].detach().cpu().numpy().tolist()
                    mu = self._vol_scaled_clamp(mu_raw, closes)
                    mid = float(self.dataset.data[idx][t, 3].item())
                    minute_metrics.append((i, name, mu, mid))

            # 3) log predictions
            minute_metrics.sort(key=lambda x: x[2], reverse=True)
            in_trading = self.TRADE_START <= t < self.TRADE_END
            can_enter = self.TRADE_START <= t <= self.LAST_ENTRY

            for (i, name, mu, mid) in minute_metrics:
                action = "HOLD"
                if in_trading and can_enter and (mu >= max(self.buy_threshold, self.hurdle)) and (name not in self.positions):
                    action = "BUY"
                self.all_predictions.append({"stock": name, "time": t, "mu": mu, "action": action})

            if in_trading:
                # (A) exits: time-based and stop-loss — NEVER limited by top_k
                for (i, name, mu, mid) in minute_metrics:
                    if name not in self.positions:
                        continue
                    pos = self.positions[name]
                    # time-based exit
                    if t - pos['entry_time'] >= self.pred_len:
                        self.sell_shares(name, mid, pos['shares'], t, reason="TIME", mu=mu)
                        continue
                    # stop-loss exit (MTM vs avg_exec_price)
                    mtm_ret = (mid / pos['avg_exec_price']) - 1.0
                    if mtm_ret <= self.stop_loss:
                        self.sell_shares(name, mid, pos['shares'], t, reason="STOP", mu=mu)
                        continue

                # (B) entries: apply hurdle, top_k, and per-minute gross add cap
                if can_enter:
                    # candidates: above hurdle & not already held
                    cands = [(i, name, mu, mid) for (i, name, mu, mid) in minute_metrics
                             if (mu >= max(self.buy_threshold, self.hurdle)) and (name not in self.positions)]
                    if self.top_k is not None:
                        cands = cands[: int(self.top_k)]
                    dollars_added = 0.0
                    dollar_limit = self.portfolio_value * max(0.0, float(self.gross_add_cap))
                    for (i, name, mu, mid) in cands:
                        if dollars_added >= dollar_limit > 0:
                            break
                        # size: 1% notional scaled by signal strength (cap at 3% signal)
                        strength = min(1.0, max(0.0, mu) / 0.03)
                        target = self.portfolio_value * 0.01 * strength
                        # respect per-minute gross add cap
                        allowed_now = min(target, max(0.0, dollar_limit - dollars_added)) if dollar_limit > 0 else target
                        spent = self.buy_dollars(name, mid, allowed_now, t, reason="ENTRY" if name not in self.entry_times else "SCALE", mu=mu)
                        if spent > 0:
                            dollars_added += spent
                            print(f"t={t:03d} BUY {name:>10s} mu={mu:+.4f} ({mu*100:5.2f}%) size=${spent:,.0f} mid={mid:.4f}")

            t += 1

        # EOD liquidation inside strategy (if enabled)
        if self.liquidate_eod and self.positions:
            tt = min(self.day_len - 1, max(0, self.TRADE_END - 1))
            for name in list(self.positions.keys()):
                idx = self.idx_by_name.get(name)
                if idx is None or tt >= len(self.dataset.data[idx]):
                    continue
                mid = float(self.dataset.data[idx][tt, 3].item())
                pos = self.positions[name]
                self.sell_shares(name, mid, pos['shares'], tt, reason="EOD", mu=0.0)

        return self.trade_log, self.positions

    # ---------------- Reporting ----------------
    def print_prediction_stats(self):
        if not self.all_predictions:
            return
        mus = [p['mu'] for p in self.all_predictions]
        print("\n=== PREDICTION STATS ===")
        print(f"Total preds: {len(self.all_predictions)} | Range: [{min(mus):.4f}, {max(mus):.4f}] | Mean±Std: {np.mean(mus):.4f}±{np.std(mus):.4f}")
        actions = [p.get('action', 'HOLD') for p in self.all_predictions]
        print(f"Actions: BUY={sum(1 for a in actions if a=='BUY')} HOLD={sum(1 for a in actions if a=='HOLD')}")

    def categorize_trades_with_timesteps(self):
        if not self.trade_log:
            print("\n=== TRADE CATEGORIZATION ===\nNo trades executed.")
            return
        print("\n=== TRADE CATEGORIZATION WITH TIMESTEPS ===")
        by_symbol = defaultdict(list)
        for tr in self.trade_log:
            by_symbol[tr.symbol].append(tr)
        for sym, trs in by_symbol.items():
            buys = [x for x in trs if x.side == 'BUY']
            sells = [x for x in trs if x.side == 'SELL']
            print(f"\n{sym}:")
            if buys:
                print(f"  BUY ({len(buys)}):")
                for x in buys:
                    print(f"    t={x.t:03d} {x.shares:.3f} @ {x.exec_px:.4f} (mid {x.mid_px:.4f}) cash=${x.cash_flow:,.2f} spread=${x.spread_paid:.2f} reason={x.reason}")
            if sells:
                print(f"  SELL ({len(sells)}):")
                for x in sells:
                    print(f"    t={x.t:03d} {x.shares:.3f} @ {x.exec_px:.4f} (mid {x.mid_px:.4f}) cash=${x.cash_flow:,.2f} spread=${x.spread_paid:.2f} reason={x.reason}")
            if buys and sells:
                avg_buy = sum(x.exec_px * x.shares for x in buys) / max(1e-12, sum(x.shares for x in buys))
                avg_sell = sum(x.exec_px * x.shares for x in sells) / max(1e-12, sum(x.shares for x in sells))
                change = (avg_sell / avg_buy - 1) * 100
                print(f"  Avg Buy={avg_buy:.4f} | Avg Sell={avg_sell:.4f} | Δ={change:+.2f}%")
        print("\n📋 OVERALL SUMMARY:")
        print(f"  Total trades: {len(self.trade_log)} | Buys: {sum(1 for t in self.trade_log if t.side=='BUY')} | Sells: {sum(1 for t in self.trade_log if t.side=='SELL')}")

    def analyze_spread_impact(self):
        if not self.trade_log:
            print("\n=== SPREAD IMPACT ANALYSIS ===\nNo trades executed.")
            return
        buys = [t for t in self.trade_log if t.side == 'BUY']
        sells = [t for t in self.trade_log if t.side == 'SELL']
        total_buy_cash = sum(t.cash_flow for t in buys)
        total_sell_cash = sum(t.cash_flow for t in sells)
        realized_pnl = total_sell_cash - total_buy_cash
        spread_paid = sum(t.spread_paid for t in self.trade_log)
        mid_exec_pnl = realized_pnl + spread_paid
        print("\n=== SPREAD IMPACT ANALYSIS ===")
        print(f"  Realized PnL (exec): ${realized_pnl:,.2f}")
        print(f"  Total recorded spread cost: ${spread_paid:,.2f}")
        print(f"  Hypothetical PnL at mid:   ${mid_exec_pnl:,.2f}")
        if abs(mid_exec_pnl) > 1e-9:
            print(f"  Spread drag: {100 * spread_paid / abs(mid_exec_pnl):.2f}% of mid-exec PnL potential")
