import torch
import numpy as np
from collections import defaultdict
import torch.nn as nn

class TradingStrategy:
	"""
	Extremely basic trading strategy with 20-minute scrolling window.
	
	Simple Rules:
	- Buy if 20-min prediction ≥ 0.10%
	- Hold if prediction < 0.10%
	- Sell if position loses > 0.50% OR after 20 minutes
	- Maintains 20-minute scrolling window updated every minute
	"""
	def __init__(self, model, device, dataset, considered_stocks, portfolio_value, seq_len=50, pred_len=20,
				 max_position_frac=0.05, spread_fee=0.0005, max_trading_minutes=330, start_trading_minute=0):
		self.model = model
		self.device = device
		self.dataset = dataset
		self.considered_stocks = considered_stocks
		self.seq_len = seq_len
		self.pred_len = pred_len
		self.max_position_frac = max_position_frac
		self.portfolio_value = portfolio_value
		self.cash = portfolio_value
		self.positions = {}
		self.distinct_stocks_bought = set()
		self.trade_log = []
		self.spread_fee = spread_fee  # 0.05% = 0.0005
		self.max_trading_minutes = max_trading_minutes
		self.start_trading_minute = start_trading_minute
		
		# Simple strategy parameters
		self.buy_threshold = 0.001  # 0.10% = 0.001
		self.stop_loss = -0.005     # -0.50% = -0.005
		self.hold_horizon = 20      # 20 minutes
		self.top_k = 10              # take top-K names per minute
		
		# Track entry times for 20-minute horizon
		self.entry_times = {}
		
		# Track all predictions for analysis
		self.all_predictions = []
	
	def _compute_simple_prediction(self, input_seq):
		"""Compute simple 20-minute prediction using the model."""
		# Normalize input sequence
		seq = input_seq.clone()
		X = seq[-self.seq_len:].unsqueeze(0).to(self.device)
		
		# Normalize window for model input
		mean = X.mean(dim=1, keepdim=True)
		std_dev = X.std(dim=1, keepdim=True)
		std_dev = torch.where(std_dev > 1e-8, std_dev, torch.ones_like(std_dev))
		X_norm = (X - mean) / std_dev
		
		with torch.no_grad():
			# Get prediction from model
			output = self.model(X_norm)
			if isinstance(output, tuple):
				# New model: (mean, logvar) - use mean
				pred = output[0].squeeze()
			else:
				# Old model: single output
				pred = output.squeeze()
			
			# Clamp to realistic bounds
			pred = torch.clamp(pred, -0.10, 0.10)
			
			return pred.item()
	
	def _compute_path_confidence(self, input_seq):
		"""Compute 20-minute prediction for the simple strategy."""
		# For the simple strategy, we just need the 20-minute prediction
		pred_20min = self._compute_simple_prediction(input_seq)
		
		# Return simplified metrics
		mu20 = pred_20min
		sigma20 = 0.01  # Fixed small uncertainty for compatibility
		p_netpos = 1.0 if mu20 >= self.buy_threshold else 0.0  # Binary decision
		H = 0.0  # No entropy calculation needed
		
		# Create dummy step predictions for compatibility
		step_means = torch.full((self.pred_len,), mu20 / self.pred_len, device=self.device)
		step_stds = torch.full((self.pred_len,), sigma20 / np.sqrt(self.pred_len), device=self.device)
		path_returns = torch.full((1,), mu20, device=self.device)
		
		return mu20, sigma20, p_netpos, H, step_means, step_stds, path_returns
	
	def cap_check(self, symbol, amount):
		"""Check position size limits."""
		max_cap = self.portfolio_value * self.max_position_frac
		current_cap = self.positions.get(symbol, {}).get('capital', 0.0)
		return min(amount, max(0.0, max_cap - current_cap))

	def buy(self, symbol, price, amount, t=None):
		"""Buy shares with spread handling."""
		if price <= 0:
			print(f"WARNING: Invalid buy price for {symbol} at t={t}")
			return 0.0
		
		# Calculate spread-adjusted price (buy at ask price = mid + spread)
		ask_price = price * (1 + self.spread_fee)
		capital = min(self.cap_check(symbol, amount), self.cash)
		
		if capital <= 0:
			return 0.0
		
		# Calculate shares based on ask price (including spread)
		shares = capital / ask_price
		if shares <= 0:
			return 0.0
		
		# Deduct capital from cash
		self.cash -= capital
		self.distinct_stocks_bought.add(symbol)
		
		if symbol in self.positions:
			pos = self.positions[symbol]
			total_shares = pos['shares'] + shares
			# Calculate weighted average price including spread
			avg_price = (pos['avg_price'] * pos['shares'] + ask_price * shares) / total_shares
			self.positions[symbol].update({
				'shares': total_shares, 
				'avg_price': avg_price, 
				'capital': pos['capital'] + capital
			})
		else:
			self.positions[symbol] = {
				'shares': shares, 
				'avg_price': ask_price, 
				'capital': capital, 
				'entry_price': ask_price
			}
		
		# Record entry time for 20-minute horizon
		self.entry_times[symbol] = t
		
		# Include in trade log
		self.trade_log.append((symbol, 'BUY', ask_price, shares, capital, t, None, self.spread_fee))
		
		return capital

	def sell(self, symbol, price, amount, t=None):
		"""Sell shares with spread handling."""
		if price <= 0:
			print(f"WARNING: Invalid sell price for {symbol} at t={t}")
			return 0.0
		
		if symbol not in self.positions:
			return 0.0
		
		pos = self.positions[symbol]
		max_shares_to_sell = pos['shares']
		shares_to_sell = min(max_shares_to_sell, amount / price)
		
		if shares_to_sell <= 0:
			return 0.0
		
		# Calculate spread-adjusted price (sell at bid price = mid - spread)
		bid_price = price * (1 - self.spread_fee)
		sell_cap = shares_to_sell * bid_price
		
		# Add to cash (after spread)
		self.cash += sell_cap
		pos['shares'] -= shares_to_sell
		pos['capital'] -= shares_to_sell * pos['avg_price']
		
		if pos['shares'] <= 1e-8:
			del self.positions[symbol]
			if symbol in self.entry_times:
				del self.entry_times[symbol]
		
		# Include in trade log
		self.trade_log.append((symbol, 'SELL', bid_price, shares_to_sell, sell_cap, t, None, self.spread_fee))
		
		return sell_cap

	def run_day(self):
		"""Main trading loop with 20-minute scrolling window."""
		n_stocks = len(self.considered_stocks)
		stock_indices = self.considered_stocks
		stock_names = [self.dataset.name_date[idx] if hasattr(self.dataset, 'name_date') else str(idx) for idx in stock_indices]
		name_to_index = {stock_names[i]: stock_indices[i] for i in range(len(stock_names))}
		day_length = self.dataset.day_length
		
		# Start trading at the configured minute-from-open, but ensure we have at least seq_len history
		t = max(self.seq_len, self.start_trading_minute)
		seqs = [self.dataset.data[idx][:t, :].clone().to(self.device) for idx in stock_indices]
		done = [False for _ in stock_indices]
		
		# Print trading time boundaries (minutes from open)
		start_min = t
		end_min = min(day_length, self.start_trading_minute + self.max_trading_minutes)
		print(f"Trading window (minutes from open): {start_min} to {end_min}")
		
		while t < min(day_length, self.start_trading_minute + self.max_trading_minutes):
			# Minute-by-minute loop with 20-minute scrolling window
			batch_X = []
			for i, seq in enumerate(seqs):
				batch_X.append(torch.zeros_like(seq) if done[i] else seq)
			
			# Compute predictions for each stock
			minute_metrics = []
			for i, name in enumerate(stock_names):
				if done[i]:
					continue
				
				# Get 20-minute prediction
				mu20, sigma20, p_netpos, H, preds, stds, path_sums = self._compute_path_confidence(seqs[i])
				minute_metrics.append((i, name, mu20, sigma20, p_netpos, H, preds, stds))
			
			# Take top-K names by prediction value
			minute_metrics.sort(key=lambda x: x[2], reverse=True)  # Sort by mu20
			minute_metrics = minute_metrics[:self.top_k]
			
			for (i, name, mu20, sigma20, p_netpos, H, preds, stds) in minute_metrics:
				current_price = self.dataset.data[stock_indices[i]][t, 3].item()
				
				# Record prediction for analysis
				self.all_predictions.append({
					'stock': name,
					'time': t,
					'mu': mu20,
					'sigma': sigma20,
					'p_netpos': p_netpos,
					'action': 'BUY' if mu20 >= self.buy_threshold else 'HOLD'
				})
				
				# Check exits for existing positions
				if name in self.positions:
					pos = self.positions[name]
					entry_price = pos['entry_price']
					entry_time = self.entry_times.get(name, t)
					
					# Calculate current P&L (accounting for spread)
					# When we sell, we get the bid price (mid - spread)
					bid_price = current_price * (1 - self.spread_fee)
					current_pnl = (bid_price - entry_price) / entry_price
					
					# Exit conditions:
					# 1. Stop loss: lose more than 0.50%
					# 2. Time horizon: 20 minutes have passed
					time_held = t - entry_time
					
					if current_pnl <= self.stop_loss or time_held >= self.hold_horizon:
						reason = "stop_loss" if current_pnl <= self.stop_loss else "time_horizon"
						print(f"  🔴 SELL {name} at t={t}: {reason}, PnL={current_pnl:.4f}, time_held={time_held}")
						self.sell(name, current_price, pos['shares'], t)
						continue
				
				# Entry logic: buy if prediction ≥ 0.10%
				if t <= 190 and name not in self.positions and mu20 >= self.buy_threshold:
					# Simple position sizing: 1% of portfolio
					buy_amount = self.portfolio_value * 0.01
					print(f"  🟢 BUY {name} at t={t}: mu20={mu20:.4f} ({mu20*100:.2f}%), size=${buy_amount:.0f}")
					self.buy(name, current_price, buy_amount, t=t)
			
			# Advance one minute for all sequences (maintain 20-minute scrolling window)
			for i in range(n_stocks):
				if done[i]:
					continue
				if t + 1 < day_length:
					seqs[i] = torch.cat([seqs[i], self.dataset.data[stock_indices[i]][t:t+1, :].clone().to(self.device)], dim=0)
				else:
					done[i] = True
			t += 1
		
		return self.trade_log, self.positions

	def print_prediction_stats(self):
		"""Print statistics about all predictions made during the day."""
		if not self.all_predictions:
			return
		
		mus = [p['mu'] for p in self.all_predictions]
		p_netpos_list = [p['p_netpos'] for p in self.all_predictions]
		
		print("\n=== PREDICTION STATISTICS ===")
		print(f"Total predictions: {len(self.all_predictions)}")
		print(f"Mu range: [{min(mus):.4f}, {max(mus):.4f}]")
		print(f"Mu mean/std: {np.mean(mus):.4f} ± {np.std(mus):.4f}")
		print(f"P_netpos range: [{min(p_netpos_list):.3f}, {max(p_netpos_list):.3f}]")
		print(f"P_netpos mean/std: {np.mean(p_netpos_list):.3f} ± {np.std(p_netpos_list):.3f}")
		
		# Analyze predictions above buy threshold
		above_threshold = [p for p in self.all_predictions if p['mu'] >= self.buy_threshold]
		below_threshold = [p for p in self.all_predictions if p['mu'] < self.buy_threshold]
		
		print(f"\nPredictions above buy threshold (≥{self.buy_threshold*100:.2f}%): {len(above_threshold)}")
		print(f"Predictions below buy threshold (<{self.buy_threshold*100:.2f}%): {len(below_threshold)}")
		
		if above_threshold:
			print("\nPredictions above threshold:")
			for p in above_threshold[:5]:  # Show first 5
				print(f"  {p['stock']} at t={p['time']}: mu={p['mu']:.4f} ({p['mu']*100:.2f}%)")

	def categorize_trades_with_timesteps(self):
		"""Categorize and summarize all trades with timestep information."""
		if not self.trade_log:
			print("\n=== TRADE CATEGORIZATION ===")
			print("No trades executed during this session.")
			return
		
		print("\n=== TRADE CATEGORIZATION WITH TIMESTEPS ===")
		
		# Group trades by symbol
		trades_by_symbol = defaultdict(list)
		for trade in self.trade_log:
			if len(trade) >= 8:  # New format with spread
				symbol, action, price, shares, capital, timestep, snr, spread = trade
			else:  # Old format without spread
				symbol, action, price, shares, capital, timestep, snr = trade
				spread = 0.0
			
			trades_by_symbol[symbol].append({
				'action': action,
				'price': price,
				'shares': shares,
				'capital': capital,
				'timestep': timestep,
				'spread': spread
			})
		
		# Print summary for each symbol
		for symbol, trades in trades_by_symbol.items():
			print(f"\n📈 {symbol}:")
			
			# Find buy and sell trades
			buy_trades = [t for t in trades if t['action'] == 'BUY']
			sell_trades = [t for t in trades if t['action'] == 'SELL']
			
			if buy_trades:
				print(f"  🟢 BUY Trades ({len(buy_trades)}):")
				for trade in buy_trades:
					spread_cost = trade['capital'] * trade['spread']
					print(f"    Timestep t={trade['timestep']}: {trade['shares']:.2f} shares @ ${trade['price']:.4f} (${trade['capital']:.2f})")
					print(f"      Spread cost: ${spread_cost:.2f} ({trade['spread']*100:.3f}%)")
			
			if sell_trades:
				print(f"  🔴 SELL Trades ({len(sell_trades)}):")
				for trade in sell_trades:
					spread_cost = trade['capital'] * trade['spread']
					print(f"    Timestep t={trade['timestep']}: {trade['shares']:.2f} shares @ ${trade['price']:.4f} (${trade['capital']:.2f})")
					print(f"      Spread cost: ${spread_cost:.2f} ({trade['spread']*100:.3f}%)")
			
			# Calculate summary if we have both buy and sell
			if buy_trades and sell_trades:
				total_buy_volume = sum(t['capital'] for t in buy_trades)
				total_sell_volume = sum(t['capital'] for t in sell_trades)
				avg_buy_price = sum(t['price'] * t['shares'] for t in buy_trades) / sum(t['shares'] for t in buy_trades)
				avg_sell_price = sum(t['price'] * t['shares'] for t in sell_trades) / sum(t['shares'] for t in sell_trades)
				price_change = ((avg_sell_price - avg_buy_price) / avg_buy_price) * 100
				
				print(f"  📊 Summary: Buy ${total_buy_volume:.2f} | Sell ${total_sell_volume:.2f}")
				print(f"  💰 Avg Buy: ${avg_buy_price:.4f} | Avg Sell: ${avg_sell_price:.4f} | Change: {price_change:+.2f}%")
		
		# Overall statistics
		total_trades = len(self.trade_log)
		total_buys = len([t for t in self.trade_log if t[1] == 'BUY'])
		total_sells = len([t for t in self.trade_log if t[1] == 'SELL'])
		
		print(f"\n📋 OVERALL TRADE SUMMARY:")
		print(f"  Total trades: {total_trades}")
		print(f"  Buy orders: {total_buys}")
		print(f"  Sell orders: {total_sells}")
		print(f"  Distinct stocks traded: {len(trades_by_symbol)}")
		
		# Timestep range
		if self.trade_log:
			timesteps = [t[5] for t in self.trade_log if t[5] is not None]
			if timesteps:
				print(f"  Trading period (minutes from open): {min(timesteps)} to {max(timesteps)}")

	def analyze_spread_impact(self):
		"""Analyze the impact of spread costs on trading performance."""
		if not self.trade_log:
			print("\n=== SPREAD IMPACT ANALYSIS ===")
			print("No trades executed during this session.")
			return
		
		print("\n=== SPREAD IMPACT ANALYSIS ===")
		
		# Separate buy and sell trades
		buy_trades = [t for t in self.trade_log if t[1] == 'BUY']
		sell_trades = [t for t in self.trade_log if t[1] == 'SELL']
		
		if not buy_trades or not sell_trades:
			print("Need both buy and sell trades to analyze spread impact.")
			return
		
		# Calculate total costs and volumes
		total_buy_volume = sum(t[4] for t in buy_trades)  # capital
		total_sell_volume = sum(t[4] for t in sell_trades)  # capital
		
		# Calculate spread costs
		total_buy_spread = 0
		total_sell_spread = 0
		for trade in buy_trades:
			if len(trade) >= 8:
				capital, spread = trade[4], trade[7]
				total_buy_spread += capital * spread
		for trade in sell_trades:
			if len(trade) >= 8:
				capital, spread = trade[4], trade[7]
				total_sell_spread += capital * spread
		
		total_spread_cost = total_buy_spread + total_sell_spread
		
		# Calculate gross and net returns
		gross_return = total_sell_volume - total_buy_volume
		net_return = gross_return - total_spread_cost
		
		print(f"💰 TRADING VOLUMES:")
		print(f"  Total Buy Volume: ${total_buy_volume:.2f}")
		print(f"  Total Sell Volume: ${total_sell_volume:.2f}")
		print(f"  Gross Return: ${gross_return:.2f}")
		
		print(f"\n💸 SPREAD COSTS:")
		print(f"  Buy Spread Cost: ${total_buy_spread:.2f} ({(total_buy_spread/total_buy_volume)*100:.3f}% of buy volume)")
		print(f"  Sell Spread Cost: ${total_sell_spread:.2f} ({(total_sell_spread/total_sell_volume)*100:.3f}% of sell volume)")
		print(f"  Total Spread Cost: ${total_spread_cost:.2f}")
		
		print(f"\n📊 NET PERFORMANCE:")
		print(f"  Gross Return: ${gross_return:.2f}")
		print(f"  Net Return (after spread): ${net_return:.2f}")
		
		if gross_return != 0:
			spread_impact_pct = (total_spread_cost / abs(gross_return)) * 100
			print(f"  Spread Impact: {spread_impact_pct:.2f}% of gross return")