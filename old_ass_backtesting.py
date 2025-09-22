
import argparse
import h5py
import numpy as np
import torch
from dataset import ClosePrice
from strategy import TradingStrategy
import matplotlib.pyplot as plt
from mambastock_model import MambaStock

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model
model = MambaStock(input_size=13, seq_len=50, pred_len=20)
# Load weights
model.load_state_dict(torch.load("mambastock_scrolling_eight.pth", map_location=device), strict=False)
# Move to device
model = model.to(device)
# Set to eval mode (unless using MC dropout for uncertainty)
model.eval()

# Enable fast GPU inference features
if device.type == 'cuda':
	try:
		import torch.backends.cudnn as cudnn
		cudnn.benchmark = True
	except Exception:
		pass
	# Optional: torch.compile can speed inference on PyTorch 2.x
	try:
		model = torch.compile(model)  # no-op on PyTorch < 2
	except Exception:
		pass

def main():
	parser = argparse.ArgumentParser(description="Backtest trading strategy with optional model.")
	parser.add_argument('--model', type=str, default=None, help='Path to model weights (e.g., mambastock_scrolling_eight.pth)')
	args = parser.parse_args()

	# --- CONFIG ---
	h5_path = "h5_test_training_two.h5"
	initial_portfolio = 100000
	seq_len = 50
	pred_len = 20
	
	# Trading window: 11:00 AM to 1:00 PM (minutes 90-210 from 9:30 AM open)
	# Focused 2-hour window during market hours

	# Load dataset
	dataset = ClosePrice(h5_path)
	print(f"Loaded dataset: {dataset.n_stocks} stocks, {dataset.day_length} minutes per day")
	
	considered_stocks = list(range(dataset.n_stocks))
	strategy = TradingStrategy(
		model=model,
		device=device,
		dataset=dataset,
		considered_stocks=considered_stocks,
		portfolio_value=initial_portfolio,
		seq_len=seq_len,
		pred_len=pred_len,
		start_trading_minute=90,       # 11:00 AM
		max_trading_minutes=120        # 2 hours → end at 210
	)
	
	# Run the strategy to get the trade log
	trade_log, final_positions = strategy.run_day()
	
	# Print prediction statistics
	strategy.print_prediction_stats()
	
	# Print trade categorization with timesteps
	strategy.categorize_trades_with_timesteps()
	
	# FIFO matching for realized P&L
	buy_queues = {name: [] for name in dataset.name_date}
	realized_pnl = 0.0
	trade_details = []
	
	for entry in trade_log:
		# Support entries with or without SNR/spread
		if len(entry) >= 7:
			symbol, action, price, shares, capital, trade_time, snr = entry[:7]
		else:  # Old format without SNR
			symbol, action, price, shares, capital, trade_time = entry
			snr = None
			
		if action == 'BUY':
			buy_queues[symbol].append({'shares': shares, 'price': price, 'capital': capital, 'time': trade_time, 'snr': snr})
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
					'return_pct': 100 * (sell_price - buy_price) / buy_price,
					'snr': buy.get('snr', None)
				})
				buy['shares'] -= matched_shares
				shares_to_sell -= matched_shares
				if buy['shares'] <= 1e-8:
					buy_queues[symbol].pop(0)
	
	# Liquidate remaining positions at the best predicted price between 11:00 and 1:00 (minutes 90 to 210)
	eod_seq_len = 90  # Use all available history up to 11:00am
	# Build batch of symbols to liquidate
	symbols_to_liq = []
	idxs = []
	queues = []
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
		symbols_to_liq.append(symbol)
		idxs.append(idx)
		queues.append(queue)
	
	if symbols_to_liq:
		# Prepare batch sequences [B, eod_seq_len, features]
		seq_list = []
		for idx in idxs:
			seq = dataset.data[idx][:eod_seq_len, :].clone().to(device)
			if seq.shape[0] < eod_seq_len:
				pad = torch.zeros((eod_seq_len - seq.shape[0], seq.shape[1]), device=seq.device)
				seq = torch.cat([pad, seq], dim=0)
			seq_list.append(seq)
		seq_for_pred = torch.stack(seq_list, dim=0)  # [B, eod_seq_len, features]
		B = seq_for_pred.shape[0]
		preds_mat = torch.empty((B, 120), device=device)
		# Batched GPU inference with inference_mode + autocast
		use_amp = (device.type == 'cuda')
		import contextlib
		amp_ctx = torch.cuda.amp.autocast(enabled=use_amp)
		with torch.inference_mode():
			with amp_ctx:
				for step in range(120):
					mean = seq_for_pred.mean(dim=1, keepdim=True)
					std = seq_for_pred.std(dim=1, keepdim=True)
					std = torch.where(std > 1e-8, std, torch.ones_like(std))
					seq_norm = (seq_for_pred - mean) / std
					input_batch = seq_norm[:, -eod_seq_len:, :]
					pred_norm = model(input_batch).squeeze().view(-1)
					# Model output is treated as actual pct_chg; enforce realistic clamp for stability
					pct_chg_pred = torch.clamp(pred_norm, -0.02, 0.02)
					preds_mat[:, step] = pct_chg_pred
					# roll forward
					next_row = seq_for_pred[:, -1, :].clone()
					next_row[:, 10] = pct_chg_pred
					seq_for_pred = torch.cat([seq_for_pred, next_row.unsqueeze(1)], dim=1)
		# Argmax on GPU
		max_idx = preds_mat.argmax(dim=1)
		sell_minutes = (91 + max_idx).tolist()
		for b, symbol in enumerate(symbols_to_liq):
			idx = idxs[b]
			sell_minute = sell_minutes[b]
			sell_price_mid = dataset.data[idx][sell_minute, 3].item()
			sell_price = sell_price_mid * (1 - getattr(strategy, 'spread_fee', 0.0))
			for buy in queues[b]:
				matched_shares = buy['shares']
				buy_price = buy['price']
				gain = (sell_price - buy_price) * matched_shares
				realized_pnl += gain
				trade_details.append({
					'symbol': symbol,
					'buy_time': buy['time'],
					'buy_price': buy_price,
					'sell_time': f'EOD_ACTUAL_{sell_minute}',
					'sell_price': sell_price,
					'shares': matched_shares,
					'gain': gain,
					'return_pct': 100 * (sell_price - buy_price) / buy_price if buy_price != 0 else 0.0,
					'snr': buy.get('snr', None)
				})
				trade_log.append((symbol, 'SELL', sell_price, matched_shares, matched_shares * sell_price, sell_minute, None, getattr(strategy, 'spread_fee', 0.0)))
	
	# Final portfolio value
	final_portfolio = initial_portfolio + realized_pnl
	print(f"\nInitial portfolio: ${initial_portfolio:,.2f}")
	print(f"Final portfolio:   ${final_portfolio:,.2f}")
	print(f"Total return:      ${realized_pnl:,.2f} ({100*realized_pnl/initial_portfolio:.2f}%)")

	# Track portfolio value at every minute (restricted to 11:00–1:00 → minutes 90–210)
	portfolio_values = []
	cash = float(initial_portfolio)
	positions = {name: [] for name in dataset.name_date}  # FIFO queues for each stock
	window_start = 90
	window_end = 210
	
	# Build time-indexed trades and ensure BUYs are processed before SELLs within each minute
	time_to_trades = {t: [] for t in range(window_start, window_end + 1)}
	
	for entry in trade_log:
		# Support entries with or without SNR/spread
		if len(entry) >= 7:
			symbol, action, price, shares, capital, trade_time, snr = entry[:7]
		else:
			symbol, action, price, shares, capital, trade_time = entry
		
		if trade_time is not None and window_start <= trade_time <= window_end:
			time_to_trades[int(trade_time)].append(entry)
	
	# Ensure per-minute ordering: BUY before SELL
	for minute in time_to_trades:
		if time_to_trades[minute]:
			time_to_trades[minute].sort(key=lambda e: 0 if (e[1] == 'BUY') else 1)

	for t in range(window_start, window_end + 1):
		# Apply trades at this minute
		for entry in time_to_trades[t]:
			# Support entries with or without SNR/spread
			if len(entry) >= 7:
				symbol, action, price, shares, capital, trade_time, snr = entry[:7]
			else:
				symbol, action, price, shares, capital, trade_time = entry
			# Use notional = price * shares for consistency; warn if capital differs
			notional = float(price) * float(shares)
			if abs((capital if capital is not None else 0.0) - notional) > max(1e-6, 0.001 * notional):
				print(f"[WARN] Capital mismatch at t={t} {symbol} {action}: logged capital={capital:.4f} vs price*shares={notional:.4f}")
			if action == 'BUY':
				cash -= notional
				positions.setdefault(symbol, [])
				positions[symbol].append({'shares': shares, 'price': price})
			elif action == 'SELL':
				shares_to_sell = shares
				sold_notional = 0.0
				if symbol not in positions or not positions[symbol]:
					print(f"[WARN] SELL at t={t} for {symbol} but no open lots to match. Shares={shares_to_sell:.4f}")
				while shares_to_sell > 0 and positions.get(symbol, []):
					lot = positions[symbol][0]
					matched_shares = min(lot['shares'], shares_to_sell)
					lot['shares'] -= matched_shares
					shares_to_sell -= matched_shares
					sold_notional += matched_shares * float(price)
					if lot['shares'] <= 1e-8:
						positions[symbol].pop(0)
				if shares_to_sell > 1e-8:
					print(f"[WARN] Unmatched SELL shares at t={t} for {symbol}: remaining {shares_to_sell:.4f} not matched to inventory")
				# Only add cash for matched portion
				cash += sold_notional
		# Calculate current portfolio value at this minute
		total_position_value = 0.0
		for symbol, lots in positions.items():
			if not lots:
				continue
			if symbol not in dataset.name_date:
				print(f"[WARN] Symbol {symbol} not found in dataset.name_date; skipping valuation at t={t}")
				continue
			idx = dataset.name_date.index(symbol)
			current_price = dataset.data[idx][t, 3].item()
			# Value positions at bid (mid minus spread) to be consistent with realized execution
			bid_price_t = current_price * (1 - getattr(strategy, 'spread_fee', 0.0))
			for lot in lots:
				total_position_value += lot['shares'] * bid_price_t
		portfolio_values.append(float(cash + total_position_value))

	# Reconcile plotted end value vs realized final (cash PnL)
	plot_end_value = portfolio_values[-1] if portfolio_values else float('nan')
	expected_final = initial_portfolio + realized_pnl
	print(f"\n[RECONCILE] Plot end value: ${plot_end_value:,.2f} | Realized final: ${expected_final:,.2f} | Diff: ${plot_end_value-expected_final:,.2f}")

	# Plot portfolio value over the 11:00–1:00 window
	plt.figure(figsize=(12, 6))
	plt.plot(range(window_start, window_end + 1), portfolio_values, label='Portfolio Value')
	plt.xlabel('Minute of Day (from open)')
	plt.ylabel('Portfolio Value ($)')
	plt.title('Portfolio Value (11:00 AM to 1:00 PM)')
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.show()
	
	# Categorize trades by SNR levels
	print("\n=== TRADE CATEGORIZATION BY SNR ===")
	
	snr_categories = {
		'SNR > 1': [],
		'SNR > 2': [],
		'SNR > 3': []
	}
	
	# Categorize based on actual SNR values
	for td in trade_details:
		symbol = td['symbol']
		buy_price = td['buy_price']
		sell_price = td['sell_price']
		return_pct = td['return_pct']
		gain = td['gain']
		snr = td.get('snr', None)
		
		if snr is not None:
			if snr > 3:
				snr_categories['SNR > 3'].append({
					'symbol': symbol,
					'buy_price': buy_price,
					'sell_price': sell_price,
					'return_pct': return_pct,
					'gain': gain,
					'snr': snr
				})
			elif snr > 2:
				snr_categories['SNR > 2'].append({
					'symbol': symbol,
					'buy_price': buy_price,
					'sell_price': sell_price,
					'return_pct': return_pct,
					'gain': gain,
					'snr': snr
				})
			elif snr > 1:
				snr_categories['SNR > 1'].append({
					'symbol': symbol,
					'buy_price': buy_price,
					'sell_price': sell_price,
					'return_pct': return_pct,
					'gain': gain,
					'snr': snr
				})
	
	# Print results for each category
	for category, trades in snr_categories.items():
		if trades:
			total_gain = sum(trade['gain'] for trade in trades)
			avg_return = np.mean([trade['return_pct'] for trade in trades])
			avg_snr = np.mean([trade['snr'] for trade in trades])
			print(f"\n{category}:")
			print(f"  Number of trades: {len(trades)}")
			print(f"  Total gain: ${total_gain:,.2f}")
			print(f"  Average return: {avg_return:.2f}%")
			print(f"  Average SNR: {avg_snr:.2f}")
			print("  Individual trades:")
			for trade in trades:
				print(f"    {trade['symbol']}: Buy ${trade['buy_price']:.2f} -> Sell ${trade['sell_price']:.2f} | Return: {trade['return_pct']:.2f}% | Gain: ${trade['gain']:.2f} | SNR: {trade['snr']:.2f}")
		else:
			print(f"\n{category}: No trades")
	
	# Per-stock net gain/loss and percent change
	print("\n=== PER-STOCK NET GAIN/LOSS ===")
	stock_invested = {}
	stock_realized = {}
	for td in trade_details:
		symbol = td['symbol']
		stock_invested.setdefault(symbol, 0.0)
		stock_realized.setdefault(symbol, 0.0)
		stock_invested[symbol] += td['buy_price'] * td['shares']
		stock_realized[symbol] += td['sell_price'] * td['shares']
	
	for symbol in stock_invested:
		invested = stock_invested[symbol]
		realized = stock_realized[symbol]
		net = realized - invested
		pct = (net / invested * 100) if invested > 0 else 0.0
		print(f"{symbol}: Invested ${invested:.2f}, Realized ${realized:.2f}, Net Gain/Loss ${net:.2f} ({pct:.2f}%)")

	# Analyze prediction patterns for high SNR trades
	print("\n=== PREDICTION ANALYSIS FOR HIGH SNR TRADES ===")
	high_snr_trades = [td for td in trade_details if td.get('snr', 0) > 3]
	if high_snr_trades:
		print(f"Found {len(high_snr_trades)} trades with SNR > 3")
		print("Analyzing prediction patterns...")
		
		# Group by SNR ranges to see patterns
		snr_ranges = {
			'SNR 3-5': [],
			'SNR 5-10': [],
			'SNR 10+': []
		}
		
		for trade in high_snr_trades:
			snr = trade['snr']
			if snr <= 5:
				snr_ranges['SNR 3-5'].append(trade)
			elif snr <= 10:
				snr_ranges['SNR 5-10'].append(trade)
			else:
				snr_ranges['SNR 10+'].append(trade)
		
		for range_name, trades in snr_ranges.items():
			if trades:
				avg_return = np.mean([t['return_pct'] for t in trades])
				avg_snr = np.mean([t['snr'] for t in trades])
				print(f"\n{range_name}:")
				print(f"  Number of trades: {len(trades)}")
				print(f"  Average return: {avg_return:.2f}%")
				print(f"  Average SNR: {avg_snr:.2f}")
				print("  Trades:")
				for trade in trades:
					print(f"    {trade['symbol']}: SNR={trade['snr']:.2f}, Return={trade['return_pct']:.2f}%")
	else:
		print("No trades with SNR > 3 found")

if __name__ == "__main__":
	main() 