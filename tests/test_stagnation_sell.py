#!/usr/bin/env python3
"""
Test script to demonstrate the new stagnation-based sell feature.
This shows how the strategy will sell positions when price is flat for 15+ minutes
and there's a positive return of at least 1%.
"""

import torch
import numpy as np
from collections import defaultdict

# Mock TradingStrategy class with just the stagnation logic
class StagnationDemo:
    def __init__(self):
        self.stagnation_threshold = 15  # 15 minutes
        self.min_profit_for_stagnation_sell = 0.01  # 1% minimum profit
        self.stagnation_range = 0.002  # ±0.2% range for stagnation
        self.stagnation_start_price = {}  # Track starting price when entering stagnation range
        self.stagnation_start_time = {}  # Track when stagnation period began
        
    def update_price_history(self, symbol, price, time):
        """Update price history for stagnation tracking (kept for compatibility)"""
        pass  # No longer needed for range-based approach
        
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
                print(f"    🔄 {symbol} exited stagnation range: {price_change*100:.2f}% from start price")
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
            return True
        
        return False

def test_stagnation_sell():
    """Test the stagnation sell logic with simulated price data"""
    demo = StagnationDemo()
    
    print("=== Testing Stagnation-Based Sell Logic ===\n")
    
    # Test Case 1: Price stays within ±0.2% range for 15+ minutes with profit
    print("Test Case 1: Price within ±0.2% range for 15+ minutes with 2% profit")
    symbol = "AAPL"
    entry_price = 100.0
    base_price = 102.0  # 2% profit
    
    # Simulate 16 minutes within ±0.2% range
    for t in range(16):
        # Add small random noise within ±0.15% (inside the 0.2% range)
        noise = np.random.uniform(-0.15, 0.15)  # ±0.15% noise (within ±0.2% range)
        price = base_price + (base_price * noise / 100)
        demo.update_price_history(symbol, price, t)
        
        net_return = (price - entry_price) / entry_price
        is_stagnant = demo.check_price_stagnation(symbol, price, t)
        
        print(f"  t={t:2d}: price=${price:.3f}, return={net_return*100:.1f}%, stagnant={is_stagnant}")
        
        # Check sell condition
        if (net_return >= demo.min_profit_for_stagnation_sell and is_stagnant):
            print(f"  🚨 STAGNATION SELL TRIGGERED! Profit={net_return*100:.1f}%, in range for {demo.stagnation_threshold}+ min")
            break
    
    print("\n" + "="*50 + "\n")
    
    # Test Case 2: Price exits the ±0.2% range (resets stagnation)
    print("Test Case 2: Price exits ±0.2% range, then re-enters")
    demo2 = StagnationDemo()
    symbol2 = "GOOGL"
    entry_price2 = 150.0
    base_price2 = 151.5  # 1% profit
    
    for t in range(20):
        if t < 8:
            # First 8 minutes: stay within range
            noise = np.random.uniform(-0.15, 0.15)  # ±0.15% (within range)
            price = base_price2 + (base_price2 * noise / 100)
        elif t == 8:
            # At t=8: exit the range significantly (>0.2%)
            price = base_price2 + (base_price2 * 0.3 / 100)  # 0.3% move (exits range)
        else:
            # After t=8: back within range (new stagnation period starts)
            noise = np.random.uniform(-0.15, 0.15)
            price = base_price2 + (base_price2 * noise / 100)
        
        demo2.update_price_history(symbol2, price, t)
        net_return = (price - entry_price2) / entry_price2
        is_stagnant = demo2.check_price_stagnation(symbol2, price, t)
        
        print(f"  t={t:2d}: price=${price:.3f}, return={net_return*100:.1f}%, stagnant={is_stagnant}")
    
    print("\n" + "="*50 + "\n")
    
    # Test Case 3: Stagnant but not profitable enough
    print("Test Case 3: Stagnant but only 0.5% profit (below 1% threshold)")
    demo3 = StagnationDemo()
    symbol3 = "MSFT"
    entry_price3 = 200.0
    current_price3 = 201.0  # Only 0.5% profit
    
    for t in range(16):
        # Very flat pricing
        noise = np.random.uniform(-0.02, 0.02)  # ±0.02% noise
        price = current_price3 + (current_price3 * noise / 100)
        demo3.update_price_history(symbol3, price, t)
        
        net_return = (price - entry_price3) / entry_price3
        is_stagnant = demo3.check_price_stagnation(symbol3, price, t)
        sell_condition = (net_return >= demo3.min_profit_for_stagnation_sell and is_stagnant)
        
        print(f"  t={t:2d}: price=${price:.3f}, return={net_return*100:.1f}%, stagnant={is_stagnant}, sell={sell_condition}")
    
    print("\n🎯 Summary:")
    print("✅ Stagnation sell triggers when:")
    print("   1. Price stays within ±0.2% range for 15+ minutes")
    print("   2. Position has ≥1% profit")
    print("❌ Does NOT trigger when:")
    print("   1. Price exits the ±0.2% range (resets timer)")
    print("   2. Profit is below 1% threshold")
    print("   3. Not enough time in range (<15 minutes)")
    print("\n📊 Range-based logic:")
    print("   • Tracks starting price when entering potential stagnation")
    print("   • Resets completely if price moves >0.2% from start")
    print("   • Only triggers after 15 consecutive minutes in range")

if __name__ == "__main__":
    test_stagnation_sell() 