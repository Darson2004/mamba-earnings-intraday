#!/usr/bin/env python3
"""
Test script to demonstrate the new timestep categorization function
"""

import torch
import numpy as np
from strategy import TradingStrategy

# Mock model and dataset for testing
class MockModel:
    def __init__(self):
        self.training = False
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False
    
    def __call__(self, x):
        # Return mock predictions
        batch_size = x.shape[0]
        return torch.randn(batch_size) * 0.02 + 0.01  # Small positive predictions

class MockDataset:
    def __init__(self):
        self.name_date = ['AAPL', 'GOOGL', 'MSFT']
        self.day_length = 400
        # Create mock data with shape [n_stocks, time_steps, features]
        self.data = torch.randn(3, 400, 15)  # 3 stocks, 400 timesteps, 15 features
        # Set close prices (feature index 3) to reasonable values
        for i in range(3):
            base_price = 100 + i * 50  # Different base prices for each stock
            for t in range(400):
                self.data[i, t, 3] = base_price + torch.randn(1) * 5  # Close price with some noise

def test_timestep_categorization():
    """Test the new timestep categorization function"""
    print("=== Testing Timestep Categorization Function ===\n")
    
    # Create mock components
    device = torch.device('cpu')
    model = MockModel()
    dataset = MockDataset()
    considered_stocks = [0, 1, 2]  # Use all 3 stocks
    portfolio_value = 10000
    
    # Create strategy
    strategy = TradingStrategy(
        model=model,
        device=device,
        dataset=dataset,
        considered_stocks=considered_stocks,
        portfolio_value=portfolio_value,
        seq_len=50,
        pred_len=20,
        snr_threshold=1.5,  # Lower threshold for testing
        max_position_frac=0.1,
        n_mc_samples=5  # Fewer samples for faster testing
    )
    
    # Run a short trading session
    print("Running trading strategy...")
    trade_log, positions = strategy.run_day()
    
    print(f"\nTrading completed!")
    print(f"Total trades executed: {len(trade_log)}")
    print(f"Final positions: {len(positions)}")
    
    # Test the new categorization function
    print("\n" + "="*50)
    strategy.categorize_trades_with_timesteps()
    
    print("\n" + "="*50)
    print("Test completed successfully!")

if __name__ == "__main__":
    test_timestep_categorization() 