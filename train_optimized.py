### Optimized version of train.py with parallelism, batch prediction, and faster dataloading

from mambastock_model import MambaStock
from dataset import ClosePrice, TimeAlignedSampler, collate_fn
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from joblib import Parallel, delayed
import os

# --- CONFIG ---
h5_path = "h5_EST_earnings_ohlcv_data_test.h5"
seq_len = 30
pred_len = 10
batch_size = 32
epochs = 10
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LOAD DATA ---
dataset = ClosePrice(h5_path)
print(f"Dataset loaded: {len(dataset)} items, {dataset.n_stocks} stocks")

sampler = TimeAlignedSampler(dataset, batch_size=batch_size, start_time=seq_len, end_time=dataset.day_length-pred_len, shuffle=True)
# Use num_workers=0 to avoid h5py multiprocessing issues
loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn, num_workers=0, pin_memory=True)

if __name__ == '__main__':
    # --- INIT MODEL ---
    model = MambaStock(input_size=7, seq_len=seq_len, pred_len=pred_len).to(device)
    model = torch.compile(model)  # For PyTorch 2.0+
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    scaler = GradScaler()

    # --- TRAIN LOOP ---
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        for batch in loader:
            close_prices = batch['close_prices'].to(device)  # [B, T, F]
            time_index = batch['time_index']
            B = close_prices.size(0)
            X = close_prices[:, -seq_len:, :]
            y = torch.zeros((B, pred_len), device=device)

            for t in range(pred_len):
                with autocast():
                    pred = model(X)
                    y_true = torch.stack([dataset.data[i, time_index + t, 6] for i in range(B)]).to(device)
                    y[:, t] = y_true
                    loss = loss_fn(pred[:, t, 6], y_true)
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    total_loss += loss.item()
                    # Append next real value for next step prediction
                    next_real = torch.stack([dataset.data[i, time_index + t, :] for i in range(B)]).unsqueeze(1).to(device)
                    X = torch.cat([X[:, 1:, :], next_real], dim=1)

            batch_count += 1

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / batch_count:.4f}")

    # --- STRATEGY FUNCTION PARALLELIZED ---
    def parallel_strategy(stock_idx):
        model.eval()
        stock_data = dataset.data[stock_idx]
        time_indices = range(seq_len, dataset.day_length - pred_len)
        preds = []
        for t in time_indices:
            X = stock_data[t-seq_len:t, :].unsqueeze(0).to(device)
            with torch.no_grad():
                p = model(X)
                preds.append(p.cpu().numpy())
        return stock_idx, np.concatenate(preds)

    print("Running strategy in parallel across stocks...")
    results = Parallel(n_jobs=-1)(delayed(parallel_strategy)(i) for i in range(dataset.n_stocks))

    # --- SAVE MODEL ---
    torch.save(model.state_dict(), "mambastock_ohlcv.pth")
