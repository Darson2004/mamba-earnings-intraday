import torch
from torch.utils.data import DataLoader
from mambastock_hdf5_dataset import MambaHDF5Dataset
from mambastock_model import MambaStock
import numpy as np
import argparse
import os

@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    mse_total, count = 0.0, 0
    all_preds, all_targets = [], []
    for x, y in dataloader:
        x, y = x.cuda(), y.cuda()
        pred = model(x)
        mse = torch.mean((pred - y) ** 2).item()
        mse_total += mse * x.size(0)
        count += x.size(0)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())
    return mse_total / count, np.concatenate(all_preds), np.concatenate(all_targets)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--h5_path", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--pred_len", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    dataset = MambaHDF5Dataset(args.h5_path, args.seq_len, args.pred_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = MambaStock(input_size=1, seq_len=args.seq_len, pred_len=args.pred_len)
    model.load_state_dict(torch.load(args.model_path))
    model.cuda()

    mse, preds, targets = evaluate(model, loader)
    print(f"Test MSE: {mse:.6f}")
