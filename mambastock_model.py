# mambastock_model.py

import torch
import torch.nn as nn
from mamba import Mamba, MambaConfig  # Using the same Mamba package as main.py

class MambaStock(nn.Module):
    def __init__(self, input_size=7, seq_len=60, pred_len=1):
        super(MambaStock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_proj = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(0.3)
        self.config = MambaConfig(d_model=64, n_layers=1)
        self.mamba = Mamba(self.config)
        self.norm = nn.LayerNorm(64)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.2)
        # Output only one prediction at a time for rolling predictions
        self.output = nn.Linear(64, 1)  # Only predict pct_chg for next step
        self.activation = nn.Tanh()

    def forward(self, x):  # x: [B, T, features]
        x = self.input_proj(x)         # → [B, T, 64]
        x = self.dropout1(x)
        x = self.mamba(x)              # → [B, T, 64]
        x = self.norm(x)
        x = self.dropout2(x)
        # Take only the last token for next-step prediction
        x = x[:, -1:, :]                           # [B, 1, 64] - only last token
        x = self.dropout3(x)
        out = self.output(x)           # → [B, 1, 1]
        return self.activation(out)
