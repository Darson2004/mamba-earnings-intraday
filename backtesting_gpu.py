import argparse
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import ClosePrice
from mambastock_model import MambaStock

# Prefer the picky strategy file if available
try:
    from strategy import TradingStrategy, Trade
except Exception:
    from bare_strategy import TradingStrategy, Trade  # fallback


# ------------------------------ Reproducibility ------------------------------

def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------ Model ------------------------------

def load_model(device: torch.device, model_path: str, input_size: int, seq_len: int, pred_len: int) -> MambaStock:
    print("🔧 Creating model…")
    model = MambaStock(input_size=input_size, seq_len=seq_len, pred_len=pred_len).to(device)
    model.eval()
    if device.type == "cuda":
        try:
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("✅ Enabled cuDNN benchmark + TF32")
        except Exception:
            pass
    if model_path and os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"✅ Loaded weights: {model_path}")
    else:
        print(f"⚠️  Weights not found: {model_path}. Using random init.")
    return model


# ------------------------------ Date bucketing ------------------------------

def extract_date(name: str) -> str:
    # Simple patterns; retain "unknown_date" by request (critique #10 excluded)
    import re
    pats = [
        r"(\d{4})-(\d{1,2})-(\d{1,2})",
        r"(\d{4})/(\d{1,2})/(\d{1,2})",
        r"(\d{4})\.(\d{1,2})\.(\d{1,2})",
        r"(\d{4})(\d{2})(\d{2})",
        r"(\d{1,2})/(\d{1,2})/(\d{4})",
    ]
    for p in pats:
        m = re.search(p, name)
        if not m:
            continue
        g = m.groups()
        if len(g[0]) == 4:
            y, mo, d = g[0], g[1].zfill(2), g[2].zfill(2)
        else:
            mo, d, y = g[0].zfill(2), g[1].zfill(2), g[2]
        return f"{y}-{mo}-{d}"
    return "unknown_date"


def build_date_batches(name_date_list: List[str]) -> List[Tuple[str, List[int]]]:
    date_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, name in enumerate(name_date_list):
        date = extract_date(name)
        date_to_indices[date].append(i)
    # Keep unknown_date bucket as requested
    batches = sorted(list(date_to_indices.items()), key=lambda kv: kv[0])
    return batches


# ------------------------------ PnL ------------------------------

def realized_pnl(trades: List[Trade]) -> float:
    buys = [t for t in trades if t.side == "BUY"]
    sells = [t for t in trades if t.side == "SELL"]
    return sum(t.cash_flow for t in sells) - sum(t.cash_flow for t in buys)


# ------------------------------ CSV & plots ------------------------------

def save_predictions_csv(strategy: TradingStrategy, stock_indices: List[int], dataset: ClosePrice, out_path: str) -> None:
    import csv
    idx_by_name = {dataset.name_date[i]: i for i in stock_indices}
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stock", "timestamp", "mu", "action", "price"])
        w.writeheader()
        for p in strategy.all_predictions:
            name = p["stock"]
            t = p["time"]
            idx = idx_by_name.get(name)
            price = dataset.data[idx][t, 3].item() if idx is not None and 0 <= t < len(dataset.data[idx]) else ""
            w.writerow({"stock": name, "timestamp": t, "mu": p["mu"], "action": p["action"], "price": price})
    print(f"✅ Saved predictions: {out_path}")


def plot_mu_vs_price(strategy: TradingStrategy, stock_indices: List[int], dataset: ClosePrice, date: str, out_png: str) -> None:
    try:
        names = list({p["stock"] for p in strategy.all_predictions})
        if not names:
            print("    ⚠️  No predictions to plot.")
            return
        names = names[:2]
        fig, axes = plt.subplots(len(names), 1, figsize=(14, 4 * len(names)))
        if len(names) == 1:
            axes = [axes]
        for ax, name in zip(axes, names):
            idx = {dataset.name_date[i]: i for i in stock_indices}.get(name)
            preds = [p for p in strategy.all_predictions if p["stock"] == name]
            ts = [p["time"] for p in preds]
            mus = [p["mu"] * 100.0 for p in preds]
            prices = [dataset.data[idx][t, 3].item() for t in ts]
            ax.plot(ts, prices, label="Price", linewidth=2)
            ax2 = ax.twinx()
            ax2.plot(ts, mus, "--", label="mu20 (%)", linewidth=2)
            ax.set_title(f"{name} — {date}")
            ax.set_xlabel("Minute since 9:30")
            ax.set_ylabel("Price")
            ax2.set_ylabel("%")
            ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"    📊 Saved plot: {out_png}")
    except Exception as e:
        print(f"    ⚠️  Plot failed: {e}")


# ------------------------------ Backtest runner ------------------------------

@torch.inference_mode()
def run_backtest(
    h5_path: str,
    model_path: str,
    *,
    initial_portfolio: float = 100000.0,
    seq_len: int = 50,
    pred_len: int = 20,
    trade_start: int = 90,
    trade_end: int = 240,
    predict_start: int = 0,
    per_name_cap: float = 0.01,
    gross_add_cap: float = 0.02,
    top_k: int = None,
    spread_bps: float = 5.0,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
    buffer_bps: float = 0.0,
    clamp_k: float = 2.0,
    seed: int = 1337,
    save_csv: bool = True,
    make_plots: bool = True,
):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    dataset = ClosePrice(h5_path)
    print(f"📦 Dataset: {dataset.n_stocks} stocks | {dataset.day_length} minutes/day")

    # Infer input_size from dataset features
    input_size = dataset.data[0].shape[1]
    model = load_model(device, model_path, input_size=input_size, seq_len=seq_len, pred_len=pred_len)

    date_batches = build_date_batches(dataset.name_date)
    print(f"📅 Trading dates: {len(date_batches)} (including 'unknown_date' if present)")

    portfolio = float(initial_portfolio)
    total_realized = 0.0

    for day_idx, (date, indices) in enumerate(date_batches, start=1):
        print("=" * 72)
        print(f"📊 DAY {day_idx}/{len(date_batches)} — {date} | Universe={len(indices)} | Start=${portfolio:,.2f}")

        strat = TradingStrategy(
            model=model,
            device=device,
            dataset=dataset,
            considered_stocks=indices,
            portfolio_value=portfolio,
            seq_len=seq_len,
            pred_len=pred_len,
            predict_start_minute=predict_start,
            start_trading_minute=trade_start,
            trade_end_minute=trade_end,
            max_position_frac=per_name_cap,
            gross_add_cap=gross_add_cap,
            top_k=top_k,
            buy_threshold=0.001,
            stop_loss=-0.005,
            liquidate_eod=True,
            spread_bps=spread_bps,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            buffer_bps=buffer_bps,
            clamp_k=clamp_k,
        )

        trades, _ = strat.run_day()
        pnl = realized_pnl(trades)
        total_realized += pnl
        portfolio += pnl

        print(f"💵 {date}: Realized=${pnl:,.2f} | End=${portfolio:,.2f} | Trades={len(trades)}")
        strat.analyze_spread_impact()

        if save_csv:
            save_predictions_csv(strat, indices, dataset, f"predictions_{date}.csv")
        if make_plots:
            plot_mu_vs_price(strat, indices, dataset, date, f"mu_vs_price_{date}.png")

    print("=" * 72)
    print("🎯 FINAL")
    print(f"  Initial: ${initial_portfolio:,.2f}")
    print(f"  Final:   ${portfolio:,.2f}")
    print(f"  Return:  ${total_realized:,.2f}  ({(total_realized/initial_portfolio)*100:.2f}%)")


# ------------------------------ CLI ------------------------------

def main():
    p = argparse.ArgumentParser(description="Backtest TradingStrategy (picky, no‑chaining, cost‑aware)")
    p.add_argument("--h5", type=str, required=True, help="Path to HDF5 file")
    p.add_argument("--model", type=str, required=True, help="Path to model weights (.pth)")
    p.add_argument("--init", type=float, default=100000.0, help="Initial portfolio value")
    p.add_argument("--seq_len", type=int, default=50)
    p.add_argument("--pred_len", type=int, default=20)
    p.add_argument("--trade_start", type=int, default=90)
    p.add_argument("--trade_end", type=int, default=240)
    p.add_argument("--predict_start", type=int, default=0)
    p.add_argument("--per_name_cap", type=float, default=0.01)
    p.add_argument("--gross_add_cap", type=float, default=0.02)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--spread_bps", type=float, default=5.0)
    p.add_argument("--commission_bps", type=float, default=0.0)
    p.add_argument("--slippage_bps", type=float, default=0.0)
    p.add_argument("--buffer_bps", type=float, default=0.0)
    p.add_argument("--clamp_k", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--no_csv", action="store_true")
    p.add_argument("--no_plots", action="store_true")

    args = p.parse_args()

    run_backtest(
        h5_path=args.h5,
        model_path=args.model,
        initial_portfolio=args.init,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        trade_start=args.trade_start,
        trade_end=args.trade_end,
        predict_start=args.predict_start,
        per_name_cap=args.per_name_cap,
        gross_add_cap=args.gross_add_cap,
        top_k=args.top_k,
        spread_bps=args.spread_bps,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        buffer_bps=args.buffer_bps,
        clamp_k=args.clamp_k,
        seed=args.seed,
        save_csv=not args.no_csv,
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
