# Running on CPU and fitting in GPU memory

Merged from two working notes written during development (`README_CPU.md` and
`MEMORY_OPTIMIZATION_GUIDE.md`). Those notes described an earlier trainer (`train_two.py`, which
used Monte Carlo dropout confidence intervals); the current trainer is `src/train_two_gpu.py`. The
guidance below is updated to its settings.

## Device selection

All training, evaluation and backtest scripts pick the device automatically:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

On CPU, `train_two_gpu.py` caps PyTorch at 8 threads and reduces any batch size above 4 to 4.
Expect CPU training to be several times slower than on a single GPU; results are otherwise
identical. `comparison_gpu.py` wraps models in `DataParallel` when more than one GPU is visible.

Install a CPU-only PyTorch build with:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Quick environment check (imports the model and dataset modules, looks for the data/weights files):

```bash
python tests/test_cpu_compatibility.py
```

## Memory knobs in `src/train_two_gpu.py`

| Setting | Current value | Effect on memory |
|---|---|---|
| `MEMORY_SAVING_MODE` | `True` | switches the settings below to their low-memory values |
| `BATCH_SIZE` | 1 (8 when off) | per-step batch |
| `MONTE_CARLO_SAMPLES` | 15 (20 when off) | defined but unused by the current trainer (left over from the MC-dropout version) |
| `SEQUENCE_LENGTH` | 90 minutes | input window length |
| `PREDICTION_LENGTH` | 20 minutes | horizon of the single log-return target |
| `STOCK_BATCH_SIZE` | 500 | stocks per forward pass; batching only engages when the universe exceeds it (the dataset caps it at 200) |
| `GRADIENT_ACCUMULATION_STEPS` | 4 | optimizer steps every 4 stock batches, simulating a larger batch |

Other memory measures already in the code:

- Automatic mixed precision (`torch.cuda.amp.autocast` + `GradScaler`) on CUDA.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation.
- `src/dataset.py` (`ClosePrice`) loads HDF5 samples lazily by default, can preload everything to
  the GPU with `load_to_gpu=True`, and caps the universe at 200 stock-days (`max_stocks`).

## If you still hit out-of-memory errors

1. Lower `STOCK_BATCH_SIZE` (for example 250 or 128) and raise `GRADIENT_ACCUMULATION_STEPS` to
   keep the effective batch constant.
2. Shorten `SEQUENCE_LENGTH` (the Mamba scan cost is linear in sequence length).
3. Lower `max_stocks` in `ClosePrice`.
4. Monitor usage with `torch.cuda.memory_allocated()`; the trainer's `monitor_memory_usage()`
   prints it periodically.
5. Fall back to CPU temporarily with `device = torch.device("cpu")`.

`tests/test_memory_fix.py`, `tests/test_memory_optimizations.py` and
`tests/test_simple_data_loading.py` load a processed HDF5 file and report memory before and after,
which is a quick way to check a configuration before a long run.
