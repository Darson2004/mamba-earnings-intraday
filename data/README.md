# Data

No market data is committed to this repository. Intraday bars from Databento and Bloomberg are
licensed and cannot be redistributed, and the HDF5 training files are derived from them. Everything
the scripts read or write under `data/` (and any `*.csv`, `*.h5`, `*.xlsx`, `*.pth` in the working
directory) is git-ignored.

## 1. Intraday bars around earnings (the MFT pipeline)

Inputs you need:

- An earnings calendar workbook, `earnings_equity_first.xlsx`: a `NAME` column (Bloomberg-style
  ticker, e.g. `JELD UN Equity`) and one or more columns whose names contain `DATE`, one per
  earnings announcement.
- A Databento account. Put the key in the environment (`DATABENTO_API_KEY`) or in a `.env` file,
  which `python-dotenv` loads.

Steps (scripts in `data_tools/`; they read and write in the current directory):

| Step | Script | Output |
|---|---|---|
| Pull 1-minute OHLCV (`XNAS.ITCH`, `ohlcv-1m`) for each announcement date and the following day | `databento_intraday.py` (set `EARNINGS_XLSX` to point at the workbook) | `test_training_two.csv` (UTC timestamps) |
| Alternative source: Bloomberg 1-minute bars (requires a Terminal session on `localhost:8194`) | `bbg_intraday_bars.py`, `xbbg_example.py` | DataFrame per ticker/date |
| Convert UTC to US/Eastern and keep 09:30-16:00 | `csv_converter.py` | `EST_earnings_ohlcv_data_*.csv` |
| Decide which of the two days is the earnings reaction day (before-open vs after-close report), by post-10:00 price range or by volume | `sifter.py`, `volume_sifter.py` | `sifted_percentage_change.csv`, `volume_sifter.csv` |
| Summary moves (09:30 to 10:00 and open-to-close) per name/day | `percentage_change.py` | `percentage_changes_four.csv` |
| Forward-fill daily fundamentals (`pe`, `total_share`, `float_share`) across missing calendar days | `extras_fixer.py` | `rty_extras_filled_all_stocks_FIXED.csv` |
| Build the HDF5 training file: one dataset per `TICKER-DATE`, 390 minutes x 13 features | `preprocess.py --input_file <minute csv> --output_file data/prices.h5` | HDF5 |
| (Older 11-column files only) add NaN masks, impute, cap at the 1st/99th percentiles | `preprocess_h5.py --input <dir>` | `*_processed.h5` |
| Split or recombine HDF5 files (by count, by day, by stock) | `split_hdf5.py`, `combine_hdf5.py` | HDF5 |

`preprocess.py` expects one minute-level CSV with columns
`ts_code, date, trade_time, open, high, low, close, volume, pe, turnover_rate, total_share, float_share`.
The join of the intraday bars with the daily fundamentals file (`rty_extras.csv`, a Russell 2000
universe) was done by hand and is not scripted here, and the source of those daily fundamentals is
not recorded in the repo.

The 13 features per minute, in column order, are:

```
open, high, low, close, volume, pe, turnover_rate, total_share, float_share,
trade_time_min, pct_chg (close vs. the 09:30 open), turnover_rate_mask, pe_nan_mask
```

The training and backtest scripts take HDF5 paths as constants or CLI flags
(`src/train_two_gpu.py`: `h5_path = "training_data.h5"`; `src/backtesting.py --h5`;
`src/backtesting_gpu.py --h5 ... --model ...`).

## 2. Single-series CSV for `src/main.py` (the first experiment)

`src/main.py` is the upstream MambaStock script with a small change: upstream parsed `trade_date`
as a daily date (`%Y%m%d`) and forecast daily A-share closes; here `trade_date` is parsed as an
intraday `HHMMSS` stamp, so the same model is fit to one day of minute bars. It reads
`<ts-code>.SH.csv` from the working directory, with TuShare-style columns
`ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, ...` (the target is
`pct_chg / 100`; the remaining numeric columns are the features). The upstream repository shipped four daily TuShare CSVs
(for example `600036.SH.csv`); they are removed here because they are market data. To reproduce
upstream's daily experiment instead, download daily bars from [TuShare](https://tushare.pro/) and
revert the one-line date parse to `format='%Y%m%d'`.

```bash
python path/to/src/main.py --ts-code 601988 --n-test 300
```
