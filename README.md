# Exponential Wake-Up Overshoot in Explicit Gaussian Point-Process Filters

Code and results for

> H. Qin and D. Wang, "Exponential wake-up overshoot in explicit Gaussian point-process filters,"
> arXiv:2610.35958, 2026. <https://arxiv.org/abs/2610.35958>

This repository contains what is needed to reproduce the numerical checks of the theory, the
synthetic and real-data experiments, and the figures. Section and appendix numbers below refer
to the arXiv version.

## Layout

| Path | Content |
|---|---|
| `experiment/` | Filters (C++ kernel `engine.cpp` with a Python wrapper), independent NumPy/SciPy reference implementation, stream generators, Binance loader, tuning, statistics, and the experiment runner |
| `experiment/frozen_params.json` | The 20 tuned filter configurations (5 methods × 4 bin widths) used on the test days |
| `experiment/theory_regions.json` | Crossing index κ and region of each of the 200 synthetic cells (computed from the theory, no data) |
| `experiment/x_coverage.json` | Cells in which the grid-based Bayes filter X is run |
| `results/results.json` | Full results: per-cell sign tests and Holm decisions (H1–H3), descriptive summaries, real-data records and decisions (R1–R3) |
| `results/real/` | Real-data scores per contract-day (20 files, 20 records each) |
| `theory/verify/` | Scripts that check the propositions numerically; `theory/outputs/` holds their outputs |
| `figures/` | Figure scripts, plotted data (`*.csv`), and the figures |
| `posthoc/` | Two descriptive checks reported in the paper (contract-level aggregation; silence-bin loss) |

Code comments and runtime messages are in Chinese. Labels in the JSON files:
`越率区` crossing region, `不越区` no-crossing region, `过渡区` transition region;
`成立` confirmed, `不成立` not confirmed.

## Requirements

- Python ≥ 3.10 with numpy, scipy, pandas, matplotlib and psutil (`pip install -r requirements.txt`).
  Tested with Python 3.13, numpy 2.4, scipy 1.18, pandas 3.0, matplotlib 3.10, psutil 7.2.
- A C++17 compiler. The build uses `g++ -O2 -std=c++17 -static -Wall -Wextra`; set `GXX` if `g++`
  is not on `PATH`. Tested with MinGW-w64 g++ (MSYS2 UCRT64) on Windows 11.

## Reproducing

All commands are run from `experiment/`.

```sh
python build.py           # compile engine.cpp -> engine.exe
python check_engine.py    # C++ kernel vs NumPy/SciPy reference, bin by bin (19 checks)
```

### Synthetic experiment (no external data)

```sh
python generate_regions.py                    # optional: recompute theory_regions.json
python run_experiment.py --synthetic-only --workers 6
python compare_results.py runs/reproduction/results.json ../results/results.json
```

The streams are generated from fixed seeds (`config.BASE`), so the rerun is bit-for-bit
identical to the published results. The run is resumable: per-stream results are saved in
`runs/<run>/synthetic/` and reused if the run is interrupted.

### Real data (Binance)

The real-data part uses daily aggTrades files of ten USDⓈ-M perpetual contracts
(SNDK, MU, TSLA, NVDA, SPY, SPX, QQQ, MSTR, COIN, AAPL, each against USDT) from
[Binance Data Collection](https://data.binance.vision) (`data/futures/um/daily/aggTrades/`).
The data are not redistributed here. Download, for every contract `<S>USDT`,

- training days 2026-10-10 and 2026-10-11 (only for re-tuning),
- test days 2026-10-17 and 2026-10-18,

as `<S>USDT-aggTrades-<day>.zip` (with the optional `.CHECKSUM` file next to it) into one folder,
and point `BINANCE_RAW` to it (default `../data/raw`). Then

```sh
python run_experiment.py --workers 6                # synthetic + real; ~1 h with 6 workers
python compare_results.py runs/reproduction/results.json ../results/results.json
```

To redo the tuning on the training days (several hours):

```sh
python binance_loader.py   # writes training_manifest.json (training days only)
python tune_train.py --workers 6
```

`tune_train.py` writes the selected configurations to `tuning/frozen_params.json` for comparison
with the published `frozen_params.json`, which the experiment runner uses.

### Theory checks and figures

The theory outputs reproduce up to floating-point rounding: with other numpy/scipy versions only
the columns that record numerical errors change, by at most about 1e-12 in absolute value.

```sh
python theory/verify/run_all.py      # from the repository root; ~2 min; rewrites theory/outputs/
python figures/make_fig_theory.py theory/outputs figures             # Figs. 1-2
python figures/cell_medians.py experiment/runs/reproduction experiment/theory_regions.json figures/cell_medians.csv
python figures/make_fig_experiment.py figures/cell_medians.csv figures/fig3_experiment.pdf  # Fig. 3
python posthoc/contract_level.py results/real
python posthoc/silence_loss.py
```

## License

Code: Apache-2.0 (see `LICENSE`). The real-data results in `results/` are derived from Binance market
data, which Binance provides under CC BY-NC-SA 4.0; they are shared under the same license with
attribution to Binance. The synthetic results and theory outputs are released under CC BY 4.0.

## Citation

```bibtex
@misc{QinWang2026,
  author = {Hanwu Qin and Dianhai Wang},
  title  = {Exponential Wake-Up Overshoot in Explicit {Gaussian} Point-Process Filters},
  year   = {2026},
  eprint = {2610.35958},
  archivePrefix = {arXiv}
}
```
