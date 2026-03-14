# ECE C147A Stock Return Prediction Project Summary

## Overview

This project studies cross-sectional stock return prediction from daily price and volume data.
The main goal is to predict short-horizon future returns and evaluate whether the predictions
produce useful stock rankings for a simple long-short strategy.

The repository evolved from an initial single-notebook prototype into a reusable pipeline with:

- data download and caching
- feature engineering on OHLCV time series
- panel dataset construction with rolling lookback windows
- recurrent and transformer model training
- cross-sectional IC / RankIC evaluation
- long-short portfolio backtesting

## Current Core Setup

- Prediction target: short-horizon forward return
- Lookback window: 60 trading days
- Evaluation focus: `mean_ic`, `mean_rank_ic`, `mean_return`, `sharpe_ratio`
- Main experiment notebook: `C147A_project_average.ipynb`
- Seeded comparison notebook: `C147A_project_seeded.ipynb`
- Single-run development notebook: `C147A_project_nonseeded.ipynb`
- Main training script: `run_project_mhattn.py`
- Core library: `stock_return_core.py`

## Models

The current comparison includes four sequence models:

- `RNN`: vanilla recurrent baseline
- `LSTM`: recurrent model with temporal attention
- `GRU`: recurrent model with temporal attention
- `TRANSFORMER`: transformer encoder baseline

In the notebook experiments, `LSTM` and `GRU` use multi-head temporal attention with
`ATTN_HEADS = 4`.

## Data and Pipeline

The main pipeline downloads daily market data, filters incomplete histories, engineers technical
features, and converts the data into a panel suitable for cross-sectional forecasting.

Typical workflow:

1. Download adjusted daily data for a chosen universe.
2. Filter tickers with weak history coverage.
3. Build rolling feature windows and forward-return targets.
4. Split the sample into train / validation / test periods.
5. Train each model and track validation loss.
6. Evaluate test predictions with IC metrics and a simple long-short backtest.

## Latest Averaged Result

The most reliable current result is the multi-seed average notebook run on the `small` universe,
using seeds `[40, 41, 42]`.

Average summary:

| Model | mean_ic | mean_rank_ic | mean_return | sharpe_ratio |
| --- | ---: | ---: | ---: | ---: |
| LSTM | 0.066692 | 0.045450 | 0.005756 | 3.713478 |
| GRU | 0.063823 | 0.043322 | 0.005960 | 3.689846 |
| TRANSFORMER | 0.061505 | 0.037881 | 0.005287 | 3.209140 |
| RNN | 0.025367 | 0.011869 | 0.002093 | 1.724486 |

Stability observations:

- `LSTM` and `GRU` are the strongest models overall.
- `GRU` is slightly more stable across seeds than `LSTM` and `TRANSFORMER`.
- `TRANSFORMER` remains competitive, but its ranking is more seed-sensitive.
- `RNN` works as a meaningful weak baseline and clearly trails the other three.

## Interpretation

The main lesson from the multi-seed average is that single-run rankings were too noisy to trust.
Across different seeds, `LSTM`, `GRU`, and `TRANSFORMER` can each look best in isolated runs.
After averaging, the more defensible conclusion is:

- `LSTM` and `GRU` are the top-performing and most reliable models on the current small-universe setup.
- `TRANSFORMER` is strong, but not consistently dominant.
- `RNN` is useful as a baseline, not as the preferred final model.

## Important Caveats

- The broad-universe experiments may contain survivorship bias if current constituent lists are
  used for earlier periods.
- Yahoo Finance coverage varies by ticker and date range.
- Results on the `small` universe are materially more volatile than broad-universe results.
- The backtest is simplified and does not model transaction costs or market impact.
- Current conclusions are strongest for relative model comparison, not for claiming a deployable
  trading strategy.

## File Guide

- `C147A_project_average.ipynb`: main multi-seed notebook with outputs
- `C147A_project_seeded.ipynb`: fixed-seed notebook with outputs
- `C147A_project_nonseeded.ipynb`: single-run development notebook
- `run_project_mhattn.py`: script entry point for attention experiments
- `run_project_attention_ts.py`: related experiment script
- `stock_return_core.py`: shared project library
- `C147_proposal.pdf`: original project proposal

Legacy files and earlier experiments are stored under `archive/` and are no longer the primary
entry points.