"""
Run cross-sectional stock return prediction experiments.

Models trained in one call:
  - TRANSFORMER  : improved hyperparameters, no attention module
  - LSTM         : with temporal attention (train_config1)
  - GRU          : with temporal attention (train_config2)
"""

from __future__ import annotations

import argparse

import pandas as pd
import torch
import torch.nn as nn

from stock_return_project4 import (
    TrainConfig,
    ExperimentData,
    download_price_history,
    prepare_experiment_data,
    make_data_loaders,
    train_model,
    predict_dataset,
    compute_daily_ic,
    backtest_long_short,
    summarize_cross_sectional_metrics,
    build_model,
    DEFAULT_LIQUID_TICKERS,
    NASDAQ100_TICKERS,
    SP500_TICKERS,
)

# ---------------------------------------------------------------------------
# Universe definitions
# ---------------------------------------------------------------------------

UNIVERSES = {
    "small": DEFAULT_LIQUID_TICKERS,   # 50 stocks — fast, good for testing
    "nasdaq100": NASDAQ100_TICKERS,    # current Nasdaq-100 constituents
    "sp500": SP500_TICKERS,            # full S&P 500 list from project file
    "auto":  None,                     # None = scrape Wikipedia at runtime
}


# ---------------------------------------------------------------------------
# Temporal attention module
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores  = self.attn(encoder_outputs).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        context = (weights.unsqueeze(-1) * encoder_outputs).sum(dim=1)
        return context, weights


# ---------------------------------------------------------------------------
# Attention-augmented LSTM / GRU regressor
# ---------------------------------------------------------------------------

class AttentionSequenceRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        cell_type: str = "LSTM",
    ) -> None:
        super().__init__()
        if cell_type not in {"LSTM", "GRU"}:
            raise ValueError(f"Only LSTM/GRU supported, got {cell_type}")

        rnn_cls = nn.LSTM if cell_type == "LSTM" else nn.GRU
        self.encoder = rnn_cls(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.last_attn_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_outputs, _ = self.encoder(x)
        context, weights   = self.attention(encoder_outputs)
        self.last_attn_weights = weights.detach()
        return self.head(context).squeeze(-1)


# ---------------------------------------------------------------------------
# Helper: train + evaluate one model
# ---------------------------------------------------------------------------

def run_experiment(
    model: nn.Module,
    model_name: str,
    experiment_data: ExperimentData,
    train_config: TrainConfig,
) -> dict:
    print(f"\n=== Training {model_name} ===", flush=True)

    loaders = make_data_loaders(
        experiment_data.datasets,
        batch_size=train_config.batch_size,
        device=train_config.device,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
        persistent_workers=train_config.persistent_workers,
    )
    model, history = train_model(model, loaders["train"], loaders["val"], train_config)

    device = train_config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    val_dataset  = experiment_data.datasets["val"]
    test_dataset = experiment_data.datasets["test"]

    val_predictions  = predict_dataset(model, loaders["val"],  val_dataset.dates,  val_dataset.tickers,  device)
    test_predictions = predict_dataset(model, loaders["test"], test_dataset.dates, test_dataset.tickers, device)

    val_ic  = compute_daily_ic(val_predictions)
    test_ic = compute_daily_ic(test_predictions)
    _, portfolio_summary = backtest_long_short(test_predictions)

    return {
        "model":             model,
        "history":           history,
        "val_predictions":   val_predictions,
        "test_predictions":  test_predictions,
        "val_ic":            val_ic,
        "test_ic":           test_ic,
        "val_summary":       summarize_cross_sectional_metrics(val_ic),
        "test_summary":      summarize_cross_sectional_metrics(test_ic),
        "portfolio_summary": portfolio_summary,
    }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LSTM (attention), GRU (attention), and Transformer."
    )
    parser.add_argument("--start",          default="2015-01-01")
    parser.add_argument("--end",            default="2024-01-01")
    parser.add_argument("--lookback",       type=int,   default=60)
    parser.add_argument("--horizon",        type=int,   default=1)
    parser.add_argument("--train-size",     type=float, default=0.7)
    parser.add_argument("--val-size",       type=float, default=0.15)
    parser.add_argument("--rolling-zscore", action="store_true")
    parser.add_argument("--rolling-window", type=int,   default=60)
    parser.add_argument(
        "--profile",
        choices=["default", "sp500_fast"],
        default="default",
        help="Training hyperparameter profile. 'sp500_fast' is tuned for broad-universe GPU runs.",
    )
    parser.add_argument(
        "--universe",
        choices=["small", "nasdaq100", "sp500", "auto"],
        default="small",
        help="Stock universe: 'small' (50 stocks), 'nasdaq100', 'sp500' (full list), 'auto' (scrape Wikipedia).",
    )
    return parser.parse_args()


def select_train_configs(profile: str, universe: str) -> tuple[TrainConfig, TrainConfig, TrainConfig]:
    _ = universe
    if profile == "sp500_fast":
        return (
            TrainConfig(
                batch_size=256,
                hidden_dim=64,
                num_layers=2,
                dropout=0.1,
                learning_rate=2e-4,
                weight_decay=1e-5,
                max_epochs=12,
                patience=4,
                log_interval=400,
            ),
            TrainConfig(
                batch_size=512,
                hidden_dim=96,
                num_layers=2,
                dropout=0.1,
                learning_rate=6e-4,
                weight_decay=1e-5,
                max_epochs=12,
                patience=4,
                log_interval=400,
            ),
            TrainConfig(
                batch_size=512,
                hidden_dim=96,
                num_layers=2,
                dropout=0.15,
                num_heads=4,
                learning_rate=2e-4,
                weight_decay=5e-5,
                max_epochs=18,
                patience=5,
                log_interval=400,
            ),
        )

    return (
        TrainConfig(
            batch_size=128,
            hidden_dim=64,
            num_layers=2,
            dropout=0.1,
            learning_rate=1e-4,
            weight_decay=1e-5,
            max_epochs=20,
            patience=5,
        ),
        TrainConfig(
            batch_size=128,
            hidden_dim=64,
            num_layers=2,
            dropout=0.1,
            learning_rate=5e-4,
            weight_decay=1e-5,
            max_epochs=20,
            patience=5,
        ),
        TrainConfig(
            batch_size=256,
            hidden_dim=128,
            num_layers=3,
            dropout=0.2,
            num_heads=8,
            learning_rate=3e-4,
            weight_decay=1e-4,
            max_epochs=30,
            patience=8,
        ),
    )


def print_data_summary(experiment_data: ExperimentData) -> None:
    print("Data summary:", flush=True)
    for split_name in ("train", "val", "test"):
        X, y, dates, tickers = experiment_data.panels[split_name]
        dataset = experiment_data.datasets[split_name]
        print(
            f"  {split_name}: panel={X.shape}, target={y.shape}, "
            f"dates={len(dates)}, tickers={len(tickers)}, samples={len(dataset)}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ #
    # Universe                                                             #
    # ------------------------------------------------------------------ #
    tickers = UNIVERSES[args.universe]
    n_label = len(tickers) if tickers is not None else "Wikipedia scrape"
    print(f"Universe: {args.universe} ({n_label})", flush=True)

    # ------------------------------------------------------------------ #
    # Per-model train configs                                              #
    # ------------------------------------------------------------------ #

    config_lstm, config_gru, config_transformer = select_train_configs(
        profile=args.profile,
        universe=args.universe,
    )
    print(f"Profile: {args.profile}", flush=True)

    # ------------------------------------------------------------------ #
    # Data                                                                 #
    # ------------------------------------------------------------------ #
    print("Downloading price history...", flush=True)
    raw_prices = download_price_history(
        start=args.start,
        end=args.end,
        tickers=tickers,        # ← passes the selected universe
    )
    n_stocks = raw_prices.columns.get_level_values(0).nunique()
    print(f"Stocks downloaded: {n_stocks}", flush=True)

    print("Preparing experiment data...", flush=True)
    experiment_data = prepare_experiment_data(
        raw_prices,
        horizon=args.horizon,
        lookback=args.lookback,
        train_size=args.train_size,
        val_size=args.val_size,
        apply_rolling_zscore=args.rolling_zscore,
        rolling_window=args.rolling_window,
    )
    print_data_summary(experiment_data)

    input_dim = len(experiment_data.feature_columns)

    # ------------------------------------------------------------------ #
    # Build models                                                         #
    # ------------------------------------------------------------------ #

    model_lstm = AttentionSequenceRegressor(
        input_dim=input_dim, hidden_dim=config_lstm.hidden_dim,
        num_layers=config_lstm.num_layers, dropout=config_lstm.dropout,
        cell_type="LSTM",
    )
    model_gru = AttentionSequenceRegressor(
        input_dim=input_dim, hidden_dim=config_gru.hidden_dim,
        num_layers=config_gru.num_layers, dropout=config_gru.dropout,
        cell_type="GRU",
    )
    model_transformer = build_model(
        model_name="TRANSFORMER", input_dim=input_dim,
        hidden_dim=config_transformer.hidden_dim,
        num_layers=config_transformer.num_layers,
        dropout=config_transformer.dropout,
        num_heads=config_transformer.num_heads,
    )

    # ------------------------------------------------------------------ #
    # Train                                                                #
    # ------------------------------------------------------------------ #

    results = {}
    results["LSTM"]        = run_experiment(model_lstm,        "LSTM",        experiment_data, config_lstm)
    results["GRU"]         = run_experiment(model_gru,         "GRU",         experiment_data, config_gru)
    results["TRANSFORMER"] = run_experiment(model_transformer, "TRANSFORMER", experiment_data, config_transformer)

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #

    summary_rows = []
    for model_name, result in results.items():
        summary_rows.append({
            "model": model_name,
            **result["test_summary"],
            **result["portfolio_summary"],
        })

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values("mean_ic", ascending=False)
        .reset_index(drop=True)
    )
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print("\n=== FINAL RESULTS ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
