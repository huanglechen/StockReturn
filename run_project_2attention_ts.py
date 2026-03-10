"""
Run cross-sectional stock return prediction experiments.

Models trained in one call:
  - TRANSFORMER  : improved hyperparameters, no attention module
  - LSTM         : with temporal attention (train_config1)
  - GRU          : with temporal attention (train_config2)
"""

from __future__ import annotations

import argparse
import math

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
    SP500_TICKERS
)


# ---------------------------------------------------------------------------
# Temporal attention module
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """
    Additive (Bahdanau-style) attention over RNN hidden states.
    Returns a context vector (weighted sum of all timestep outputs)
    and the attention weights.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, encoder_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # encoder_outputs: (batch, seq_len, hidden_dim)
        scores = self.attn(encoder_outputs).squeeze(-1)           # (batch, seq_len)
        weights = torch.softmax(scores, dim=-1)                   # (batch, seq_len)
        context = (weights.unsqueeze(-1) * encoder_outputs).sum(dim=1)  # (batch, hidden_dim)
        return context, weights


# ---------------------------------------------------------------------------
# Attention-augmented LSTM / GRU regressor
# ---------------------------------------------------------------------------

class AttentionSequenceRegressor(nn.Module):
    """
    LSTM or GRU encoder with temporal attention.
    Prediction is made from the attention context vector, not just
    the last hidden state, so the model can focus on any past timestep.
    """

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
            raise ValueError(
                f"AttentionSequenceRegressor only supports LSTM/GRU, got {cell_type}"
            )

        rnn_cls = nn.LSTM if cell_type == "LSTM" else nn.GRU
        self.encoder = rnn_cls(
            input_dim,
            hidden_dim,
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
        self.last_attn_weights: torch.Tensor | None = None  # for inspection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_outputs, _ = self.encoder(x)          # (batch, seq_len, hidden_dim)
        context, weights = self.attention(encoder_outputs)
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
        experiment_data.datasets, batch_size=train_config.batch_size
    )
    model, history = train_model(model, loaders["train"], loaders["val"], train_config)

    device = train_config.device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    val_dataset  = experiment_data.datasets["val"]
    test_dataset = experiment_data.datasets["test"]

    val_predictions = predict_dataset(
        model, loaders["val"], val_dataset.dates, val_dataset.tickers, device
    )
    test_predictions = predict_dataset(
        model, loaders["test"], test_dataset.dates, test_dataset.tickers, device
    )

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
        "--universe",
        choices=["small", "sp500", "auto"],
        default="small",
        help="Stock universe to use: 'small' (50), 'sp500' (full list), 'auto' (scrape Wikipedia).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    tickers = UNIVERSES[args.universe]
    print(f"Universe: {args.universe} ({len(tickers) if tickers else 'Wikipedia scrape'})", flush=True)

    # ------------------------------------------------------------------ #
    # Per-model train configs                                              #
    # ------------------------------------------------------------------ #

    # LSTM  — train_config1 (your screenshot)
    config_lstm = TrainConfig(
        batch_size=128,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        num_heads=4,        # not used by LSTM
        learning_rate=1e-4,
        weight_decay=1e-5,
        max_epochs=20,
        patience=5,
    )

    # GRU   — train_config2 (your screenshot)
    config_gru = TrainConfig(
        batch_size=128,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        num_heads=4,        # not used by GRU
        learning_rate=5e-4,
        weight_decay=1e-5,
        max_epochs=20,
        patience=5,
    )

    # TRANSFORMER — improved hyperparameters
    config_transformer = TrainConfig(
        batch_size=256,
        hidden_dim=128,     # was 64
        num_layers=3,       # was 2
        dropout=0.2,        # was 0.1
        num_heads=8,        # was 4  (128 / 8 = 16 ✓)
        learning_rate=3e-4, # lower LR suits transformers
        weight_decay=1e-4,
        max_epochs=30,
        patience=8,
    )

    # ------------------------------------------------------------------ #
    # Data                                                                 #
    # ------------------------------------------------------------------ #
    print("Downloading price history...", flush=True)

    raw_prices = download_price_history(start=args.start, end=args.end)

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

    input_dim = len(experiment_data.feature_columns)

    # ------------------------------------------------------------------ #
    # Build models                                                         #
    # ------------------------------------------------------------------ #

    model_lstm = AttentionSequenceRegressor(
        input_dim=input_dim,
        hidden_dim=config_lstm.hidden_dim,
        num_layers=config_lstm.num_layers,
        dropout=config_lstm.dropout,
        cell_type="LSTM",
    )

    model_gru = AttentionSequenceRegressor(
        input_dim=input_dim,
        hidden_dim=config_gru.hidden_dim,
        num_layers=config_gru.num_layers,
        dropout=config_gru.dropout,
        cell_type="GRU",
    )

    model_transformer = build_model(
        model_name="TRANSFORMER",
        input_dim=input_dim,
        hidden_dim=config_transformer.hidden_dim,
        num_layers=config_transformer.num_layers,
        dropout=config_transformer.dropout,
        num_heads=config_transformer.num_heads,
    )

    # ------------------------------------------------------------------ #
    # Train all three                                                      #
    # ------------------------------------------------------------------ #

    results = {}
    results["LSTM"]        = run_experiment(model_lstm,        "LSTM",        experiment_data, config_lstm)
    results["GRU"]         = run_experiment(model_gru,         "GRU",         experiment_data, config_gru)
    results["TRANSFORMER"] = run_experiment(model_transformer, "TRANSFORMER", experiment_data, config_transformer)

    # ------------------------------------------------------------------ #
    # Summary table                                                        #
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
