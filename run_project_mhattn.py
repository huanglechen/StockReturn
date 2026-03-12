"""
Run cross-sectional stock return prediction experiments.

Models:
  - LSTM  : with temporal attention (single-head or multi-head, toggled by --attn-heads)
  - GRU   : with temporal attention (single-head or multi-head, toggled by --attn-heads)
  - TRANSFORMER : original, no attention module

Attention modes:
  --attn-heads 1   →  simple additive attention (original single linear layer)
  --attn-heads 4   →  multi-head attention (recommended)

Example runs:
  # Single-head attention (baseline)
  python run_project_mhattn.py --attn-heads 1

  # Multi-head attention (new)
  python run_project_mhattn.py --attn-heads 4

  # Multi-head + full S&P 500
  python run_project_mhattn.py --attn-heads 4 --universe sp500
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
    SP500_TICKERS,
)

# ---------------------------------------------------------------------------
# Universe definitions
# ---------------------------------------------------------------------------

UNIVERSES = {
    "small": DEFAULT_LIQUID_TICKERS,
    "sp500": SP500_TICKERS,
    "auto":  None,
}


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------

class SingleHeadAttention(nn.Module):
    """
    Simple additive attention — scores each timestep independently.
    No query vector; one global importance weight per timestep.
    """
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # enc: (B, T, D)
        weights = torch.softmax(self.attn(enc).squeeze(-1), dim=-1)   # (B, T)
        context = (weights.unsqueeze(-1) * enc).sum(dim=1)            # (B, D)
        return context, weights


class MultiHeadTemporalAttention(nn.Module):
    """
    Multi-head attention where the LAST hidden state acts as the query
    and attends over all timesteps as keys/values.

    - Multiple heads let the model focus on different temporal patterns
      simultaneously (e.g. short-term momentum vs longer-term trend).
    - Using the last hidden state as query gives the attention a
      meaningful signal to look for, unlike the single-head version
      which scores timesteps blindly.
    - Context is concatenated with the last hidden state so no
      information is lost compared to the vanilla RNN.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.W_q  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_out = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = enc.shape
        H, HD   = self.num_heads, self.head_dim

        # Last hidden state as query, all timesteps as keys and values
        query = self.W_q(enc[:, -1:, :])                  # (B, 1, D)
        keys  = self.W_k(enc)                              # (B, T, D)
        vals  = self.W_v(enc)                              # (B, T, D)

        # Reshape to multi-head form
        query = query.view(B, 1, H, HD).transpose(1, 2)   # (B, H, 1, HD)
        keys  = keys.view(B, T, H, HD).transpose(1, 2)    # (B, H, T, HD)
        vals  = vals.view(B, T, H, HD).transpose(1, 2)    # (B, H, T, HD)

        # Scaled dot-product attention from query to all keys
        scores  = (query @ keys.transpose(-2, -1)) * self.scale  # (B, H, 1, T)
        weights = torch.softmax(scores.squeeze(2), dim=-1)        # (B, H, T)

        # Weighted sum of values
        context = (weights.unsqueeze(-1) * vals).sum(dim=2)       # (B, H, HD)
        context = self.W_out(context.view(B, D))                  # (B, D)

        # Average weights across heads for inspection
        return context, weights.mean(dim=1)                       # (B, T)


# ---------------------------------------------------------------------------
# Attention-augmented LSTM / GRU
# ---------------------------------------------------------------------------

class AttentionSequenceRegressor(nn.Module):
    """
    LSTM or GRU with temporal attention.

    When num_attn_heads == 1: uses SingleHeadAttention.
    When num_attn_heads  > 1: uses MultiHeadTemporalAttention.

    In both cases the attention context is CONCATENATED with the last
    hidden state before the prediction head, so no RNN information is lost.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        cell_type: str = "LSTM",
        num_attn_heads: int = 4,
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

        if num_attn_heads == 1:
            self.attention = SingleHeadAttention(hidden_dim)
        else:
            self.attention = MultiHeadTemporalAttention(hidden_dim, num_heads=num_attn_heads)

        # Input is context (D) + last hidden state (D) = 2D
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.last_attn_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_out, hidden = self.encoder(x)            # enc_out: (B, T, D)

        # Extract last hidden state
        if isinstance(hidden, tuple):                # LSTM returns (h, c)
            hidden = hidden[0]
        last_hidden = hidden[-1]                     # (B, D)

        # Attention context
        context, weights = self.attention(enc_out)   # (B, D), (B, T)
        self.last_attn_weights = weights.detach()

        # Concatenate and predict
        combined = torch.cat([last_hidden, context], dim=-1)  # (B, 2D)
        return self.head(combined).squeeze(-1)


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

    loaders = make_data_loaders(experiment_data.datasets, batch_size=train_config.batch_size)
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
        description="Train LSTM/GRU with attention and Transformer."
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
        help="Stock universe: 'small' (50), 'sp500' (full list), 'auto' (Wikipedia).",
    )
    parser.add_argument(
        "--attn-heads",
        type=int,
        default=4,
        help="Number of attention heads for LSTM/GRU. 1 = simple additive, 4+ = multi-head (recommended).",
    )
    return parser.parse_args()


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
    print(f"Universe  : {args.universe} ({n_label} tickers)", flush=True)
    print(f"Attn heads: {args.attn_heads} ({'multi-head' if args.attn_heads > 1 else 'single-head'})", flush=True)

    # ------------------------------------------------------------------ #
    # Per-model configs                                                    #
    # With the new 2D concatenated input to the head, hidden_dim=64 is    #
    # still fine — the head now sees 128-dim input automatically.         #
    # ------------------------------------------------------------------ #

    config_lstm = TrainConfig(
        batch_size=128,
        hidden_dim=64,
        num_layers=2,
        dropout=0.15,
        learning_rate=1e-4,   # slightly higher than before — concatenation helps gradient flow
        weight_decay=1e-5,
        max_epochs=25,
        patience=6,
        grad_clip=1.0,
    )
    config_gru = TrainConfig(
        batch_size=128,
        hidden_dim=64,
        num_layers=2,
        dropout=0.15,
        learning_rate=1e-4,
        weight_decay=1e-5,
        max_epochs=25,
        patience=6,
        grad_clip=1.0,
    )
    config_transformer = TrainConfig(
        batch_size=256,
        hidden_dim=128,
        num_layers=3,
        dropout=0.2,
        num_heads=8,
        learning_rate=3e-4,
        weight_decay=1e-4,
        max_epochs=30,
        patience=8,
        grad_clip=1.0,
    )

    # ------------------------------------------------------------------ #
    # Data                                                                 #
    # ------------------------------------------------------------------ #
    print("\nDownloading price history...", flush=True)
    raw_prices = download_price_history(start=args.start, end=args.end, tickers=tickers)
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
        num_attn_heads=args.attn_heads,
    )
    model_gru = AttentionSequenceRegressor(
        input_dim=input_dim,
        hidden_dim=config_gru.hidden_dim,
        num_layers=config_gru.num_layers,
        dropout=config_gru.dropout,
        cell_type="GRU",
        num_attn_heads=args.attn_heads,
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
