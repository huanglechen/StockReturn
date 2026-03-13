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
    flatten_price_frame,
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


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def filter_by_history_coverage(
    raw_prices: pd.DataFrame,
    min_coverage_ratio: float = 0.95,
    start_buffer_days: int = 5,
    end_buffer_days: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    flat_prices = flatten_price_frame(raw_prices).dropna(subset=["Close"]).copy()
    all_dates = pd.Index(sorted(pd.to_datetime(flat_prices["Date"].unique())))
    date_to_pos = {date: idx for idx, date in enumerate(all_dates)}

    coverage = (
        flat_prices.groupby("Ticker")
        .agg(
            first_date=("Date", "min"),
            last_date=("Date", "max"),
            n_dates=("Date", "nunique"),
        )
        .reset_index()
    )
    coverage["first_pos"] = pd.to_datetime(coverage["first_date"]).map(date_to_pos)
    coverage["last_pos"] = pd.to_datetime(coverage["last_date"]).map(date_to_pos)
    coverage["coverage_ratio"] = coverage["n_dates"] / len(all_dates)

    eligible_tickers = coverage.loc[
        (coverage["first_pos"] <= start_buffer_days)
        & (coverage["last_pos"] >= len(all_dates) - 1 - end_buffer_days)
        & (coverage["coverage_ratio"] >= min_coverage_ratio),
        "Ticker",
    ].tolist()

    filtered_prices = raw_prices.loc[
        :,
        raw_prices.columns.get_level_values(0).isin(eligible_tickers),
    ].copy()
    return filtered_prices, coverage


def summarize_prediction_diagnostics(predictions: pd.DataFrame) -> dict[str, float]:
    if predictions.empty:
        return {
            "rows": 0.0,
            "dates": 0.0,
            "prediction_std": float("nan"),
            "median_cross_sectional_std": float("nan"),
            "median_prediction_nunique": float("nan"),
            "fraction_constant_dates": float("nan"),
            "mse": float("nan"),
            "zero_baseline_mse": float("nan"),
        }

    per_date = predictions.groupby("Date")["prediction"].agg(["std", "nunique"])
    errors = (predictions["prediction"] - predictions["target"]) ** 2
    return {
        "rows": float(len(predictions)),
        "dates": float(predictions["Date"].nunique()),
        "prediction_std": float(predictions["prediction"].std(ddof=1)),
        "median_cross_sectional_std": float(per_date["std"].fillna(0.0).median()),
        "median_prediction_nunique": float(per_date["nunique"].median()),
        "fraction_constant_dates": float((per_date["nunique"] < 2).mean()),
        "mse": float(errors.mean()),
        "zero_baseline_mse": float((predictions["target"] ** 2).mean()),
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
    val_diagnostics = summarize_prediction_diagnostics(val_predictions)
    test_diagnostics = summarize_prediction_diagnostics(test_predictions)

    if test_diagnostics["fraction_constant_dates"] == 1.0:
        print(
            f"Warning: {model_name} predictions are constant within every test date; "
            "IC will be NaN and long-short returns will collapse to 0.",
            flush=True,
        )
    if pd.notna(val_diagnostics["mse"]) and pd.notna(val_diagnostics["zero_baseline_mse"]):
        improvement = val_diagnostics["zero_baseline_mse"] - val_diagnostics["mse"]
        if improvement <= 1e-6:
            print(
                f"Warning: {model_name} validation MSE ({val_diagnostics['mse']:.6f}) is "
                f"indistinguishable from the zero-prediction baseline "
                f"({val_diagnostics['zero_baseline_mse']:.6f}); the model likely collapsed "
                "to the unconditional mean.",
                flush=True,
            )

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
        "val_diagnostics":   val_diagnostics,
        "test_diagnostics":  test_diagnostics,
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
        "--profile",
        choices=["default", "sp500_stable"],
        default="default",
        help="Training hyperparameter profile. 'sp500_stable' is tuned to avoid collapsed RNN predictions on the broad universe.",
    )
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
    parser.add_argument(
        "--clean-history",
        action="store_true",
        help="Filter tickers to those with near-full sample history before training.",
    )
    parser.add_argument("--min-coverage-ratio", type=float, default=0.95)
    parser.add_argument("--start-buffer-days", type=int, default=5)
    parser.add_argument("--end-buffer-days", type=int, default=5)
    parser.add_argument(
        "--target-cs-zscore",
        action="store_true",
        help="Train on per-date cross-sectionally standardized targets while keeping raw returns for evaluation/backtests.",
    )
    return parser.parse_args()


def select_train_configs(profile: str, device: str) -> tuple[TrainConfig, TrainConfig, TrainConfig]:
    if profile == "sp500_stable":
        return (
            TrainConfig(
                batch_size=256,
                hidden_dim=96,
                num_layers=2,
                dropout=0.05,
                learning_rate=3e-4,
                weight_decay=1e-5,
                max_epochs=20,
                patience=8,
                grad_clip=1.0,
                device=device,
                log_interval=400,
            ),
            TrainConfig(
                batch_size=256,
                hidden_dim=96,
                num_layers=2,
                dropout=0.05,
                learning_rate=5e-4,
                weight_decay=1e-5,
                max_epochs=20,
                patience=8,
                grad_clip=1.0,
                device=device,
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
                grad_clip=1.0,
                device=device,
                log_interval=400,
            ),
        )

    return (
        TrainConfig(
            batch_size=128,
            hidden_dim=64,
            num_layers=2,
            dropout=0.15,
            learning_rate=1e-4,
            weight_decay=1e-5,
            max_epochs=25,
            patience=6,
            grad_clip=1.0,
            device=device,
        ),
        TrainConfig(
            batch_size=128,
            hidden_dim=64,
            num_layers=2,
            dropout=0.15,
            learning_rate=1e-4,
            weight_decay=1e-5,
            max_epochs=25,
            patience=6,
            grad_clip=1.0,
            device=device,
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
            grad_clip=1.0,
            device=device,
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = resolve_device()

    # ------------------------------------------------------------------ #
    # Universe                                                             #
    # ------------------------------------------------------------------ #
    tickers = UNIVERSES[args.universe]
    n_label = len(tickers) if tickers is not None else "Wikipedia scrape"
    config_lstm, config_gru, config_transformer = select_train_configs(args.profile, device)
    print(f"Universe  : {args.universe} ({n_label} tickers)", flush=True)
    print(f"Profile   : {args.profile}", flush=True)
    print(f"Attn heads: {args.attn_heads} ({'multi-head' if args.attn_heads > 1 else 'single-head'})", flush=True)
    print(f"Target    : {'cross-sectional z-score' if args.target_cs_zscore else 'raw return'}", flush=True)
    print(f"Device    : {device}", flush=True)
    if args.profile == "sp500_stable" and args.attn_heads > 1:
        print(
            "Warning: sp500_stable is most reliable with --attn-heads 1; "
            "multi-head attention may still collapse on the broad universe.",
            flush=True,
        )

    # ------------------------------------------------------------------ #
    # Data                                                                 #
    # ------------------------------------------------------------------ #
    print("\nDownloading price history...", flush=True)
    raw_prices = download_price_history(start=args.start, end=args.end, tickers=tickers)
    n_stocks = raw_prices.columns.get_level_values(0).nunique()
    print(f"Stocks downloaded: {n_stocks}", flush=True)

    if args.clean_history:
        raw_prices, coverage = filter_by_history_coverage(
            raw_prices,
            min_coverage_ratio=args.min_coverage_ratio,
            start_buffer_days=args.start_buffer_days,
            end_buffer_days=args.end_buffer_days,
        )
        kept_tickers = raw_prices.columns.get_level_values(0).nunique()
        dropped = coverage.loc[
            ~coverage["Ticker"].isin(raw_prices.columns.get_level_values(0)),
            "Ticker",
        ].head(10).tolist()
        print(
            f"History filter kept {kept_tickers} / {coverage.shape[0]} tickers "
            f"(coverage >= {args.min_coverage_ratio:.0%}, "
            f"start <= {args.start_buffer_days}d, end <= {args.end_buffer_days}d).",
            flush=True,
        )
        print(f"Dropped examples: {dropped}", flush=True)

    print("Preparing experiment data...", flush=True)
    experiment_data = prepare_experiment_data(
        raw_prices,
        horizon=args.horizon,
        lookback=args.lookback,
        train_size=args.train_size,
        val_size=args.val_size,
        apply_rolling_zscore=args.rolling_zscore,
        rolling_window=args.rolling_window,
        target_cross_sectional_zscore=args.target_cs_zscore,
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
