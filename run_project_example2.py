import argparse

import numpy as np
import pandas as pd
import torch

from stock_return_project3 import (
    TrainConfig,
    download_price_history,
    prepare_experiment_data,
    run_model_suite,
    build_model,
    make_data_loaders,
    train_model,
    predict_dataset,
    compute_daily_ic,
    backtest_long_short,
    summarize_cross_sectional_metrics,
    build_summary_frame,
)


# ---------------------------------------------------------------------------
# Attention-aware Transformer (drop-in upgrade, no changes to project file)
# ---------------------------------------------------------------------------
import math
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        positions = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerRegressorWithAttention(nn.Module):
    """
    Transformer regressor that also stores attention weights from the last
    encoder layer for post-hoc interpretability.

    Attention weights shape: (batch, num_heads, seq_len, seq_len)
    Access via model.last_attn_weights after a forward pass.
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.2,
        feedforward_dim: int | None = None,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")

        feedforward_dim = feedforward_dim or model_dim * 4
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.position = PositionalEncoding(model_dim, dropout=dropout)

        # Stack encoder layers manually so we can hook the last one
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=True,   # Pre-LN: more stable training
            )
            for _ in range(num_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, 1),
        )

        self.last_attn_weights: torch.Tensor | None = None  # populated on forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(x)
        hidden = self.position(hidden)

        for i, layer in enumerate(self.encoder_layers):
            if i == len(self.encoder_layers) - 1:
                # Capture attention weights from final layer
                hidden, attn = self._forward_layer_with_attn(layer, hidden)
                self.last_attn_weights = attn.detach()
            else:
                hidden = layer(hidden)

        return self.head(hidden[:, -1]).squeeze(-1)

    @staticmethod
    def _forward_layer_with_attn(
        layer: nn.TransformerEncoderLayer, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one TransformerEncoderLayer and return (output, attn_weights)."""
        # Pre-LN path (norm_first=True)
        src = layer.norm1(x)
        attn_out, attn_weights = layer.self_attn(
            src, src, src, need_weights=True, average_attn_weights=False
        )
        x = x + layer.dropout1(attn_out)
        x = x + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(x))))))
        return x, attn_weights


# ---------------------------------------------------------------------------
# Attention visualisation helper
# ---------------------------------------------------------------------------

def plot_attention_heatmap(
    model: TransformerRegressorWithAttention,
    sample_x: torch.Tensor,
    head_idx: int = 0,
    title: str = "Attention weights (last encoder layer)",
):
    """
    Run one forward pass and plot the attention heatmap for a chosen head.

    Args:
        model:    trained TransformerRegressorWithAttention
        sample_x: tensor of shape (1, seq_len, input_dim)
        head_idx: which attention head to visualise
        title:    plot title
    """
    import matplotlib.pyplot as plt

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        model(sample_x.to(device))

    attn = model.last_attn_weights  # (1, num_heads, T, T)
    if attn is None:
        raise RuntimeError("No attention weights captured. Run a forward pass first.")

    attn_head = attn[0, head_idx].cpu().numpy()  # (T, T)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attn_head, aspect="auto", cmap="viridis", origin="upper")
    ax.set_xlabel("Key timestep")
    ax.set_ylabel("Query timestep")
    ax.set_title(f"{title}\nHead {head_idx}")
    fig.colorbar(im, ax=ax, label="Attention weight")
    fig.tight_layout()
    return fig


def plot_mean_attention_over_time(
    model: TransformerRegressorWithAttention,
    loader,
    device: str,
    n_batches: int = 20,
    title: str = "Mean attention (last timestep → all past) averaged over heads & samples",
):
    """
    Average the attention from the last query position (most recent timestep)
    to all key positions across multiple batches. This shows *which past
    timesteps* the model focuses on when making its prediction.
    """
    import matplotlib.pyplot as plt

    model.eval()
    accumulated = None
    count = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            x = batch["x"].to(device)
            model(x)
            attn = model.last_attn_weights  # (B, H, T, T)
            # last query row: (B, H, T)
            last_row = attn[:, :, -1, :].mean(dim=(0, 1)).cpu().numpy()
            accumulated = last_row if accumulated is None else accumulated + last_row
            count += 1

    if accumulated is None or count == 0:
        raise RuntimeError("No batches processed.")

    mean_attn = accumulated / count
    timesteps = np.arange(len(mean_attn))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(timesteps, mean_attn, color="steelblue", alpha=0.8)
    ax.set_xlabel("Lookback timestep (0 = oldest, T-1 = most recent)")
    ax.set_ylabel("Mean attention weight")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cross-sectional stock return prediction experiments."
    )
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--train-size", type=float, default=0.7)
    parser.add_argument("--val-size", type=float, default=0.15)

    # --- Improved transformer defaults ---
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)   # was 64
    parser.add_argument("--num-layers", type=int, default=3)     # was 2
    parser.add_argument("--dropout", type=float, default=0.2)    # was 0.1
    parser.add_argument("--num-heads", type=int, default=8)      # was 4
    parser.add_argument("--lr", type=float, default=3e-4)        # was 1e-3
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)        # was 20
    parser.add_argument("--patience", type=int, default=8)       # was 5

    parser.add_argument(
        "--models",
        nargs="+",
        default=["TRANSFORMER"],
        help="Subset of models to train.",
    )
    parser.add_argument(
        "--attention-model",
        action="store_true",
        help="Use the attention-aware transformer variant and plot attention maps.",
    )
    parser.add_argument(
        "--rolling-zscore",
        action="store_true",
        help="Apply per-stock rolling standardization after cross-sectional z-score.",
    )
    parser.add_argument("--rolling-window", type=int, default=60)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

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

    train_config = TrainConfig(
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_heads=args.num_heads,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        patience=args.patience,
    )

    # ------------------------------------------------------------------
    # Option A: attention-aware transformer (--attention-model flag)
    # ------------------------------------------------------------------
    if args.attention_model:
        print("=== Training TransformerRegressorWithAttention ===", flush=True)

        input_dim = len(experiment_data.feature_columns)
        model = TransformerRegressorWithAttention(
            input_dim=input_dim,
            model_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
        )

        loaders = make_data_loaders(experiment_data.datasets, batch_size=args.batch_size)
        model, history = train_model(model, loaders["train"], loaders["val"], train_config)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        test_dataset = experiment_data.datasets["test"]
        test_predictions = predict_dataset(
            model, loaders["test"], test_dataset.dates, test_dataset.tickers, device
        )
        test_ic = compute_daily_ic(test_predictions)
        _, portfolio_summary = backtest_long_short(test_predictions)
        test_summary = summarize_cross_sectional_metrics(test_ic)

        print("\n=== ATTENTION TRANSFORMER RESULTS ===")
        summary_df = pd.DataFrame([{"model": "TRANSFORMER_ATTN", **test_summary, **portfolio_summary}])
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 160)
        print(summary_df.to_string(index=False))

        # --- Attention plots ---
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # 1. Single-sample heatmap
            sample = next(iter(loaders["test"]))["x"][:1]
            fig_heatmap = plot_attention_heatmap(model, sample, head_idx=0)
            fig_heatmap.savefig("attention_heatmap.png", dpi=150)
            print("Saved: attention_heatmap.png", flush=True)

            # 2. Mean attention over time
            fig_mean = plot_mean_attention_over_time(model, loaders["test"], device=device)
            fig_mean.savefig("attention_mean_over_time.png", dpi=150)
            print("Saved: attention_mean_over_time.png", flush=True)

            plt.close("all")
        except Exception as exc:
            print(f"Attention plot skipped: {exc}", flush=True)

    # ------------------------------------------------------------------
    # Option B: standard model suite (default)
    # ------------------------------------------------------------------
    else:
        results = run_model_suite(
            experiment_data,
            model_names=args.models,
            train_config=train_config,
        )

        summary = build_summary_frame(results)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 160)
        print("\n=== RESULTS ===")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
