import argparse

import pandas as pd

from stock_return_core import (
    TrainConfig,
    download_price_history,
    prepare_experiment_data,
    run_model_suite,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the cross-sectional stock return prediction experiments."
    )
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--train-size", type=float, default=0.7)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["RNN", "LSTM", "GRU", "TRANSFORMER"],
        help="Subset of models to train.",
    )
    parser.add_argument(
        "--rolling-zscore",
        action="store_true",
        help="Apply per-stock rolling standardization after cross-sectional z-score.",
    )
    parser.add_argument("--rolling-window", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_prices = download_price_history(start=args.start, end=args.end)
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
        max_epochs=args.epochs,
        patience=args.patience,
    )
    results = run_model_suite(
        experiment_data,
        model_names=args.models,
        train_config=train_config,
    )

    summary_rows = []
    for model_name, result in results.items():
        summary_rows.append(
            {
                "model": model_name,
                **result["test_summary"],
                **result["portfolio_summary"],
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("mean_ic", ascending=False)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
