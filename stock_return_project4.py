from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import OneCycleLR

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - only hit when yfinance is unavailable.
    yf = None


DEFAULT_FEATURE_COLUMNS = [
    "log_return_1d",
    "volatility_20d",
    "sma_10_gap",
    "sma_20_gap",
    "ema_20_gap",
    "macd",
    "macd_signal",
    "high_low_spread",
    "open_close_gap",
    "log_volume",
    "rel_volume",
    "excess_return",
]

# Fallback universe used when scraping the live S&P 500 membership fails.
# This still satisfies the proposal's "liquid subset of U.S. equities" option.
DEFAULT_LIQUID_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK.B", "JPM", "LLY", "AVGO",
    "V", "XOM", "UNH", "MA", "COST", "ORCL", "NFLX", "HD", "PG", "JNJ",
    "ABBV", "BAC", "KO", "CRM", "WMT", "AMD", "MRK", "CVX", "PEP", "TMO",
    "LIN", "ADBE", "MCD", "DIS", "CSCO", "ABT", "GE", "VZ", "DHR", "TXN",
    "CMCSA", "AMGN", "INTU", "QCOM", "CAT", "PFE", "IBM", "NOW", "PM", "UBER",
]


@dataclass
class ExperimentData:
    cleaned_frame: pd.DataFrame
    featured_frame: pd.DataFrame
    splits: dict[str, pd.DataFrame]
    panels: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    datasets: dict[str, "PanelSequenceDataset"]
    feature_columns: list[str]
    target_column: str


@dataclass
class TrainConfig:
    batch_size: int = 256
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    num_heads: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 25
    patience: int = 5
    grad_clip: float | None = 1.0
    device: str | None = None
    verbose: bool = True
    log_interval: int = 200


def load_sp500_constituents() -> pd.DataFrame:
    try:
        sp500 = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            storage_options={"User-Agent": "Mozilla/5.0"},
        )[0]
        sp500["YahooSymbol"] = sp500["Symbol"].str.replace(".", "-", regex=False)
        return sp500
    except Exception:
        fallback = pd.DataFrame({"Symbol": DEFAULT_LIQUID_TICKERS})
        fallback["YahooSymbol"] = fallback["Symbol"].str.replace(".", "-", regex=False)
        return fallback


def download_price_history(
    start: str = "2015-01-01",
    end: str = "2024-01-01",
    tickers: list[str] | None = None,
    auto_adjust: bool = True,
    cache_path: str | Path | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance is required to download market data.")

    if tickers is None:
        sp500 = load_sp500_constituents()
        original_tickers = sp500["Symbol"].tolist()
        yahoo_tickers = sp500["YahooSymbol"].tolist()
    else:
        original_tickers = list(tickers)
        yahoo_tickers = [ticker.replace(".", "-") for ticker in tickers]

    if cache_path is None:
        cache_dir = Path("data_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        universe_name = "sp500" if tickers is None else f"subset_{len(original_tickers)}"
        cache_path = cache_dir / f"{universe_name}_{start}_{end}_autoadj{int(auto_adjust)}.pkl"
    else:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

    if use_cache and cache_path.exists():
        cached = pd.read_pickle(cache_path)
        if isinstance(cached, pd.DataFrame):
            return cached

    raw = yf.download(
        yahoo_tickers,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=auto_adjust,
        threads=True,
        progress=False,
    )

    if not isinstance(raw.columns, pd.MultiIndex):
        raise ValueError("Expected yfinance to return a MultiIndex price frame.")

    if raw.columns.names == ["Price", "Ticker"]:
        raw = raw.swaplevel(axis=1).sort_index(axis=1)

    inverse_map = dict(zip(yahoo_tickers, original_tickers))
    raw.columns = pd.MultiIndex.from_arrays(
        [
            [inverse_map.get(symbol, symbol) for symbol in raw.columns.get_level_values(0)],
            raw.columns.get_level_values(1),
        ],
        names=raw.columns.names,
    )
    raw = raw.sort_index(axis=1)
    if use_cache:
        raw.to_pickle(cache_path)
    return raw


def flatten_price_frame(raw_prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(raw_prices.columns, pd.MultiIndex):
        raise ValueError("Expected raw_prices to use a 2-level column MultiIndex.")

    try:
        stacked = raw_prices.stack(level=0, future_stack=True)
    except TypeError:
        stacked = raw_prices.stack(level=0)

    flat = stacked.reset_index().rename_axis(columns=None)
    index_columns = list(flat.columns[:2])
    flat = flat.rename(columns={index_columns[0]: "Date", index_columns[1]: "Ticker"})
    if "Adj Close" in flat.columns and flat["Adj Close"].isna().all():
        flat = flat.drop(columns=["Adj Close"])
    flat = flat.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return flat


def trim_and_fill_history(
    prices: pd.DataFrame,
    required_cols: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume"),
    stock_col: str = "Ticker",
    date_col: str = "Date",
) -> pd.DataFrame:
    missing = [col for col in [stock_col, date_col, *required_cols] if col not in prices.columns]
    if missing:
        raise KeyError(f"Missing required columns {missing}. Available columns: {list(prices.columns)}")

    prices = prices.sort_values([stock_col, date_col]).copy()
    required = list(required_cols)
    cleaned_groups: list[pd.DataFrame] = []

    def _trim_one_stock(group: pd.DataFrame) -> pd.DataFrame:
        valid_mask = group["Close"].notna().to_numpy()
        if not valid_mask.any():
            return group.iloc[0:0]

        first_valid = int(np.flatnonzero(valid_mask)[0])
        last_valid = int(np.flatnonzero(valid_mask)[-1])
        trimmed = group.iloc[first_valid : last_valid + 1].copy()
        trimmed[required] = trimmed[required].ffill()
        trimmed = trimmed.dropna(subset=required)
        return trimmed

    for _, group in prices.groupby(stock_col, sort=False):
        trimmed = _trim_one_stock(group)
        if not trimmed.empty:
            cleaned_groups.append(trimmed)

    if not cleaned_groups:
        return prices.iloc[0:0].copy()

    cleaned = pd.concat(cleaned_groups, ignore_index=True)
    cleaned = cleaned.sort_values([stock_col, date_col]).reset_index(drop=True)
    return cleaned


def add_technical_features(
    prices: pd.DataFrame,
    horizon: int = 1,
    stock_col: str = "Ticker",
    date_col: str = "Date",
) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("horizon must be at least 1 trading day.")

    prices = prices.sort_values([stock_col, date_col]).copy()
    grouped = prices.groupby(stock_col, group_keys=False)

    prices["log_return_1d"] = grouped["Close"].transform(lambda s: np.log(s).diff())
    prices["volatility_20d"] = grouped["log_return_1d"].transform(
        lambda s: s.rolling(window=20, min_periods=20).std()
    )
    prices["sma_10_gap"] = grouped["Close"].transform(
        lambda s: s / s.rolling(window=10, min_periods=10).mean() - 1.0
    )
    prices["sma_20_gap"] = grouped["Close"].transform(
        lambda s: s / s.rolling(window=20, min_periods=20).mean() - 1.0
    )
    prices["ema_20_gap"] = grouped["Close"].transform(
        lambda s: s / s.ewm(span=20, adjust=False).mean() - 1.0
    )
    prices["macd"] = grouped["Close"].transform(
        lambda s: s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    )
    prices["macd_signal"] = prices.groupby(stock_col, group_keys=False)["macd"].transform(
        lambda s: s.ewm(span=9, adjust=False).mean()
    )
    prices["high_low_spread"] = (prices["High"] - prices["Low"]) / prices["Close"]
    prices["open_close_gap"] = (prices["Open"] - prices["Close"]) / prices["Close"]
    prices["log_volume"] = np.log1p(prices["Volume"])

    # Relative volume: today's volume vs its own 20-day rolling average.
    # Uses raw Volume (before log) so the ratio is meaningful.
    # min_periods=1 avoids NaN at the start of each stock's history.
    prices["rel_volume"] = grouped["Volume"].transform(
        lambda s: s / s.rolling(window=20, min_periods=1).mean()
    )

    # Excess return: stock's 1-day log return minus the cross-sectional mean
    # on the same date. Encodes relative momentum vs peers — directly aligned
    # with what IC measures. Computed after log_return_1d is already in place.
    market_return = prices.groupby(date_col, group_keys=False)["log_return_1d"].transform("mean")
    prices["excess_return"] = prices["log_return_1d"] - market_return

    target_col = f"target_return_{horizon}d"
    prices[target_col] = grouped["Close"].transform(
        lambda s: s.shift(-horizon) / s - 1.0
    )
    return prices


def winsorize_cross_section(
    frame: pd.DataFrame,
    feature_cols: list[str],
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    date_col: str = "Date",
) -> pd.DataFrame:
    clipped = frame.copy()
    for feature in feature_cols:
        grouped = clipped.groupby(date_col)[feature]
        lower = grouped.transform(lambda s: s.quantile(lower_quantile))
        upper = grouped.transform(lambda s: s.quantile(upper_quantile))
        clipped[feature] = clipped[feature].clip(lower=lower, upper=upper)
    return clipped


def cross_sectional_standardize(
    frame: pd.DataFrame,
    feature_cols: list[str],
    date_col: str = "Date",
    eps: float = 1e-8,
) -> pd.DataFrame:
    standardized = frame.copy()
    grouped = standardized.groupby(date_col)[feature_cols]
    means = grouped.transform("mean")
    stds = grouped.transform("std")
    safe_stds = stds.mask(stds.abs() <= eps, 1.0).fillna(1.0)
    standardized[feature_cols] = (standardized[feature_cols] - means) / safe_stds
    return standardized


def rolling_standardize(
    frame: pd.DataFrame,
    feature_cols: list[str],
    stock_col: str = "Ticker",
    date_col: str = "Date",
    window: int = 60,
    min_periods: int | None = None,
    eps: float = 1e-8,
) -> pd.DataFrame:
    min_periods = min_periods or window
    standardized = frame.sort_values([stock_col, date_col]).copy()
    grouped = standardized.groupby(stock_col, group_keys=False)

    for feature in feature_cols:
        rolling_mean = grouped[feature].transform(
            lambda s: s.rolling(window=window, min_periods=min_periods).mean()
        )
        rolling_std = grouped[feature].transform(
            lambda s: s.rolling(window=window, min_periods=min_periods).std()
        )
        safe_std = rolling_std.mask(rolling_std.abs() <= eps, 1.0).fillna(1.0)
        standardized[feature] = (standardized[feature] - rolling_mean) / safe_std

    return standardized


def split_by_time(
    frame: pd.DataFrame,
    date_col: str = "Date",
    train_size: float = 0.7,
    val_size: float = 0.15,
) -> dict[str, pd.DataFrame]:
    if not 0 < train_size < 1:
        raise ValueError("train_size must lie in (0, 1).")
    if not 0 <= val_size < 1:
        raise ValueError("val_size must lie in [0, 1).")
    if train_size + val_size >= 1:
        raise ValueError("train_size + val_size must be strictly less than 1.")

    unique_dates = np.sort(pd.to_datetime(frame[date_col].unique()))
    if len(unique_dates) < 3:
        raise ValueError("Need at least 3 unique dates to form train/val/test splits.")

    n_dates = len(unique_dates)
    train_end = min(max(1, int(n_dates * train_size)), n_dates - 2)
    val_end = min(max(train_end + 1, int(n_dates * (train_size + val_size))), n_dates - 1)

    train_dates = unique_dates[:train_end]
    val_dates = unique_dates[train_end:val_end]
    test_dates = unique_dates[val_end:]

    return {
        "train": frame[frame[date_col].isin(train_dates)].copy(),
        "val": frame[frame[date_col].isin(val_dates)].copy(),
        "test": frame[frame[date_col].isin(test_dates)].copy(),
    }


def build_panel_tensors(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    date_col: str = "Date",
    stock_col: str = "Ticker",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_panel = (
        frame.pivot(index=date_col, columns=stock_col, values=target_col).sort_index()
    )
    dates = pd.to_datetime(target_panel.index).to_numpy()
    tickers = target_panel.columns.to_numpy()
    features = []

    for feature in feature_cols:
        panel = (
            frame.pivot(index=date_col, columns=stock_col, values=feature)
            .reindex(index=target_panel.index, columns=target_panel.columns)
            .to_numpy(dtype=np.float32)
        )
        features.append(panel)

    X = np.stack(features, axis=-1).astype(np.float32)
    y = target_panel.to_numpy(dtype=np.float32)
    return X, y, dates, tickers


class PanelSequenceDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lookback: int = 60,
        dates: np.ndarray | None = None,
        tickers: np.ndarray | None = None,
    ) -> None:
        if lookback < 1:
            raise ValueError("lookback must be positive.")

        self.X = X
        self.y = y
        self.lookback = lookback
        self.dates = np.arange(len(X)) if dates is None else dates
        self.tickers = np.arange(X.shape[1]) if tickers is None else tickers
        self.valid_pairs: list[tuple[int, int]] = []

        for stock_idx in range(X.shape[1]):
            for time_idx in range(lookback, X.shape[0]):
                target = y[time_idx, stock_idx]
                window = X[time_idx - lookback : time_idx, stock_idx]
                if np.isnan(target) or np.isnan(window).any():
                    continue
                self.valid_pairs.append((time_idx, stock_idx))

        if not self.valid_pairs:
            raise ValueError("No valid samples found. Check preprocessing or lookback length.")

    def __len__(self) -> int:
        return len(self.valid_pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        time_idx, stock_idx = self.valid_pairs[index]
        return {
            "x": torch.from_numpy(self.X[time_idx - self.lookback : time_idx, stock_idx]).float(),
            "y": torch.tensor(self.y[time_idx, stock_idx]).float(),
            "time_idx": torch.tensor(time_idx, dtype=torch.long),
            "stock_idx": torch.tensor(stock_idx, dtype=torch.long),
        }


class SequenceRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        cell_type: str = "RNN",
    ) -> None:
        super().__init__()
        rnn_map = {
            "RNN": nn.RNN,
            "LSTM": nn.LSTM,
            "GRU": nn.GRU,
        }
        if cell_type not in rnn_map:
            raise ValueError(f"Unsupported recurrent cell type: {cell_type}")

        rnn_cls = rnn_map[cell_type]
        self.encoder = rnn_cls(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, hidden = self.encoder(x)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        last_hidden = hidden[-1]
        return self.head(last_hidden).squeeze(-1)


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
        if x.size(1) > self.pe.size(1):
            raise ValueError("Sequence length exceeds positional encoding capacity.")
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        feedforward_dim: int | None = None,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")

        feedforward_dim = feedforward_dim or model_dim * 4
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.position = PositionalEncoding(model_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(x)
        hidden = self.position(hidden)
        hidden = self.encoder(hidden)
        return self.head(hidden[:, -1]).squeeze(-1)


def build_model(
    model_name: str,
    input_dim: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    num_heads: int = 4,
) -> nn.Module:
    model_name = model_name.upper()
    if model_name in {"RNN", "LSTM", "GRU"}:
        return SequenceRegressor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            cell_type=model_name,
        )
    if model_name == "TRANSFORMER":
        return TransformerRegressor(
            input_dim=input_dim,
            model_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def make_data_loaders(
    datasets: dict[str, PanelSequenceDataset],
    batch_size: int = 256,
) -> dict[str, DataLoader]:
    return {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False),
    }


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float | None = None,
    split_name: str = "train",
    verbose: bool = False,
    log_interval: int = 200,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_count = 0
    num_batches = max(len(loader), 1)

    for batch_idx, batch in enumerate(loader, start=1):
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            predictions = model(x)
            loss = criterion(predictions, y)

        if is_training:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        total_count += x.size(0)

        if verbose and (batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == num_batches):
            running_loss = total_loss / max(total_count, 1)
            print(
                f"[{split_name}] batch {batch_idx}/{num_batches} - loss: {running_loss:.6f}",
                flush=True,
            )

    return total_loss / max(total_count, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
) -> tuple[nn.Module, dict[str, list[float]]]:
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.MSELoss()
    history = {"train_loss": [], "val_loss": []}

    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=config.max_epochs,
    ) if isinstance(model, TransformerRegressor) else None
    
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = math.inf
    patience_left = config.patience

    for epoch_idx in range(config.max_epochs):
        epoch_start = time.perf_counter()
        train_loss = _run_epoch(
            model,
            train_loader,
            criterion,
            device=device,
            optimizer=optimizer,
            grad_clip=config.grad_clip,
            split_name="train",
            verbose=config.verbose,
            log_interval=config.log_interval,
        )
        val_loss = _run_epoch(
            model,
            val_loader,
            criterion,
            device=device,
            split_name="val",
            verbose=config.verbose,
            log_interval=config.log_interval,
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        if scheduler is not None:
            scheduler.step()
            
        epoch_seconds = time.perf_counter() - epoch_start

        if config.verbose:
            print(
                f"Epoch {epoch_idx + 1}/{config.max_epochs} - "
                f"train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}, "
                f"time: {epoch_seconds:.1f}s",
                flush=True,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = config.patience
            if config.verbose:
                print("Validation improved; checkpoint updated.", flush=True)
        else:
            patience_left -= 1
            if config.verbose:
                print(f"No improvement. Early-stop patience left: {patience_left}", flush=True)
            if patience_left == 0:
                if config.verbose:
                    print("Early stopping triggered.", flush=True)
                break

    model.load_state_dict(best_state)
    return model, history


def predict_dataset(
    model: nn.Module,
    loader: DataLoader,
    dates: np.ndarray,
    tickers: np.ndarray,
    device: str | None = None,
) -> pd.DataFrame:
    device = device or next(model.parameters()).device.type
    model.eval()
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            predictions = model(x).cpu().numpy()
            targets = batch["y"].cpu().numpy()
            time_idx = batch["time_idx"].cpu().numpy()
            stock_idx = batch["stock_idx"].cpu().numpy()

            for pred, target, t_idx, s_idx in zip(predictions, targets, time_idx, stock_idx):
                rows.append(
                    {
                        "Date": pd.Timestamp(dates[int(t_idx)]),
                        "Ticker": str(tickers[int(s_idx)]),
                        "prediction": float(pred),
                        "target": float(target),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["Date", "Ticker", "prediction", "target"])
    return pd.DataFrame(rows).sort_values(["Date", "Ticker"]).reset_index(drop=True)


def compute_daily_ic(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, group in predictions.groupby("Date"):
        group = group.dropna(subset=["prediction", "target"])
        if len(group) < 2:
            continue
        if group["prediction"].nunique() < 2 or group["target"].nunique() < 2:
            continue
        ic = group["prediction"].corr(group["target"], method="pearson")
        pred_rank = group["prediction"].rank(method="average")
        target_rank = group["target"].rank(method="average")
        rank_ic = pred_rank.corr(target_rank, method="pearson")
        if pd.isna(ic) or pd.isna(rank_ic):
            continue
        rows.append({"Date": date, "IC": float(ic), "RankIC": float(rank_ic), "n_assets": len(group)})
    if not rows:
        return pd.DataFrame(columns=["Date", "IC", "RankIC", "n_assets"])
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


def summarize_cross_sectional_metrics(ic_frame: pd.DataFrame) -> dict[str, float]:
    if ic_frame.empty:
        return {
            "mean_ic": math.nan,
            "std_ic": math.nan,
            "ic_t_stat": math.nan,
            "mean_rank_ic": math.nan,
            "std_rank_ic": math.nan,
            "rank_ic_t_stat": math.nan,
        }

    def _summary(series: pd.Series) -> tuple[float, float, float]:
        values = series.dropna()
        if values.empty:
            return math.nan, math.nan, math.nan
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        if len(values) <= 1 or std == 0.0:
            t_stat = math.nan
        else:
            t_stat = mean / (std / math.sqrt(len(values)))
        return mean, std, float(t_stat) if not math.isnan(t_stat) else math.nan

    mean_ic, std_ic, ic_t_stat = _summary(ic_frame["IC"])
    mean_rank_ic, std_rank_ic, rank_ic_t_stat = _summary(ic_frame["RankIC"])
    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "ic_t_stat": ic_t_stat,
        "mean_rank_ic": mean_rank_ic,
        "std_rank_ic": std_rank_ic,
        "rank_ic_t_stat": rank_ic_t_stat,
    }


def backtest_long_short(
    predictions: pd.DataFrame,
    long_quantile: float = 0.2,
    short_quantile: float = 0.2,
    annualization_factor: int = 252,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if not 0 < long_quantile < 1 or not 0 < short_quantile < 1:
        raise ValueError("Quantiles must lie in (0, 1).")

    rows = []
    for date, group in predictions.groupby("Date"):
        group = group.dropna(subset=["prediction", "target"])
        if len(group) < 5:
            continue

        long_cutoff = group["prediction"].quantile(1.0 - long_quantile)
        short_cutoff = group["prediction"].quantile(short_quantile)
        longs = group[group["prediction"] >= long_cutoff]
        shorts = group[group["prediction"] <= short_cutoff]
        if longs.empty or shorts.empty:
            continue

        strategy_return = float(longs["target"].mean() - shorts["target"].mean())
        rows.append(
            {
                "Date": date,
                "strategy_return": strategy_return,
                "long_count": int(len(longs)),
                "short_count": int(len(shorts)),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["Date", "strategy_return", "long_count", "short_count"]), {
            "mean_return": math.nan,
            "volatility": math.nan,
            "sharpe_ratio": math.nan,
        }

    backtest = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    if backtest.empty:
        return backtest, {
            "mean_return": math.nan,
            "volatility": math.nan,
            "sharpe_ratio": math.nan,
        }

    mean_return = float(backtest["strategy_return"].mean())
    volatility = float(backtest["strategy_return"].std(ddof=1)) if len(backtest) > 1 else 0.0
    sharpe_ratio = math.nan
    if volatility > 0:
        sharpe_ratio = mean_return / volatility * math.sqrt(annualization_factor)

    return backtest, {
        "mean_return": mean_return,
        "volatility": volatility,
        "sharpe_ratio": float(sharpe_ratio) if not math.isnan(sharpe_ratio) else math.nan,
    }


def prepare_experiment_data(
    raw_prices: pd.DataFrame,
    feature_cols: list[str] | None = None,
    horizon: int = 1,
    lookback: int = 60,
    train_size: float = 0.7,
    val_size: float = 0.15,
    winsorize: bool = True,
    apply_rolling_zscore: bool = False,
    rolling_window: int = 60,
) -> ExperimentData:
    feature_cols = list(feature_cols or DEFAULT_FEATURE_COLUMNS)
    cleaned = trim_and_fill_history(flatten_price_frame(raw_prices))
    featured = add_technical_features(cleaned, horizon=horizon)

    if winsorize:
        featured = winsorize_cross_section(featured, feature_cols)
    featured = cross_sectional_standardize(featured, feature_cols)
    if apply_rolling_zscore:
        featured = rolling_standardize(featured, feature_cols, window=rolling_window)

    target_col = f"target_return_{horizon}d"
    featured = featured.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    splits = split_by_time(featured, train_size=train_size, val_size=val_size)
    panels: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    datasets: dict[str, PanelSequenceDataset] = {}

    for split_name, split_frame in splits.items():
        panels[split_name] = build_panel_tensors(split_frame, feature_cols, target_col)
        X, y, dates, tickers = panels[split_name]
        datasets[split_name] = PanelSequenceDataset(
            X,
            y,
            lookback=lookback,
            dates=dates,
            tickers=tickers,
        )

    return ExperimentData(
        cleaned_frame=cleaned,
        featured_frame=featured,
        splits=splits,
        panels=panels,
        datasets=datasets,
        feature_columns=feature_cols,
        target_column=target_col,
    )


def run_single_experiment(
    model_name: str,
    experiment_data: ExperimentData,
    train_config: TrainConfig | None = None,
) -> dict[str, Any]:
    train_config = train_config or TrainConfig()
    if train_config.verbose:
        print(f"=== Running model: {model_name.upper()} ===", flush=True)
    loaders = make_data_loaders(experiment_data.datasets, batch_size=train_config.batch_size)
    model = build_model(
        model_name=model_name,
        input_dim=len(experiment_data.feature_columns),
        hidden_dim=train_config.hidden_dim,
        num_layers=train_config.num_layers,
        dropout=train_config.dropout,
        num_heads=train_config.num_heads,
    )
    model, history = train_model(model, loaders["train"], loaders["val"], train_config)

    device = train_config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    val_dataset = experiment_data.datasets["val"]
    test_dataset = experiment_data.datasets["test"]
    val_predictions = predict_dataset(model, loaders["val"], val_dataset.dates, val_dataset.tickers, device)
    test_predictions = predict_dataset(model, loaders["test"], test_dataset.dates, test_dataset.tickers, device)

    val_ic = compute_daily_ic(val_predictions)
    test_ic = compute_daily_ic(test_predictions)
    portfolio_curve, portfolio_summary = backtest_long_short(test_predictions)

    if train_config.verbose:
        if val_predictions.empty:
            print("Warning: validation prediction table is empty.", flush=True)
        if test_predictions.empty:
            print("Warning: test prediction table is empty.", flush=True)
        if val_ic.empty:
            print("Warning: no valid daily IC rows were produced on validation.", flush=True)
        if test_ic.empty:
            print("Warning: no valid daily IC rows were produced on test.", flush=True)
        if portfolio_curve.empty:
            print("Warning: no valid long-short portfolio rows were produced on test.", flush=True)

    return {
        "model": model,
        "history": history,
        "val_predictions": val_predictions,
        "test_predictions": test_predictions,
        "val_ic": val_ic,
        "test_ic": test_ic,
        "val_summary": summarize_cross_sectional_metrics(val_ic),
        "test_summary": summarize_cross_sectional_metrics(test_ic),
        "portfolio_curve": portfolio_curve,
        "portfolio_summary": portfolio_summary,
    }


def run_model_suite(
    experiment_data: ExperimentData,
    model_names: list[str] | None = None,
    train_config: TrainConfig | None = None,
) -> dict[str, dict[str, Any]]:
    model_names = model_names or ["RNN", "LSTM", "GRU", "TRANSFORMER"]
    results: dict[str, dict[str, Any]] = {}
    total_models = len(model_names)

    for idx, model_name in enumerate(model_names, start=1):
        if train_config is not None and train_config.verbose:
            print(f"=== Model {idx}/{total_models} ===", flush=True)
        results[model_name] = run_single_experiment(model_name, experiment_data, train_config)

    return results


def build_summary_frame(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    summary_rows = []
    for model_name, result in results.items():
        summary_rows.append(
            {
                "model": model_name,
                **result["test_summary"],
                **result["portfolio_summary"],
            }
        )
    return pd.DataFrame(summary_rows).sort_values("mean_ic", ascending=False).reset_index(drop=True)


def plot_training_histories(
    results: dict[str, dict[str, Any]],
    model_names: list[str] | None = None,
) -> Any:
    import matplotlib.pyplot as plt

    selected_models = model_names or list(results.keys())
    if not selected_models:
        raise ValueError("No models available to plot.")

    fig, axes = plt.subplots(
        len(selected_models),
        1,
        figsize=(10, max(4, 3.5 * len(selected_models))),
        squeeze=False,
    )

    for ax, model_name in zip(axes.flat, selected_models):
        history = results[model_name]["history"]
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        ax.plot(epochs, history["train_loss"], marker="o", label="Train loss")
        ax.plot(epochs, history["val_loss"], marker="s", label="Val loss")
        ax.set_title(f"{model_name} Loss History")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    return fig


def plot_ic_series(
    results: dict[str, dict[str, Any]],
    model_name: str,
    split: str = "test",
    rolling_window: int = 20,
) -> Any:
    import matplotlib.pyplot as plt

    ic_key = f"{split}_ic"
    if ic_key not in results[model_name]:
        raise KeyError(f"{ic_key} not found for model {model_name}.")

    ic_frame = results[model_name][ic_key].copy()
    if ic_frame.empty:
        raise ValueError(f"No IC data available for model {model_name} on split {split}.")

    ic_frame["IC_roll"] = ic_frame["IC"].rolling(rolling_window, min_periods=1).mean()
    ic_frame["RankIC_roll"] = ic_frame["RankIC"].rolling(rolling_window, min_periods=1).mean()

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(ic_frame["Date"], ic_frame["IC"], alpha=0.35, label="Daily IC")
    axes[0].plot(ic_frame["Date"], ic_frame["IC_roll"], linewidth=2, label=f"{rolling_window}D rolling IC")
    axes[0].axhline(0.0, color="black", linewidth=1, alpha=0.7)
    axes[0].set_title(f"{model_name} {split.title()} IC")
    axes[0].set_ylabel("IC")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(ic_frame["Date"], ic_frame["RankIC"], alpha=0.35, label="Daily Rank IC")
    axes[1].plot(
        ic_frame["Date"],
        ic_frame["RankIC_roll"],
        linewidth=2,
        label=f"{rolling_window}D rolling Rank IC",
    )
    axes[1].axhline(0.0, color="black", linewidth=1, alpha=0.7)
    axes[1].set_title(f"{model_name} {split.title()} Rank IC")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Rank IC")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    return fig


def plot_portfolio_curve(
    results: dict[str, dict[str, Any]],
    model_name: str,
) -> Any:
    import matplotlib.pyplot as plt

    portfolio_curve = results[model_name]["portfolio_curve"].copy()
    if portfolio_curve.empty:
        raise ValueError(f"No portfolio curve available for model {model_name}.")

    portfolio_curve["cumulative_return"] = (1.0 + portfolio_curve["strategy_return"]).cumprod() - 1.0

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(portfolio_curve["Date"], portfolio_curve["cumulative_return"], linewidth=2)
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.7)
    ax.set_title(f"{model_name} Long-Short Portfolio")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_summary_bars(
    summary: pd.DataFrame,
    metrics: list[str] | None = None,
) -> Any:
    import matplotlib.pyplot as plt

    metrics = metrics or ["mean_ic", "mean_rank_ic", "sharpe_ratio"]
    missing = [metric for metric in metrics if metric not in summary.columns]
    if missing:
        raise KeyError(f"Summary is missing metrics: {missing}")

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)

    for ax, metric in zip(axes.flat, metrics):
        ordered = summary.sort_values(metric, ascending=False)
        ax.bar(ordered["model"], ordered[metric])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig
