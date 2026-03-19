from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_TICKER_FILES: Dict[str, str] = {
    "AAPL": "apple_stock.csv",
    "MSFT": "microsoft_stock.csv",
    "TSLA": "tesla_stock.csv",
    "DUK": "duk_stock.csv",
    "PGR": "pgr_stock.csv",
}


@dataclass
class DataBundle:
    dates: np.ndarray
    feature_array: np.ndarray
    next_gross_array: np.ndarray
    price_array: np.ndarray
    tickers: List[str]
    feature_names: List[str]


def _load_single_stock_csv(path: Path, ticker: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    out = df[required].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    for col in required[1:]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna().sort_values("Date").reset_index(drop=True)

    rename_map = {col: f"{ticker}_{col.replace(' ', '_')}" for col in required if col != "Date"}
    return out.rename(columns=rename_map)


def load_merged_market_data(data_dir: Path, tickers: Sequence[str]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for ticker in tickers:
        stock_df = _load_single_stock_csv(data_dir / DEFAULT_TICKER_FILES[ticker], ticker)
        merged = stock_df if merged is None else merged.merge(stock_df, on="Date", how="inner")
    if merged is None:
        raise ValueError("No ticker data loaded.")
    return merged.sort_values("Date").reset_index(drop=True)


def build_feature_dataframe(
    merged: pd.DataFrame,
    tickers: Sequence[str],
    ma_short: int = 5,
    ma_long: int = 20,
    vol_lookback: int = 20,
) -> Tuple[pd.DataFrame, List[str]]:
    df = merged.copy()
    feature_names: List[str] = []

    for ticker in tickers:
        adj = f"{ticker}_Adj_Close"
        opn = f"{ticker}_Open"
        high = f"{ticker}_High"
        low = f"{ticker}_Low"
        close = f"{ticker}_Close"
        vol = f"{ticker}_Volume"
        prev_adj = df[adj].shift(1)

        df[f"{ticker}_gap"] = df[opn] / prev_adj - 1.0
        df[f"{ticker}_co"] = df[close] / df[opn] - 1.0
        df[f"{ticker}_ho"] = df[high] / df[opn] - 1.0
        df[f"{ticker}_lo"] = df[low] / df[opn] - 1.0
        df[f"{ticker}_ret1"] = df[adj].pct_change()
        df[f"{ticker}_ma{ma_short}_ratio"] = df[adj] / df[adj].rolling(ma_short).mean() - 1.0
        df[f"{ticker}_ma{ma_long}_ratio"] = df[adj] / df[adj].rolling(ma_long).mean() - 1.0
        df[f"{ticker}_vol{vol_lookback}"] = df[f"{ticker}_ret1"].rolling(vol_lookback).std()
        df[f"{ticker}_vol_chg"] = np.log1p(df[vol]).diff()
        df[f"next_gross_{ticker}"] = df[adj].shift(-1) / df[adj]

        feature_names.extend(
            [
                "gap",
                "co",
                "ho",
                "lo",
                "ret1",
                f"ma{ma_short}_ratio",
                f"ma{ma_long}_ratio",
                f"vol{vol_lookback}",
                "vol_chg",
            ]
        )

    df = df.dropna().reset_index(drop=True)
    return df, sorted(set(feature_names))


def fit_standardizer(train_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((features - mean) / std).astype(np.float32)


def build_data_bundle(
    data_dir: Path,
    tickers: Sequence[str],
    train_ratio: float = 0.8,
) -> Tuple[DataBundle, int, np.ndarray, np.ndarray]:
    merged = load_merged_market_data(data_dir, tickers)
    feat_df, feature_names = build_feature_dataframe(merged, tickers)

    per_asset_feature_names = [
        "gap",
        "co",
        "ho",
        "lo",
        "ret1",
        "ma5_ratio",
        "ma20_ratio",
        "vol20",
        "vol_chg",
    ]
    feature_blocks = []
    next_gross_blocks = []
    price_blocks = []
    for ticker in tickers:
        feature_cols = [f"{ticker}_{name}" for name in per_asset_feature_names]
        feature_blocks.append(feat_df[feature_cols].to_numpy(dtype=np.float32))
        next_gross_blocks.append(feat_df[[f"next_gross_{ticker}"]].to_numpy(dtype=np.float32))
        price_blocks.append(feat_df[[f"{ticker}_Adj_Close"]].to_numpy(dtype=np.float32))

    feature_array = np.stack(feature_blocks, axis=1)
    next_gross_array = np.concatenate(next_gross_blocks, axis=1)
    price_array = np.concatenate(price_blocks, axis=1)
    dates = feat_df["Date"].to_numpy()

    split_idx = int(len(dates) * train_ratio)
    train_features = feature_array[:split_idx]
    mean, std = fit_standardizer(train_features.reshape(train_features.shape[0], -1))
    normalized = apply_standardizer(feature_array.reshape(feature_array.shape[0], -1), mean, std)
    feature_array = normalized.reshape(feature_array.shape)

    bundle = DataBundle(
        dates=dates,
        feature_array=feature_array.astype(np.float32),
        next_gross_array=next_gross_array.astype(np.float32),
        price_array=price_array.astype(np.float32),
        tickers=list(tickers),
        feature_names=per_asset_feature_names,
    )
    return bundle, split_idx, mean, std
