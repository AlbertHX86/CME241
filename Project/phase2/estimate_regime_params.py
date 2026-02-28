"""
Regime-based MDP parameter estimation from CSV data.
Outputs:
- P (2x2)
- R_low, R_high (length-3 mean gross returns)
- Sigma_low, Sigma_high (3x3 covariance of gross returns)
- Rf
Stock order: [AAPL, TSLA, MSFT].
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, List

TICKERS = ["AAPL", "TSLA", "MSFT"]
MIN_SAMPLES_WARN = 30


class RegimeParamsEstimator:
    """
    Estimates MDP parameters from CSV: P (2x2), R_low, R_high (length 3), Rf.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        state_path: Optional[Path] = None,
        stock_paths: Optional[Dict[str, Path]] = None,
        tickers: Optional[List[str]] = None,
        rf_gross: float = 1.0,
    ):
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent / "data"
            if not data_dir.exists():
                data_dir = Path("/mnt/data")
        self.data_dir = Path(data_dir)
        self.rf_gross = float(rf_gross)
        self.tickers = list(tickers) if tickers is not None else list(TICKERS)
        self.state_path = Path(state_path) if state_path else self.data_dir / "sp500_market_state_2014_2023.csv"
        if stock_paths is None:
            default_map = {
                "AAPL": self.data_dir / "apple_stock.csv",
                "TSLA": self.data_dir / "tesla_stock.csv",
                "MSFT": self.data_dir / "microsoft_stock.csv",
            }
            stock_paths = {}
            for ticker in self.tickers:
                stock_paths[ticker] = default_map.get(ticker, self.data_dir / f"{ticker.lower()}_stock.csv")
        self.stock_paths = {k: Path(v) for k, v in stock_paths.items()}
        self._merged: Optional[pd.DataFrame] = None
        self._returns_df: Optional[pd.DataFrame] = None
        self._z_aligned: Optional[pd.Series] = None
        self._counts: Optional[dict] = None

    def _load_and_merge(self) -> pd.DataFrame:
        state = pd.read_csv(self.state_path)
        date_col = "datetime" if "datetime" in state.columns else state.columns[0]
        bool_col = "bool" if "bool" in state.columns else state.columns[1]
        state[date_col] = pd.to_datetime(state[date_col])
        state = state.rename(columns={date_col: "date", bool_col: "z"})
        state = state[["date", "z"]].dropna().sort_values("date").reset_index(drop=True)

        dfs = [state]
        for ticker in self.tickers:
            path = self.stock_paths[ticker]
            df = pd.read_csv(path)
            dc = "Date" if "Date" in df.columns else df.columns[0]
            ac = "Adj Close" if "Adj Close" in df.columns else [c for c in df.columns if "adj" in c.lower() or "close" in c.lower()][-1]
            df[dc] = pd.to_datetime(df[dc])
            df = df[[dc, ac]].rename(columns={dc: "date", ac: f"adj_{ticker}"})
            df = df.dropna().sort_values("date").reset_index(drop=True)
            dfs.append(df)

        merged = dfs[0]
        for d in dfs[1:]:
            merged = merged.merge(d, on="date", how="inner")
        return merged.sort_values("date").reset_index(drop=True)

    def _compute_gross_returns_and_align_regime(self, merged: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        out = merged[["date", "z"]].copy()
        for ticker in self.tickers:
            out[f"gross_{ticker}"] = merged[f"adj_{ticker}"].shift(-1) / merged[f"adj_{ticker}"]
        out = out.iloc[:-1].copy()
        return out, out["z"]

    def _estimate_P(self, z: pd.Series) -> np.ndarray:
        z = z.astype(int)
        N = np.zeros((2, 2))
        for t in range(len(z) - 1):
            i, j = z.iloc[t], z.iloc[t + 1]
            if 0 <= i <= 1 and 0 <= j <= 1:
                N[i, j] += 1
        P = np.zeros((2, 2))
        for i in range(2):
            row_sum = N[i, :].sum()
            P[i, :] = (N[i, :] / row_sum) if row_sum > 0 else (np.eye(2)[i])
        return P

    def _estimate_R_low_R_high(self, df: pd.DataFrame, z: pd.Series) -> Tuple[np.ndarray, np.ndarray, dict]:
        n = len(self.tickers)
        R_low = np.zeros(n)
        R_high = np.zeros(n)
        counts = {"low": [], "high": []}
        for k, ticker in enumerate(self.tickers):
            g = df[f"gross_{ticker}"]
            g0, g1 = g[z == 0].dropna(), g[z == 1].dropna()
            R_low[k], R_high[k] = g0.mean(), g1.mean()
            counts["low"].append(len(g0))
            counts["high"].append(len(g1))
        return R_low, R_high, counts

    def _estimate_covariances(self, df: pd.DataFrame, z: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        cols = [f"gross_{ticker}" for ticker in self.tickers]
        low = df.loc[z == 0, cols].dropna()
        high = df.loc[z == 1, cols].dropna()

        # Use diagonal fallback when observations are insufficient.
        if len(low) >= 2:
            Sigma_low = low.cov().to_numpy(dtype=float)
        elif len(low) == 1:
            v = low.iloc[0].to_numpy(dtype=float)
            Sigma_low = np.diag(np.maximum(v * 0.0, 1e-12))
        else:
            Sigma_low = np.eye(len(self.tickers)) * 1e-12

        if len(high) >= 2:
            Sigma_high = high.cov().to_numpy(dtype=float)
        elif len(high) == 1:
            v = high.iloc[0].to_numpy(dtype=float)
            Sigma_high = np.diag(np.maximum(v * 0.0, 1e-12))
        else:
            Sigma_high = np.eye(len(self.tickers)) * 1e-12

        # Numerical stabilization.
        Sigma_low = Sigma_low + np.eye(len(self.tickers)) * 1e-12
        Sigma_high = Sigma_high + np.eye(len(self.tickers)) * 1e-12
        return Sigma_low, Sigma_high

    def _sanity_checks(self, merged: pd.DataFrame, df: pd.DataFrame, z: pd.Series,
                       P: np.ndarray, R_low: np.ndarray, R_high: np.ndarray,
                       Sigma_low: np.ndarray, Sigma_high: np.ndarray,
                       Rf: float, counts: dict) -> None:
        print("Date range (merged):", merged["date"].min(), "to", merged["date"].max())
        print("z=0:", (z == 0).sum(), "| z=1:", (z == 1).sum())
        print("P row sums:", P.sum(axis=1))
        print("R_low:", R_low, "| R_high:", R_high, "| Rf:", Rf)
        print("diag(Sigma_low):", np.diag(Sigma_low))
        print("diag(Sigma_high):", np.diag(Sigma_high))
        for ticker, n0, n1 in zip(self.tickers, counts["low"], counts["high"]):
            if n0 < MIN_SAMPLES_WARN or n1 < MIN_SAMPLES_WARN:
                print("WARNING: {} low={} high={}".format(ticker, n0, n1))

    def build_params(self, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        merged = self._load_and_merge()
        df, z = self._compute_gross_returns_and_align_regime(merged)
        P = self._estimate_P(z)
        R_low, R_high, counts = self._estimate_R_low_R_high(df, z)
        Sigma_low, Sigma_high = self._estimate_covariances(df, z)
        Rf = self.rf_gross
        self._merged, self._returns_df, self._z_aligned, self._counts = merged, df, z, counts
        if verbose:
            self._sanity_checks(merged, df, z, P, R_low, R_high, Sigma_low, Sigma_high, Rf, counts)
        return P, R_low, R_high, Sigma_low, Sigma_high, Rf


def build_params(
    state_path: Optional[Path] = None,
    stock_paths: Optional[Dict[str, Path]] = None,
    tickers: Optional[List[str]] = None,
    rf_gross: float = 1.0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    data_dir = Path(__file__).resolve().parent / "data"
    if not data_dir.exists():
        data_dir = Path("/mnt/data")
    est = RegimeParamsEstimator(
        data_dir=data_dir,
        state_path=state_path,
        stock_paths=stock_paths,
        tickers=tickers,
        rf_gross=rf_gross,
    )
    return est.build_params(verbose=verbose)
