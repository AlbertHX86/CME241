import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class MarkovDecisionProcess:

    def __init__(self, P, R_low, R_high, Sigma_low, Sigma_high, Rf, beta, K, n_assets, risk_aversion):
        self.P = np.asarray(P, dtype=float)
        assert self.P.shape == (2, 2)
        self.R = {
            0: np.asarray(R_low, dtype=float),
            1: np.asarray(R_high, dtype=float),
        }
        self.Sigma = {
            0: np.asarray(Sigma_low, dtype=float),
            1: np.asarray(Sigma_high, dtype=float),
        }
        self.Rf = float(Rf)
        self.beta = float(beta)
        self.a = float(risk_aversion)
        self.n = int(n_assets)
        assert self.R[0].shape == (self.n,) and self.R[1].shape == (self.n,) and self.a > 0
        assert self.Sigma[0].shape == (self.n, self.n) and self.Sigma[1].shape == (self.n, self.n)
        self.W = [i / K for i in range(K + 1)]
        self._actions = self._build_feasible_actions()

    def states(self) -> List[int]:
        return [0, 1]

    def actions(self, state: int) -> List[np.ndarray]:
        return self._actions

    def transition_prob(self, state: int, action: np.ndarray, next_state: int) -> float:
        return float(self.P[state, next_state])

    def reward(self, state: int, action: np.ndarray, next_state: int) -> float:
        w = np.asarray(action, dtype=float)
        # Mean-variance certainty equivalent under regime-conditioned moments.
        mu = self.R[next_state]
        Sigma = self.Sigma[next_state]
        mean_gross = float(np.dot(w, mu)) + (1.0 - np.sum(w)) * self.Rf
        var_gross = float(w @ Sigma @ w)
        return float(mean_gross - 0.5 * self.a * var_gross)

    def _build_feasible_actions(self) -> List[np.ndarray]:
        acts: List[np.ndarray] = []
        for w_tuple in product(self.W, repeat=self.n):
            w = np.array(w_tuple, dtype=float)
            if np.sum(w) <= 1.0 + 1e-12:
                acts.append(w)
        return acts


def finite_horizon_dp(mdp: MarkovDecisionProcess, T: int) -> Tuple[List[Dict[int, float]], List[Dict[int, np.ndarray]]]:
    states = mdp.states()
    V: List[Dict[int, float]] = [dict() for _ in range(T + 1)]
    pi: List[Dict[int, np.ndarray]] = [dict() for _ in range(T)]
    for z in states:
        V[T][z] = 0.0
    for t in range(T - 1, -1, -1):
        for z in states:
            best_val, best_action = -np.inf, None
            for a in mdp.actions(z):
                exp_val = sum(
                    mdp.transition_prob(z, a, z_next) * (mdp.reward(z, a, z_next) + mdp.beta * V[t + 1][z_next])
                    for z_next in states
                )
                if exp_val > best_val:
                    best_val, best_action = exp_val, a
            V[t][z] = float(best_val)
            pi[t][z] = np.array(best_action, dtype=float)
    return V, pi


def run_train_test_backtest(
    data_dir: Path,
    tickers: Optional[List[str]] = None,
    stock_paths: Optional[Dict[str, Path]] = None,
    output_tag: str = "default",
    train_ratio: float = 0.8,
    beta: float = 0.99,
    K: int = 10,
    risk_aversion: float = 10.0,
    rf_gross: float = 1.0,
) -> pd.DataFrame:
    from estimate_regime_params import RegimeParamsEstimator, TICKERS

    use_tickers = list(tickers) if tickers is not None else list(TICKERS)
    estimator = RegimeParamsEstimator(data_dir=data_dir, stock_paths=stock_paths, tickers=use_tickers, rf_gross=rf_gross)
    merged = estimator._load_and_merge()
    returns_df, _ = estimator._compute_gross_returns_and_align_regime(merged)

    n = len(returns_df)
    if n < 10:
        raise ValueError("Not enough aligned observations for train/test split.")
    split_idx = int(n * train_ratio)
    split_idx = min(max(split_idx, 1), n - 1)

    train_df = returns_df.iloc[:split_idx].copy()
    test_df = returns_df.iloc[split_idx:].copy().reset_index(drop=True)

    z_train = train_df["z"].astype(int)
    P = estimator._estimate_P(z_train)
    R_low, R_high, _ = estimator._estimate_R_low_R_high(train_df, z_train)
    Sigma_low, Sigma_high = estimator._estimate_covariances(train_df, z_train)
    Rf = float(estimator.rf_gross)

    mdp = MarkovDecisionProcess(
        P=P,
        R_low=R_low,
        R_high=R_high,
        Sigma_low=Sigma_low,
        Sigma_high=Sigma_high,
        Rf=Rf,
        beta=beta,
        K=K,
        n_assets=len(use_tickers),
        risk_aversion=risk_aversion,
    )
    _, pi = finite_horizon_dp(mdp, T=len(test_df))

    rows: List[Dict[str, float]] = []
    wealth = 1.0
    gross_cols = [f"gross_{ticker}" for ticker in use_tickers]

    for t in range(len(test_df)):
        z_t = int(test_df.loc[t, "z"])
        w = np.asarray(pi[t][z_t], dtype=float)
        gross_vec = test_df.loc[t, gross_cols].to_numpy(dtype=float)
        portfolio_gross = float(np.dot(w, gross_vec) + (1.0 - np.sum(w)) * Rf)
        wealth *= portfolio_gross

        row = {
            "date": pd.to_datetime(test_df.loc[t, "date"]).strftime("%Y-%m-%d"),
            "regime": z_t,
            "portfolio_gross": portfolio_gross,
            "cum_wealth": wealth,
        }
        for i, ticker in enumerate(use_tickers):
            row[f"w_{ticker}"] = float(w[i])
            row[f"realized_gross_{ticker}"] = float(gross_vec[i])
        rows.append(row)

    result = pd.DataFrame(rows)

    # Buy-and-hold benchmark curves for individual stocks over the same test dates.
    for ticker in use_tickers:
        result[f"cum_{ticker}"] = result[f"realized_gross_{ticker}"].cumprod()

    # S&P 500 benchmark curve aligned to test dates.
    sp500_candidates = [
        data_dir / "S&P 500 Historical Data.csv",
        data_dir / "S&P 500 Historical Data (1).csv",
    ]
    sp500_source: Optional[Path] = None
    for candidate in sp500_candidates:
        if candidate.exists():
            sp500_source = candidate
            break

    if sp500_source is not None:
        sp = pd.read_csv(sp500_source)
        dcol = "Date" if "Date" in sp.columns else sp.columns[0]
        if "Price" in sp.columns:
            pcol = "Price"
        else:
            close_like = [c for c in sp.columns if "adj" in c.lower() or "close" in c.lower() or "price" in c.lower()]
            pcol = close_like[-1] if close_like else sp.columns[1]
        sp[dcol] = pd.to_datetime(sp[dcol], errors="coerce")
        sp[pcol] = pd.to_numeric(sp[pcol].astype(str).str.replace(",", "", regex=False), errors="coerce")
        sp = sp[[dcol, pcol]].rename(columns={dcol: "date", pcol: "sp500_price"})
        sp = sp.dropna().sort_values("date").reset_index(drop=True)
        sp["sp500_gross"] = sp["sp500_price"].shift(-1) / sp["sp500_price"]
        sp = sp.iloc[:-1][["date", "sp500_gross"]].copy()

        test_dates = pd.DataFrame({"date": pd.to_datetime(result["date"])})
        aligned = test_dates.merge(sp, on="date", how="left")
        aligned["sp500_gross"] = aligned["sp500_gross"].fillna(1.0)
        result["cum_SP500"] = aligned["sp500_gross"].cumprod()
    else:
        print("WARNING: No S&P 500 CSV found for benchmark curve.")

    output_path = data_dir / f"test_decisions_and_return_curve_{output_tag}.csv"
    result.to_csv(output_path, index=False)

    curve_path = data_dir / f"test_return_curve_{output_tag}.png"
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4.5))
        x = pd.to_datetime(result["date"])
        plt.plot(x, result["cum_wealth"], label="Strategy", linewidth=2.0, color="black")
        for ticker in use_tickers:
            plt.plot(x, result[f"cum_{ticker}"], label=ticker, linewidth=1.3)
        if "cum_SP500" in result.columns:
            plt.plot(x, result["cum_SP500"], label="S&P 500", linewidth=1.6, linestyle="--")
        plt.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
        plt.title("Test Set: Strategy vs Stocks vs S&P 500")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Wealth")
        plt.legend()
        plt.tight_layout()
        plt.savefig(curve_path, dpi=140)
        plt.close()
        print(f"Saved return curve figure to: {curve_path}")
        if sp500_source is not None:
            print(f"S&P 500 source used: {sp500_source}")
    except Exception as e:
        print(f"Skipped return curve figure ({type(e).__name__}: {e})")

    print("Train/Test split:")
    print(f"  train rows: {len(train_df)} ({train_ratio:.0%})")
    print(f"  test rows:  {len(test_df)} ({1.0 - train_ratio:.0%})")
    print(f"  tickers:    {use_tickers}")
    print(f"Saved test decisions + return curve to: {output_path}")
    print(f"Final test cumulative wealth: {wealth:.6f}")
    return result


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent / "data"
    out = run_train_test_backtest(
        data_dir=data_dir,
        output_tag="default",
        train_ratio=0.8,
        beta=0.99,
        K=10,
        risk_aversion=10.0,
        rf_gross=1.0,
    )
    print("\nTest decisions preview:")
    print(out.head(10).to_string(index=False))
