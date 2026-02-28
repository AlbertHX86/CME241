"""
Compute Asset/Strategy metrics from test_decisions_and_return_curve_*.csv:
  Final Wealth, Ann. Return, Ann. Volatility, Sharpe Ratio.

Formulas (daily gross returns g_t = price_{t+1}/price_t or portfolio_gross):
  - Final Wealth = product of g_t over test period (or last row of cum_*).
  - Ann. Return = (Final_Wealth)^(252/N) - 1  with N = number of trading days.
  - Ann. Volatility = std(daily_ret) * sqrt(252),  daily_ret = g_t - 1.
  - Sharpe Ratio = (Ann_Return - r_f) / Ann_Volatility  (r_f = 0 by default).

CSV must have: portfolio_gross (strategy), realized_gross_DUK, realized_gross_PGR for assets.
"""
import numpy as np
import pandas as pd
from pathlib import Path

TRADING_DAYS_PER_YEAR = 252
RF_ANNUAL = 0.0  # risk-free rate for Sharpe; set to 0 or (e.g. 0.02) if desired


def compute_metrics(gross_series: pd.Series) -> dict:
    """
    From a series of daily gross returns (e.g. realized_gross_XXX or portfolio_gross).
    Returns: final_wealth, ann_return, ann_vol, sharpe.
    """
    gross = np.asarray(gross_series, dtype=float)
    gross = gross[~np.isnan(gross)]
    if len(gross) == 0:
        return {"final_wealth": np.nan, "ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan}
    n = len(gross)
    final_wealth = float(np.prod(gross))
    daily_ret = gross - 1.0
    ann_return = (final_wealth ** (TRADING_DAYS_PER_YEAR / n)) - 1.0
    ann_vol = float(np.std(daily_ret)) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (ann_return - RF_ANNUAL) / ann_vol if ann_vol > 1e-12 else np.nan
    return {
        "final_wealth": final_wealth,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
    }


def report_metrics(csv_path: Path, tickers: list = None) -> pd.DataFrame:
    """
    Read backtest CSV and compute Final Wealth, Ann. Return, Ann. Vol, Sharpe
    for each ticker (Asset buy-and-hold) and for Strategy.
    """
    df = pd.read_csv(csv_path)
    if tickers is None:
        tickers = [c.replace("cum_", "") for c in df.columns if c.startswith("cum_") and c != "cum_wealth"]
        tickers = [t for t in tickers if t != "wealth" and t != "SP500"]
    rows = []
    for name, gross_col in [
        ("Strategy", "portfolio_gross"),
    ]:
        if gross_col not in df.columns:
            continue
        m = compute_metrics(df[gross_col])
        rows.append({
            "Asset/Strategy": name,
            "Final Wealth": m["final_wealth"],
            "Ann. Return": m["ann_return"],
            "Ann. Volatility": m["ann_vol"],
            "Sharpe Ratio": m["sharpe"],
        })
    for ticker in tickers:
        col = f"realized_gross_{ticker}"
        if col not in df.columns:
            continue
        m = compute_metrics(df[col])
        rows.append({
            "Asset/Strategy": ticker,
            "Final Wealth": m["final_wealth"],
            "Ann. Return": m["ann_return"],
            "Ann. Volatility": m["ann_vol"],
            "Sharpe Ratio": m["sharpe"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/all_5_stocks/test_decisions_and_return_curve_aapl_duk_pgr_msft_tsla.csv")
    if not csv_path.exists():
        csv_path = Path(__file__).resolve().parent / "data" / "test_decisions_and_return_curve_aapl_duk_pgr_msft_tsla.csv"
    if not csv_path.exists():
        print("Usage: python compute_metrics_from_backtest_csv.py <path_to_csv>")
        print("Expected CSV columns: date, portfolio_gross, realized_gross_DUK, realized_gross_PGR, ...")
        sys.exit(1)
    result = report_metrics(csv_path, tickers=["DUK", "PGR"])
    print(result.to_string(index=False))
    result.to_csv(csv_path.parent / "metrics_duk_pgr_strategy.csv", index=False)
