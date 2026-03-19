from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def evaluate_policy(env, agent) -> pd.DataFrame:
    obs = env.reset()
    rows: List[Dict[str, float]] = []
    done = False
    while not done:
        action, _, _ = agent.act(obs, deterministic=True)
        next_obs, reward, done, info = env.step(action)
        row: Dict[str, float] = {
            "date": pd.to_datetime(info["date"]),
            "reward": float(reward),
            "portfolio_gross": float(info["portfolio_gross"]),
            "net_gross": float(info["net_gross"]),
            "log_return": float(info["log_return"]),
            "excess_log_return": float(info["excess_log_return"]),
            "turnover": float(info["turnover"]),
            "cum_wealth": float(info["wealth"]),
            "w_cash": float(np.asarray(info["weights"])[0]),
            "rf_gross": float(info["rf_gross"]),
        }
        for i, ticker in enumerate(env.bundle.tickers):
            row[f"w_{ticker}"] = float(np.asarray(info["weights"])[i + 1])
            row[f"gross_{ticker}"] = float(np.asarray(info["gross_vec"])[i])
        rows.append(row)
        obs = next_obs
    return pd.DataFrame(rows)


def compute_performance_metrics(eval_df: pd.DataFrame) -> Dict[str, float]:
    if len(eval_df) == 0:
        raise ValueError("eval_df is empty.")
    daily_log = eval_df["log_return"].to_numpy(dtype=np.float32)
    daily_simple = np.exp(daily_log) - 1.0
    wealth = float(eval_df["cum_wealth"].iloc[-1])
    ann_return = float((1.0 + daily_simple.mean()) ** 252 - 1.0)
    ann_vol = float(daily_simple.std(ddof=0) * np.sqrt(252.0))
    sharpe = float(np.sqrt(252.0) * daily_simple.mean() / (daily_simple.std(ddof=0) + 1e-8))
    cum = eval_df["cum_wealth"].to_numpy(dtype=np.float32)
    running_max = np.maximum.accumulate(cum)
    drawdown = cum / np.maximum(running_max, 1e-8) - 1.0
    max_drawdown = float(drawdown.min())
    avg_turnover = float(eval_df["turnover"].mean())
    return {
        "final_wealth": wealth,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "avg_turnover": avg_turnover,
    }
