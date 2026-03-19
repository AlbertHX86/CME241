from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _to_serializable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def make_run_dir(base_dir: str | Path = "output", run_name: str = "portfolio_rl") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(dict(payload)), f, indent=2, ensure_ascii=False)


def write_dataframe(path: str | Path, rows: pd.DataFrame) -> None:
    rows.to_csv(path, index=False)


def write_training_history(path: str | Path, history: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(list(history))
    df.to_csv(path, index=False)
    return df


def build_eval_report(eval_df: pd.DataFrame) -> pd.DataFrame:
    report = eval_df.copy()
    risky_gross_cols = [col for col in report.columns if col.startswith("gross_")]
    for col in risky_gross_cols:
        ticker = col.replace("gross_", "")
        report[f"value_{ticker}"] = report[col].cumprod()

    report["value_cash"] = report["rf_gross"].cumprod()
    report["value_equal_weight"] = report[risky_gross_cols].mean(axis=1).cumprod()

    mean_weights = [float(report["w_cash"].mean())]
    tickers = [col.replace("gross_", "") for col in risky_gross_cols]
    for ticker in tickers:
        mean_weights.append(float(report[f"w_{ticker}"].mean()))

    avg_hold_gross = mean_weights[0] * report["rf_gross"].to_numpy()
    for i, ticker in enumerate(tickers):
        avg_hold_gross = avg_hold_gross + mean_weights[i + 1] * report[f"gross_{ticker}"].to_numpy()
    report["value_average_hold"] = np.cumprod(avg_hold_gross)
    return report


def plot_eval_curves(report_df: pd.DataFrame, path: str | Path) -> None:
    plt.figure(figsize=(12, 7))
    plt.plot(report_df["date"], report_df["cum_wealth"], label="Portfolio", linewidth=2.0, color="black")
    plt.plot(report_df["date"], report_df["value_equal_weight"], label="Equal Weight", linewidth=1.5, linestyle="--")
    plt.plot(report_df["date"], report_df["value_average_hold"], label="Average Hold", linewidth=1.5, linestyle=":")
    value_cols = [col for col in report_df.columns if col.startswith("value_") and col not in {"value_equal_weight", "value_average_hold", "value_cash"}]
    for col in value_cols:
        plt.plot(report_df["date"], report_df[col], label=col.replace("value_", ""), alpha=0.8)
    plt.title("Test Set Wealth Paths")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Value")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_turnover(report_df: pd.DataFrame, path: str | Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(report_df["date"], report_df["turnover"], color="tab:red")
    axes[0].set_title("Daily Turnover")
    axes[0].set_ylabel("Turnover")
    axes[1].plot(report_df["date"], report_df["w_cash"], label="Cash", linewidth=1.5)
    weight_cols = [col for col in report_df.columns if col.startswith("w_") and col != "w_cash"]
    for col in weight_cols:
        axes[1].plot(report_df["date"], report_df[col], label=col.replace("w_", ""), alpha=0.75)
    axes[1].set_title("Portfolio Weights")
    axes[1].set_ylabel("Weight")
    axes[1].legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curves(history_df: pd.DataFrame, reward_path: str | Path, losses_path: str | Path) -> None:
    if history_df.empty:
        return

    plt.figure(figsize=(10, 5))
    if "total_steps" in history_df.columns and "rollout_reward_mean" in history_df.columns:
        x = history_df["total_steps"]
        y = history_df["rollout_reward_mean"]
        plt.plot(x, y, color="tab:blue", linewidth=1.8, marker="o", markersize=5)
        if len(history_df) == 1:
            plt.scatter(x, y, color="tab:blue", s=45, zorder=3)
        plt.xlabel("Total Steps")
        plt.ylabel("Mean Rollout Reward")
        plt.title("Training Reward Curve")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(reward_path, dpi=150)
    plt.close()

    loss_cols = [col for col in ["actor_loss", "critic_loss", "entropy"] if col in history_df.columns]
    if not loss_cols:
        return

    plt.figure(figsize=(10, 6))
    x = history_df["total_steps"] if "total_steps" in history_df.columns else history_df.index
    for col in loss_cols:
        plt.plot(x, history_df[col], label=col, marker="o", markersize=5)
        if len(history_df) == 1:
            plt.scatter(x, history_df[col], s=45, zorder=3)
    plt.xlabel("Total Steps")
    plt.ylabel("Value")
    plt.title("Training Diagnostics")
    plt.legend(loc="best")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(losses_path, dpi=150)
    plt.close()


def save_run_artifacts(
    run_dir: str | Path,
    config: Any,
    summary: Mapping[str, Any],
    train_history: Iterable[Dict[str, Any]],
    eval_df: pd.DataFrame,
) -> Dict[str, str]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    summary_path = run_dir / "summary.json"
    history_path = run_dir / "train_history.csv"
    eval_path = run_dir / "eval_trades.csv"
    eval_report_path = run_dir / "eval_report.csv"
    curves_path = run_dir / "test_wealth_paths.png"
    turnover_path = run_dir / "turnover_and_weights.png"
    reward_curve_path = run_dir / "training_reward_curve.png"
    losses_curve_path = run_dir / "training_diagnostics.png"

    write_json(config_path, {"config": _to_serializable(config)})
    write_json(summary_path, dict(summary))
    history_df = write_training_history(history_path, train_history)
    write_dataframe(eval_path, eval_df)
    eval_report = build_eval_report(eval_df)
    write_dataframe(eval_report_path, eval_report)
    plot_eval_curves(eval_report, curves_path)
    plot_turnover(eval_report, turnover_path)
    plot_training_curves(history_df, reward_curve_path, losses_curve_path)

    return {
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "summary_path": str(summary_path),
        "history_path": str(history_path),
        "eval_path": str(eval_path),
        "eval_report_path": str(eval_report_path),
        "curves_path": str(curves_path),
        "turnover_plot_path": str(turnover_path),
        "reward_curve_path": str(reward_curve_path),
        "diagnostics_curve_path": str(losses_curve_path),
    }
