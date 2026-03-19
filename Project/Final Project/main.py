from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from .config import load_experiment_config
from .data import build_data_bundle
from .env import PortfolioEnv
from .eval import compute_performance_metrics, evaluate_policy
from .ppo import PPOAgent, train_ppo
from .results import make_run_dir, save_run_artifacts


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(config_path: str | Path, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return _run_experiment_internal(config_path, overrides=overrides, output_base_dir="output")


def _run_experiment_internal(
    config_path: str | Path,
    overrides: Dict[str, Any] | None = None,
    output_base_dir: str | Path = "output",
) -> Dict[str, Any]:
    cfg = load_experiment_config(config_path, overrides=overrides)
    set_seed(cfg.train.seed)

    bundle, split_idx, _, _ = build_data_bundle(
        Path(cfg.data.data_dir),
        cfg.tickers,
        train_ratio=cfg.data.train_ratio,
    )

    train_env = PortfolioEnv(bundle=bundle, start_idx=0, end_idx=split_idx, config=cfg.env)
    eval_env = PortfolioEnv(
        bundle=bundle,
        start_idx=split_idx - cfg.eval_env.window_size,
        end_idx=len(bundle.dates),
        config=cfg.eval_env,
    )

    agent = PPOAgent(
        obs_dim=train_env.obs_dim,
        action_dim=train_env.action_dim,
        config=cfg.ppo,
        window_size=train_env.config.window_size,
        n_assets=train_env.n_assets,
        n_features=train_env.n_features,
    )

    history = train_ppo(
        train_env,
        agent,
        total_timesteps=cfg.train.total_timesteps,
        rollout_steps=cfg.train.rollout_steps,
        update_epochs=cfg.train.update_epochs,
        batch_size=cfg.train.batch_size,
        show_progress=True,
    )
    eval_df = evaluate_policy(eval_env, agent)
    metrics = compute_performance_metrics(eval_df)

    summary = {
        "backbone_type": cfg.ppo.backbone_type,
        "tickers": cfg.tickers,
        "train_steps": cfg.train.total_timesteps,
        "reward_horizon": cfg.env.reward_horizon,
        "turnover_penalty": cfg.env.turnover_penalty,
        "reward_vol_penalty": cfg.env.reward_vol_penalty,
        "eval_rows": int(len(eval_df)),
        "train_history_tail": history[-1] if history else {},
        "metrics": metrics,
    }
    run_dir = make_run_dir(base_dir=output_base_dir, run_name=f"main_{cfg.ppo.backbone_type}")
    artifacts = save_run_artifacts(
        run_dir=run_dir,
        config=cfg,
        summary=summary,
        train_history=history,
        eval_df=eval_df,
    )
    summary["artifacts"] = artifacts
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the portfolio RL main experiment.")
    parser.add_argument(
        "--config",
        default="configs/portfolio_rl_base.json",
        help="Path to the experiment config JSON.",
    )
    parser.add_argument(
        "--backbone",
        choices=["mlp", "temporal_cnn"],
        default=None,
        help="Optional override for model backbone type.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Optional override for total training timesteps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides: Dict[str, Any] = {}
    if args.backbone is not None:
        overrides["model"] = {"backbone_type": args.backbone}
    if args.timesteps is not None:
        overrides.setdefault("train", {})
        overrides["train"]["total_timesteps"] = args.timesteps

    summary = _run_experiment_internal(args.config, overrides=overrides or None, output_base_dir="output")
    print(summary)


if __name__ == "__main__":
    main()
