from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import load_experiment_config
from .data import build_data_bundle
from .env import PortfolioEnv
from .eval import compute_performance_metrics, evaluate_policy
from .ppo import PPOAgent, train_ppo
from .results import make_run_dir, save_run_artifacts


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_minimal_experiment() -> dict:
    cfg = load_experiment_config("configs/portfolio_rl_base.json")
    set_seed(cfg.train.seed)
    bundle, split_idx, _, _ = build_data_bundle(Path(cfg.data.data_dir), cfg.tickers, train_ratio=cfg.data.train_ratio)

    train_env = PortfolioEnv(
        bundle=bundle,
        start_idx=0,
        end_idx=split_idx,
        config=cfg.env,
    )
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
    result = {
        "backbone_type": cfg.ppo.backbone_type,
        "reward_horizon": cfg.env.reward_horizon,
        "turnover_penalty": cfg.env.turnover_penalty,
        "reward_vol_penalty": cfg.env.reward_vol_penalty,
        "train_history_tail": history[-1] if history else {},
        "eval_rows": int(len(eval_df)),
        "metrics": metrics,
    }
    run_dir = make_run_dir(run_name=f"minimal_{cfg.ppo.backbone_type}")
    artifact_paths = save_run_artifacts(
        run_dir=run_dir,
        config=cfg,
        summary=result,
        train_history=history,
        eval_df=eval_df,
    )
    result["artifacts"] = artifact_paths
    return result


if __name__ == "__main__":
    result = run_minimal_experiment()
    print(result)
