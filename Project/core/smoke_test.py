from __future__ import annotations

from pathlib import Path

from .config import load_experiment_config
from .data import build_data_bundle
from .env import PortfolioEnv
from .eval import compute_performance_metrics, evaluate_policy
from .ppo import PPOAgent, train_ppo


def run_smoke_test(backbone_type: str = "mlp") -> dict:
    cfg = load_experiment_config(
        "configs/portfolio_rl_base.json",
        overrides={
            "model": {"backbone_type": backbone_type, "hidden_size": 64},
            "train": {"total_timesteps": 256, "rollout_steps": 64, "update_epochs": 1, "batch_size": 32, "seed": 7},
            "env": {"episode_length": 32},
            "eval_env": {"episode_length": None}
        },
    )
    bundle, split_idx, _, _ = build_data_bundle(Path(cfg.data.data_dir), cfg.tickers, train_ratio=cfg.data.train_ratio)
    assert bundle.feature_array.ndim == 3
    assert bundle.next_gross_array.shape[1] == len(cfg.tickers)

    train_env = PortfolioEnv(
        bundle=bundle,
        start_idx=0,
        end_idx=split_idx,
        config=cfg.env,
    )
    obs = train_env.reset()
    assert obs.shape == (train_env.obs_dim,)

    agent = PPOAgent(
        obs_dim=train_env.obs_dim,
        action_dim=train_env.action_dim,
        config=cfg.ppo,
        window_size=train_env.config.window_size,
        n_assets=train_env.n_assets,
        n_features=train_env.n_features,
    )
    action, _, _ = agent.act(obs, deterministic=False)
    next_obs, reward, done, info = train_env.step(action)
    assert next_obs.shape == (train_env.obs_dim,)
    assert isinstance(reward, float)
    assert "excess_log_return" in info
    assert abs(action.sum() - 1.0) < 1e-3

    history = train_ppo(
        train_env,
        agent,
        total_timesteps=cfg.train.total_timesteps,
        rollout_steps=cfg.train.rollout_steps,
        update_epochs=cfg.train.update_epochs,
        batch_size=cfg.train.batch_size,
    )
    assert len(history) > 0

    eval_env = PortfolioEnv(
        bundle=bundle,
        start_idx=split_idx - cfg.eval_env.window_size,
        end_idx=len(bundle.dates),
        config=cfg.eval_env,
    )
    eval_df = evaluate_policy(eval_env, agent)
    metrics = compute_performance_metrics(eval_df)
    assert len(eval_df) > 0
    assert "final_wealth" in metrics
    return {"backbone_type": backbone_type, "history_tail": history[-1], "metrics": metrics, "rows": len(eval_df)}


if __name__ == "__main__":
    print(run_smoke_test("mlp"))
    print(run_smoke_test("temporal_cnn"))
