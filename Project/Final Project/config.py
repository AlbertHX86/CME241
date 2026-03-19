from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .env import EnvConfig
from .ppo import PPOConfig


@dataclass
class TrainConfig:
    total_timesteps: int
    rollout_steps: int
    update_epochs: int
    batch_size: int
    seed: int


@dataclass
class DataConfig:
    data_dir: str
    train_ratio: float


@dataclass
class ExperimentConfig:
    tickers: List[str]
    data: DataConfig
    env: EnvConfig
    eval_env: EnvConfig
    ppo: PPOConfig
    train: TrainConfig


def _merge_dict(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_experiment_config(path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if overrides:
        raw = _merge_dict(raw, overrides)

    data_cfg = DataConfig(**raw["data"])
    env_cfg = EnvConfig(train_mode=True, **raw["env"])
    eval_env_cfg = EnvConfig(train_mode=False, **raw["eval_env"])
    ppo_kwargs = dict(raw["ppo"])
    ppo_kwargs.update(raw.get("model", {}))
    ppo_cfg = PPOConfig(**ppo_kwargs)
    train_cfg = TrainConfig(**raw["train"])
    return ExperimentConfig(
        tickers=list(raw["tickers"]),
        data=data_cfg,
        env=env_cfg,
        eval_env=eval_env_cfg,
        ppo=ppo_cfg,
        train=train_cfg,
    )
