from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .data import DataBundle


@dataclass
class EnvConfig:
    window_size: int = 20
    episode_length: Optional[int] = 64
    transaction_cost: float = 0.001
    annual_rf_rate: float = 0.04
    turnover_penalty: float = 1e-3
    reward_horizon: int = 1
    reward_vol_penalty: float = 0.0
    reward_vol_lookback: int = 20
    train_mode: bool = True
    seed: int = 42

    @property
    def rf_gross(self) -> float:
        return float((1.0 + self.annual_rf_rate) ** (1.0 / 252.0))


class PortfolioEnv:
    def __init__(self, bundle: DataBundle, start_idx: int, end_idx: int, config: EnvConfig):
        if end_idx - start_idx <= config.window_size + 2:
            raise ValueError("Segment too short for selected window size.")
        self.bundle = bundle
        self.start_idx = int(start_idx)
        self.end_idx = int(end_idx)
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        self.n_assets = bundle.feature_array.shape[1]
        self.n_features = bundle.feature_array.shape[2]
        self.action_dim = self.n_assets + 1
        self.obs_dim = config.window_size * self.n_assets * self.n_features + self.action_dim

        self.current_idx: Optional[int] = None
        self.prev_action: Optional[np.ndarray] = None
        self.wealth: Optional[float] = None
        self.steps = 0
        self.fixed_eval_start = self.start_idx + self.config.window_size - 1
        self.episode_log_returns: List[float] = []

    def _get_observation(self) -> np.ndarray:
        assert self.current_idx is not None
        assert self.prev_action is not None
        start = self.current_idx - self.config.window_size + 1
        end = self.current_idx + 1
        feat_window = self.bundle.feature_array[start:end]
        return np.concatenate([feat_window.reshape(-1), self.prev_action]).astype(np.float32)

    def reset(self) -> np.ndarray:
        min_start = self.start_idx + self.config.window_size - 1
        max_start = self.end_idx - 2
        if self.config.episode_length is not None:
            max_start = min(max_start, self.end_idx - self.config.episode_length - 1)
        if max_start < min_start:
            raise ValueError("No valid reset index. Reduce window_size or episode_length.")

        if self.config.train_mode:
            self.current_idx = int(self.rng.integers(min_start, max_start + 1))
        else:
            self.current_idx = int(self.fixed_eval_start)

        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.prev_action[0] = 1.0
        self.wealth = 1.0
        self.steps = 0
        self.episode_log_returns = []
        return self._get_observation()

    @staticmethod
    def _normalize_action(action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, 1e-8, None)
        return (action / action.sum()).astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, object]]:
        if self.current_idx is None or self.prev_action is None or self.wealth is None:
            raise RuntimeError("Call reset before step.")

        weights = self._normalize_action(action)
        cash_weight = float(weights[0])
        risky_weights = weights[1:]
        gross_vec = self.bundle.next_gross_array[self.current_idx]

        turnover = float(np.abs(weights - self.prev_action).sum())
        trading_cost = self.config.transaction_cost * turnover
        portfolio_gross = float(cash_weight * self.config.rf_gross + np.dot(risky_weights, gross_vec))
        net_gross = max(portfolio_gross - trading_cost, 1e-8)
        log_ret = float(np.log(net_gross))
        excess_log_ret = float(log_ret - np.log(self.config.rf_gross))
        reward = float(
            self._compute_reward(
                weights,
                risky_weights,
                turnover,
                current_idx=self.current_idx,
                one_step_excess=excess_log_ret,
                one_step_log_ret=log_ret,
            )
        )

        self.wealth *= net_gross
        self.prev_action = weights
        self.episode_log_returns.append(log_ret)
        self.current_idx += 1
        self.steps += 1

        done = self.current_idx >= self.end_idx - 1
        if self.config.episode_length is not None and self.steps >= self.config.episode_length:
            done = True

        info: Dict[str, object] = {
            "date": self.bundle.dates[self.current_idx - 1],
            "weights": weights.copy(),
            "gross_vec": gross_vec.copy(),
            "portfolio_gross": portfolio_gross,
            "net_gross": net_gross,
            "log_return": log_ret,
            "excess_log_return": excess_log_ret,
            "turnover": turnover,
            "wealth": float(self.wealth),
            "rf_gross": float(self.config.rf_gross),
            "reward_vol_penalty": float(self.config.reward_vol_penalty),
        }
        next_obs = np.zeros(self.obs_dim, dtype=np.float32) if done else self._get_observation()
        return next_obs, reward, done, info

    def _compute_reward(
        self,
        weights: np.ndarray,
        risky_weights: np.ndarray,
        turnover: float,
        current_idx: int,
        one_step_excess: float,
        one_step_log_ret: float,
    ) -> float:
        if self.config.reward_horizon <= 1:
            base_reward = float(one_step_excess)
        else:
            horizon_end = min(current_idx + self.config.reward_horizon, self.end_idx - 1)
            future_excess = 0.0
            for idx in range(current_idx, horizon_end):
                future_gross_vec = self.bundle.next_gross_array[idx]
                future_gross = float(weights[0] * self.config.rf_gross + np.dot(risky_weights, future_gross_vec))
                future_gross = max(future_gross, 1e-8)
                future_excess += float(np.log(future_gross) - np.log(self.config.rf_gross))
            base_reward = float(future_excess)

        recent = (self.episode_log_returns + [one_step_log_ret])[-self.config.reward_vol_lookback:]
        recent_vol = float(np.std(recent)) if len(recent) >= 2 else 0.0
        return float(
            base_reward
            - self.config.turnover_penalty * turnover
            - self.config.reward_vol_penalty * recent_vol
        )
