from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .models import DirichletActor, MLPBackbone, TemporalCNNBackbone, ValueCritic


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    entropy_coef: float = 1e-3
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    backbone_type: str = "mlp"
    hidden_size: int = 128


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: PPOConfig,
        device: str = "cpu",
        window_size: int | None = None,
        n_assets: int | None = None,
        n_features: int | None = None,
    ):
        self.device = torch.device(device)
        self.config = config
        actor_backbone = self._build_backbone(obs_dim, action_dim, window_size, n_assets, n_features)
        critic_backbone = self._build_backbone(obs_dim, action_dim, window_size, n_assets, n_features)
        self.actor = DirichletActor(actor_backbone, action_dim).to(self.device)
        self.critic = ValueCritic(critic_backbone).to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

    def _build_backbone(
        self,
        obs_dim: int,
        action_dim: int,
        window_size: int | None,
        n_assets: int | None,
        n_features: int | None,
    ):
        backbone_type = self.config.backbone_type.lower()
        if backbone_type == "mlp":
            return MLPBackbone(obs_dim, hidden_sizes=(self.config.hidden_size, self.config.hidden_size))
        if backbone_type == "temporal_cnn":
            if window_size is None or n_assets is None or n_features is None:
                raise ValueError("TemporalCNN backbone requires window_size, n_assets, and n_features.")
            return TemporalCNNBackbone(
                window_size=window_size,
                n_assets=n_assets,
                n_features=n_features,
                action_dim=action_dim,
                hidden_size=self.config.hidden_size,
            )
        raise ValueError(f"Unknown backbone_type: {self.config.backbone_type}")

    def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.actor.distribution(obs_t)
        action_t = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action_t)
        value = self.critic(obs_t)
        return (
            action_t.squeeze(0).detach().cpu().numpy(),
            float(log_prob.item()),
            float(value.item()),
        )

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.actor.distribution(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(obs)
        return log_prob, entropy, values

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray,
        epochs: int = 4,
        batch_size: int = 128,
    ) -> Dict[str, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)

        n = obs_t.size(0)
        stats: Dict[str, float] = {}
        for _ in range(epochs):
            idx = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                batch_idx = idx[start:start + batch_size]
                batch_obs = obs_t[batch_idx]
                batch_actions = actions_t[batch_idx]
                batch_old_log_probs = old_log_probs_t[batch_idx]
                batch_returns = returns_t[batch_idx]
                batch_adv = adv_t[batch_idx]

                new_log_probs, entropy, values = self.evaluate_actions(batch_obs, batch_actions)
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                surrogate1 = ratios * batch_adv
                surrogate2 = torch.clamp(ratios, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * batch_adv
                actor_loss = -torch.min(surrogate1, surrogate2).mean() - self.config.entropy_coef * entropy.mean()
                critic_loss = F.mse_loss(values, batch_returns)

                self.actor_optim.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                (self.config.value_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_optim.step()

                stats = {
                    "actor_loss": float(actor_loss.item()),
                    "critic_loss": float(critic_loss.item()),
                    "entropy": float(entropy.mean().item()),
                }
        return stats


def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_nonterminal = 1.0 - dones[t]
            next_values = next_value
        else:
            next_nonterminal = 1.0 - dones[t]
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return returns.astype(np.float32), advantages.astype(np.float32)


def train_ppo(
    env,
    agent: PPOAgent,
    total_timesteps: int = 2048,
    rollout_steps: int = 256,
    update_epochs: int = 4,
    batch_size: int = 128,
    show_progress: bool = True,
) -> List[Dict[str, float]]:
    history: List[Dict[str, float]] = []
    obs = env.reset()
    total_steps = 0
    update_idx = 0
    total_updates = max(1, int(np.ceil(total_timesteps / rollout_steps)))

    while total_steps < total_timesteps:
        obs_list, act_list, logp_list, rew_list, done_list, val_list = ([] for _ in range(6))

        for _ in range(rollout_steps):
            action, log_prob, value = agent.act(obs, deterministic=False)
            next_obs, reward, done, _ = env.step(action)

            obs_list.append(obs)
            act_list.append(action)
            logp_list.append(log_prob)
            rew_list.append(reward)
            done_list.append(float(done))
            val_list.append(value)

            obs = env.reset() if done else next_obs
            total_steps += 1
            if total_steps >= total_timesteps:
                break

        _, _, next_value = agent.act(obs, deterministic=True)
        obs_arr = np.asarray(obs_list, dtype=np.float32)
        act_arr = np.asarray(act_list, dtype=np.float32)
        logp_arr = np.asarray(logp_list, dtype=np.float32)
        rew_arr = np.asarray(rew_list, dtype=np.float32)
        done_arr = np.asarray(done_list, dtype=np.float32)
        val_arr = np.asarray(val_list, dtype=np.float32)

        returns, advantages = compute_gae(
            rewards=rew_arr,
            dones=done_arr,
            values=val_arr,
            next_value=next_value,
            gamma=agent.config.gamma,
            gae_lambda=agent.config.gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        stats = agent.update(obs_arr, act_arr, logp_arr, returns, advantages, epochs=update_epochs, batch_size=batch_size)
        stats["total_steps"] = total_steps
        stats["rollout_reward_mean"] = float(rew_arr.mean()) if len(rew_arr) else 0.0
        history.append(stats)
        update_idx += 1
        if show_progress:
            pct = 100.0 * min(total_steps, total_timesteps) / max(total_timesteps, 1)
            print(
                f"[train] update {update_idx}/{total_updates} | "
                f"steps {total_steps}/{total_timesteps} ({pct:.1f}%) | "
                f"reward_mean {stats['rollout_reward_mean']:.6f}"
            )

    return history
