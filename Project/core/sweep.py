from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .main import _run_experiment_internal
from .results import make_run_dir, write_dataframe, write_json


def load_matrix_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_research_matrix(
    config_path: str,
    backbones: List[str],
    seeds: List[int],
    timesteps: int | List[int] | None = None,
    turnover_penalties: List[float] | None = None,
    reward_horizons: List[int] | None = None,
    reward_vol_penalties: List[float] | None = None,
) -> Dict[str, Any]:
    sweep_dir = make_run_dir(run_name="research_matrix")
    rows: List[Dict[str, Any]] = []
    timesteps_list = timesteps if isinstance(timesteps, list) else [timesteps]
    turnover_penalties = [None] if turnover_penalties is None else turnover_penalties
    reward_horizons = [None] if reward_horizons is None else reward_horizons
    reward_vol_penalties = [None] if reward_vol_penalties is None else reward_vol_penalties

    for backbone in backbones:
        for seed in seeds:
            for train_steps in timesteps_list:
                for turnover_penalty in turnover_penalties:
                    for reward_horizon in reward_horizons:
                        for reward_vol_penalty in reward_vol_penalties:
                            overrides: Dict[str, Any] = {
                                "model": {"backbone_type": backbone},
                                "train": {"seed": seed},
                                "env": {"seed": seed},
                                "eval_env": {"seed": seed},
                            }
                            if train_steps is not None:
                                overrides["train"]["total_timesteps"] = train_steps
                            if turnover_penalty is not None:
                                overrides["env"]["turnover_penalty"] = turnover_penalty
                                overrides["eval_env"]["turnover_penalty"] = turnover_penalty
                            if reward_horizon is not None:
                                overrides["env"]["reward_horizon"] = reward_horizon
                                overrides["eval_env"]["reward_horizon"] = reward_horizon
                            if reward_vol_penalty is not None:
                                overrides["env"]["reward_vol_penalty"] = reward_vol_penalty
                                overrides["eval_env"]["reward_vol_penalty"] = reward_vol_penalty

                            print(
                                f"[sweep] start backbone={backbone} seed={seed} "
                                f"timesteps={overrides['train'].get('total_timesteps')} "
                                f"turnover_penalty={overrides['env'].get('turnover_penalty')} "
                                f"reward_horizon={overrides['env'].get('reward_horizon')} "
                                f"reward_vol_penalty={overrides['env'].get('reward_vol_penalty')}"
                            )
                            summary = _run_experiment_internal(
                                config_path,
                                overrides=overrides,
                                output_base_dir=sweep_dir,
                            )
                            print(
                                f"[sweep] done  backbone={backbone} seed={seed} | "
                                f"h={summary['reward_horizon']} | tp={summary['turnover_penalty']} | "
                                f"vp={summary['reward_vol_penalty']} | "
                                f"sharpe={summary['metrics']['sharpe_ratio']:.4f} | "
                                f"wealth={summary['metrics']['final_wealth']:.4f}"
                            )
                            rows.append(
                                {
                                    "backbone_type": backbone,
                                    "seed": seed,
                                    "train_steps": summary["train_steps"],
                                    "reward_horizon": summary["reward_horizon"],
                                    "turnover_penalty": summary["turnover_penalty"],
                                    "reward_vol_penalty": summary["reward_vol_penalty"],
                                    **summary["metrics"],
                                    "run_dir": summary["artifacts"]["run_dir"],
                                    "summary_path": summary["artifacts"]["summary_path"],
                                }
                            )

    summary_df = pd.DataFrame(rows)
    write_dataframe(Path(sweep_dir) / "matrix_results.csv", summary_df)
    agg_df = (
        summary_df.groupby(["backbone_type", "train_steps", "reward_horizon", "turnover_penalty", "reward_vol_penalty"])[["final_wealth", "annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown", "avg_turnover"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg_df.columns = ["_".join(col).strip("_") for col in agg_df.columns.to_flat_index()]
    write_dataframe(Path(sweep_dir) / "matrix_aggregate.csv", agg_df)
    meta = {
        "config_path": config_path,
        "backbones": backbones,
        "seeds": seeds,
        "timesteps": timesteps_list,
        "turnover_penalties": turnover_penalties,
        "reward_horizons": reward_horizons,
        "reward_vol_penalties": reward_vol_penalties,
        "sweep_dir": str(sweep_dir),
    }
    write_json(Path(sweep_dir) / "matrix_meta.json", meta)
    return {"sweep_dir": str(sweep_dir), "rows": len(summary_df), "aggregate_rows": len(agg_df)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an automated research matrix.")
    parser.add_argument("--config", default="configs/portfolio_rl_base.json")
    parser.add_argument(
        "--matrix-config",
        default=None,
        help="Optional JSON file containing experiment_config, backbones, seeds, and timesteps.",
    )
    parser.add_argument("--backbones", nargs="+", default=["mlp", "temporal_cnn"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 19])
    parser.add_argument("--timesteps", nargs="+", type=int, default=[1024])
    parser.add_argument("--turnover-penalties", nargs="+", type=float, default=None)
    parser.add_argument("--reward-horizons", nargs="+", type=int, default=None)
    parser.add_argument("--reward-vol-penalties", nargs="+", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config
    backbones = list(args.backbones)
    seeds = list(args.seeds)
    timesteps = args.timesteps

    if args.matrix_config is not None:
        matrix_cfg = load_matrix_config(args.matrix_config)
        config_path = matrix_cfg.get("experiment_config", config_path)
        backbones = list(matrix_cfg.get("backbones", backbones))
        seeds = list(matrix_cfg.get("seeds", seeds))
        timesteps = matrix_cfg.get("timesteps", timesteps)
        turnover_penalties = matrix_cfg.get("turnover_penalties", args.turnover_penalties)
        reward_horizons = matrix_cfg.get("reward_horizons", args.reward_horizons)
        reward_vol_penalties = matrix_cfg.get("reward_vol_penalties", args.reward_vol_penalties)
    else:
        turnover_penalties = args.turnover_penalties
        reward_horizons = args.reward_horizons
        reward_vol_penalties = args.reward_vol_penalties

    result = run_research_matrix(
        config_path=config_path,
        backbones=backbones,
        seeds=seeds,
        timesteps=timesteps,
        turnover_penalties=turnover_penalties,
        reward_horizons=reward_horizons,
        reward_vol_penalties=reward_vol_penalties,
    )
    print(result)


if __name__ == "__main__":
    main()
