from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
import json
import random
import shutil
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from offline.market_pipeline import ASSET_UNIVERSES
from offline.ppo_training import PPOTrainingConfig, train_asset_agent


def _risk_search_space(asset_class: str) -> dict:
    if asset_class == "crypto":
        return {
            "drawdown": [0.05, 0.10, 0.20],
            "downside": [0.25, 0.50, 1.00],
            "max_asset_weight": [0.20, 0.25, 0.30, 0.35],
            "target_volatility": [0.025, 0.035, 0.050],
            "target_volatility_penalty": [0.25, 0.50, 1.00],
        }
    if asset_class == "etf":
        return {
            "drawdown": [0.02, 0.05, 0.10],
            "downside": [0.10, 0.25, 0.50],
            "max_asset_weight": [0.25, 0.35, 0.50],
            "target_volatility": [0.008, 0.012, 0.016],
            "target_volatility_penalty": [0.10, 0.25, 0.50],
        }
    return {
        "drawdown": [0.02, 0.05, 0.10],
        "downside": [0.10, 0.25, 0.50],
        "max_asset_weight": [0.25, 0.35, 0.45],
        "target_volatility": [0.010, 0.014, 0.020],
        "target_volatility_penalty": [0.10, 0.25, 0.50],
    }


def _sample_config(
    rng: random.Random,
    *,
    asset_class: str,
    timesteps: int,
    eval_freq: int,
    seed: int,
) -> PPOTrainingConfig:
    n_steps = rng.choice([256, 512, 1024])
    batch_candidates = [64, 128, 256]
    batch_size = rng.choice([candidate for candidate in batch_candidates if candidate <= n_steps])
    layer_options = [(128, 128), (256, 128), (256, 256), (256, 256, 128)]
    risk_space = _risk_search_space(asset_class)
    learning_rate = rng.choice([1e-4, 3e-4, 7e-4])
    clip_range = rng.choice([0.1, 0.2, 0.3])
    learning_rate_schedule = rng.choice(["constant", "linear"])
    clip_range_schedule = rng.choice(["constant", "linear"])
    return PPOTrainingConfig(
        total_timesteps=timesteps,
        eval_freq=eval_freq,
        episode_length=rng.choice([126, 252]),
        window_size=rng.choice([30, 60, 90]),
        eval_ratio=0.2,
        risk_low=0.2,
        risk_high=0.8,
        transaction_fee=rng.choice([0.0005, 0.001, 0.002]),
        turnover_penalty=rng.choice([0.0005, 0.001, 0.0025]),
        variance_penalty_scale=rng.choice([0.5, 1.0, 2.0]),
        concentration_penalty_scale=rng.choice([0.0, 0.01, 0.02]),
        drawdown_penalty_scale=rng.choice(risk_space["drawdown"]),
        downside_penalty_scale=rng.choice(risk_space["downside"]),
        benchmark_reward_scale=rng.choice([0.0, 0.1, 0.25, 0.5]),
        target_daily_volatility=rng.choice(risk_space["target_volatility"]),
        target_volatility_penalty_scale=rng.choice(risk_space["target_volatility_penalty"]),
        max_asset_weight=rng.choice(risk_space["max_asset_weight"]),
        max_cash_weight=rng.choice([0.75, 0.85, 0.95]),
        cash_enabled=True,
        cash_annual_return=0.04,
        reward_scale=100.0,
        learning_rate=learning_rate,
        learning_rate_schedule=learning_rate_schedule,
        learning_rate_final=(
            learning_rate * rng.choice([0.05, 0.10, 0.25])
            if learning_rate_schedule == "linear"
            else None
        ),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=rng.choice([5, 10, 15, 20]),
        gamma=rng.choice([0.98, 0.99, 0.995]),
        gae_lambda=rng.choice([0.92, 0.95, 0.98]),
        clip_range=clip_range,
        clip_range_schedule=clip_range_schedule,
        clip_range_final=(
            clip_range * rng.choice([0.25, 0.50, 0.75])
            if clip_range_schedule == "linear"
            else None
        ),
        ent_coef=rng.choice([0.0, 0.005, 0.01, 0.02]),
        vf_coef=rng.choice([0.4, 0.5, 0.7]),
        normalize_advantage=True,
        use_sde=rng.choice([False, False, True]),
        sde_sample_freq=rng.choice([-1, 4, 8]),
        policy_layers=rng.choice(layer_options),
        activation_fn=rng.choice(["tanh", "relu"]),
        target_kl=rng.choice([0.02, 0.03, 0.05]),
        max_grad_norm=rng.choice([0.3, 0.5, 0.7]),
        n_envs=rng.choice([2, 4]),
        eval_episodes=2,
        seed=seed,
        device="cpu",
    )


def _score_report(report: dict) -> float:
    return (
        float(report["eval_mean_sharpe"])
        + float(report["eval_mean_benchmark_alpha"]) * 50.0
        + float(report["eval_mean_final_value"]) - 1.0
        - float(report["eval_mean_max_drawdown"]) * 2.0
    )


def _evaluate_trial(asset_dir: Path, asset_class: str, config: PPOTrainingConfig) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"stockify_tune_{asset_class}_") as tmp_dir:
        tmp_root = Path(tmp_dir) / "processed"
        tmp_asset_dir = tmp_root / asset_class
        shutil.copytree(asset_dir, tmp_asset_dir)
        report = train_asset_agent(asset_class=asset_class, artifact_root=tmp_root, config=config)
        payload = report.__dict__
        payload["score"] = _score_report(payload)
        payload["config"] = asdict(config)
        return payload


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Random-search tuning for Stockify PPO sub-agents.")
    parser.add_argument(
        "--asset-class",
        action="append",
        choices=sorted(ASSET_UNIVERSES.keys()),
        dest="asset_classes",
        help="Asset class to tune. Defaults to all PPO sub-agents.",
    )
    parser.add_argument(
        "--artifact-root",
        default=str(REPO_ROOT / "artifacts" / "processed"),
        help="Artifact root containing active processed bundles.",
    )
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=8000)
    parser.add_argument("--eval-freq", type=int, default=1000)
    parser.add_argument(
        "--final-timesteps",
        type=int,
        default=25000,
        help="Timesteps for the final retrain when --apply-best is enabled.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--apply-best",
        action="store_true",
        help="Retrain the real asset bundle with the best discovered config.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifact_root = Path(args.artifact_root)
    rng = random.Random(args.seed)
    all_results = {}

    for asset_class in args.asset_classes or sorted(ASSET_UNIVERSES.keys()):
        asset_dir = artifact_root / asset_class
        trials = []
        best_trial = None
        for trial_id in range(args.trials):
            config = _sample_config(
                rng,
                asset_class=asset_class,
                timesteps=args.timesteps,
                eval_freq=args.eval_freq,
                seed=args.seed + trial_id,
            )
            result = _evaluate_trial(asset_dir, asset_class, config)
            result["trial_id"] = trial_id
            trials.append(result)
            if best_trial is None or result["score"] > best_trial["score"]:
                best_trial = result

        assert best_trial is not None
        output = {
            "best_trial": best_trial,
            "trials": trials,
        }
        (asset_dir / "tuning_results.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        all_results[asset_class] = {
            "best_trial_id": best_trial["trial_id"],
            "score": best_trial["score"],
            "eval_mean_final_value": best_trial["eval_mean_final_value"],
            "eval_mean_sharpe": best_trial["eval_mean_sharpe"],
        }

        if args.apply_best:
            best_config_payload = dict(best_trial["config"])
            best_config_payload["total_timesteps"] = args.final_timesteps
            best_config_payload["eval_freq"] = max(1000, min(args.eval_freq * 2, args.final_timesteps // 4))
            best_config = PPOTrainingConfig(**best_config_payload)
            final_report = train_asset_agent(
                asset_class=asset_class,
                artifact_root=artifact_root,
                config=best_config,
            )
            all_results[asset_class]["applied_report"] = final_report.__dict__

    print(json.dumps(all_results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
