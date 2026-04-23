from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from offline.market_pipeline import ASSET_UNIVERSES
from offline.ppo_training import PPOTrainingConfig, train_asset_agent


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Retrain Stockify PPO sub-agents on refreshed OHLCV features.")
    parser.add_argument(
        "--asset-class",
        action="append",
        choices=sorted(ASSET_UNIVERSES.keys()),
        dest="asset_classes",
        help="Asset class to train. Defaults to all PPO sub-agents.",
    )
    parser.add_argument(
        "--artifact-root",
        default=str(REPO_ROOT / "artifacts" / "processed"),
        help="Artifact root containing the per-asset processed bundles.",
    )
    parser.add_argument("--total-timesteps", type=int, default=25_000)
    parser.add_argument("--eval-freq", type=int, default=2_500)
    parser.add_argument("--episode-length", type=int, default=252)
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--risk-low", type=float, default=0.2)
    parser.add_argument("--risk-high", type=float, default=0.8)
    parser.add_argument("--transaction-fee", type=float, default=0.001)
    parser.add_argument("--turnover-penalty", type=float, default=0.001)
    parser.add_argument("--variance-penalty-scale", type=float, default=1.0)
    parser.add_argument("--concentration-penalty-scale", type=float, default=0.01)
    parser.add_argument("--drawdown-penalty-scale", type=float, default=0.02)
    parser.add_argument("--downside-penalty-scale", type=float, default=0.25)
    parser.add_argument("--benchmark-reward-scale", type=float, default=0.25)
    parser.add_argument("--target-daily-volatility", type=float)
    parser.add_argument("--target-volatility-penalty-scale", type=float, default=0.0)
    parser.add_argument("--max-asset-weight", type=float)
    parser.add_argument("--reward-scale", type=float, default=100.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--learning-rate-schedule",
        choices=["constant", "linear"],
        default="constant",
    )
    parser.add_argument("--learning-rate-final", type=float)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument(
        "--clip-range-schedule",
        choices=["constant", "linear"],
        default="constant",
    )
    parser.add_argument("--clip-range-final", type=float)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--no-normalize-advantage", action="store_true")
    parser.add_argument("--use-sde", action="store_true")
    parser.add_argument("--sde-sample-freq", type=int, default=-1)
    parser.add_argument(
        "--policy-layers",
        default="128,128",
        help="Comma-separated hidden layer sizes for the PPO policy/value networks.",
    )
    parser.add_argument(
        "--activation-fn",
        choices=["tanh", "relu", "elu", "leaky_relu"],
        default="tanh",
        help="Activation function for PPO policy/value MLPs.",
    )
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--no-orthogonal-init", action="store_true")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--min-feature-variance", type=float, default=1e-10)
    parser.add_argument("--feature-clip", type=float, default=5.0)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PPOTrainingConfig(
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq,
        episode_length=args.episode_length,
        window_size=args.window_size,
        eval_ratio=args.eval_ratio,
        risk_low=args.risk_low,
        risk_high=args.risk_high,
        transaction_fee=args.transaction_fee,
        turnover_penalty=args.turnover_penalty,
        variance_penalty_scale=args.variance_penalty_scale,
        concentration_penalty_scale=args.concentration_penalty_scale,
        drawdown_penalty_scale=args.drawdown_penalty_scale,
        downside_penalty_scale=args.downside_penalty_scale,
        benchmark_reward_scale=args.benchmark_reward_scale,
        target_daily_volatility=args.target_daily_volatility,
        target_volatility_penalty_scale=args.target_volatility_penalty_scale,
        max_asset_weight=args.max_asset_weight,
        reward_scale=args.reward_scale,
        learning_rate=args.learning_rate,
        learning_rate_schedule=args.learning_rate_schedule,
        learning_rate_final=args.learning_rate_final,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        clip_range_schedule=args.clip_range_schedule,
        clip_range_final=args.clip_range_final,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        normalize_advantage=not args.no_normalize_advantage,
        use_sde=args.use_sde,
        sde_sample_freq=args.sde_sample_freq,
        policy_layers=tuple(
            int(part.strip()) for part in args.policy_layers.split(",") if part.strip()
        ),
        activation_fn=args.activation_fn,
        orthogonal_init=not args.no_orthogonal_init,
        target_kl=args.target_kl,
        max_grad_norm=args.max_grad_norm,
        n_envs=max(args.n_envs, 1),
        min_feature_variance=args.min_feature_variance,
        feature_clip=args.feature_clip,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        device=args.device,
    )
    reports = []
    artifact_root = Path(args.artifact_root)
    for asset_class in args.asset_classes or sorted(ASSET_UNIVERSES.keys()):
        reports.append(
            train_asset_agent(
                asset_class=asset_class,
                artifact_root=artifact_root,
                config=config,
            ).__dict__
        )
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
