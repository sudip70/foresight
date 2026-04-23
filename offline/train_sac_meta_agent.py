from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from offline.sac_training import SACMetaTrainingConfig, train_meta_agent


def _parse_layers(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_risk_values(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Retrain Stockify SAC meta-allocation agent.")
    parser.add_argument(
        "--artifact-root",
        default=str(REPO_ROOT / "artifacts" / "processed"),
        help="Artifact root containing PPO sub-agent bundles and meta artifacts.",
    )
    parser.add_argument("--total-timesteps", type=int, default=150_000)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--episode-length", type=int, default=252)
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--risk-values", default="0.3,0.5,0.7")
    parser.add_argument("--transaction-fee", type=float, default=0.001)
    parser.add_argument("--turnover-penalty", type=float, default=0.001)
    parser.add_argument("--volatility-penalty-scale", type=float, default=0.5)
    parser.add_argument("--concentration-penalty-scale", type=float, default=0.01)
    parser.add_argument("--drawdown-penalty-scale", type=float, default=0.05)
    parser.add_argument("--downside-penalty-scale", type=float, default=0.25)
    parser.add_argument("--benchmark-reward-scale", type=float, default=0.5)
    parser.add_argument("--target-daily-volatility", type=float, default=0.014)
    parser.add_argument("--target-volatility-penalty-scale", type=float, default=0.25)
    parser.add_argument("--reward-scale", type=float, default=100.0)
    parser.add_argument("--max-asset-weight", type=float, default=0.20)
    parser.add_argument("--max-stock-weight", type=float, default=0.85)
    parser.add_argument("--max-crypto-weight", type=float, default=0.30)
    parser.add_argument("--max-etf-weight", type=float, default=0.70)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--ent-coef", default="auto")
    parser.add_argument("--policy-layers", default="256,256")
    parser.add_argument(
        "--activation-fn",
        choices=["relu", "tanh", "elu", "leaky_relu"],
        default="relu",
    )
    parser.add_argument("--use-sde", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ent_coef: str | float
    try:
        ent_coef = float(args.ent_coef)
    except ValueError:
        ent_coef = args.ent_coef

    report = train_meta_agent(
        artifact_root=Path(args.artifact_root),
        config=SACMetaTrainingConfig(
            total_timesteps=args.total_timesteps,
            eval_freq=args.eval_freq,
            episode_length=args.episode_length,
            window_size=args.window_size,
            eval_ratio=args.eval_ratio,
            risk_values=_parse_risk_values(args.risk_values),
            transaction_fee=args.transaction_fee,
            turnover_penalty=args.turnover_penalty,
            volatility_penalty_scale=args.volatility_penalty_scale,
            concentration_penalty_scale=args.concentration_penalty_scale,
            drawdown_penalty_scale=args.drawdown_penalty_scale,
            downside_penalty_scale=args.downside_penalty_scale,
            benchmark_reward_scale=args.benchmark_reward_scale,
            target_daily_volatility=args.target_daily_volatility,
            target_volatility_penalty_scale=args.target_volatility_penalty_scale,
            reward_scale=args.reward_scale,
            max_asset_weight=args.max_asset_weight,
            max_stock_weight=args.max_stock_weight,
            max_crypto_weight=args.max_crypto_weight,
            max_etf_weight=args.max_etf_weight,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            ent_coef=ent_coef,
            policy_layers=_parse_layers(args.policy_layers),
            activation_fn=args.activation_fn,
            use_sde=args.use_sde,
            seed=args.seed,
            device=args.device,
        ),
    )
    print(json.dumps(report.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
