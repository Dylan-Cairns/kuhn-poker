"""Training entrypoint scaffold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sb3_contrib import MaskablePPO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kuhn_poker.wrappers import make_masked_sb3_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Kuhn Poker MaskablePPO policy.")
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("checkpoints/maskable_ppo_kuhn"),
        help="Output path without extension, e.g. checkpoints/maskable_ppo_kuhn",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = make_masked_sb3_env(seed=args.seed)

    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=args.seed,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
    )
    model.learn(total_timesteps=args.total_timesteps)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_out)
    env.close()

    print(f"Training complete. Timesteps: {args.total_timesteps}")
    print(f"Model saved to: {args.model_out}.zip")


if __name__ == "__main__":
    main()
