"""Training entrypoint scaffold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kuhn_poker.env import KuhnPokerAECEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Kuhn Poker MaskablePPO policy.")
    parser.add_argument("--total-timesteps", type=int, default=20_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = KuhnPokerAECEnv()
    env.reset()
    print("Training scaffold is ready.")
    print(
        "Next roadmap step: add SB3 + sb3-contrib MaskablePPO loop and PettingZoo wrappers."
    )
    print(f"Requested timesteps: {args.total_timesteps}")


if __name__ == "__main__":
    main()
