"""Evaluation scaffold for baseline opponents."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kuhn_poker.constants import AGENT_NAMES
from kuhn_poker.env import KuhnPokerAECEnv
from kuhn_poker.opponents import sample_random_legal_action, simple_heuristic_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline Kuhn Poker evaluation scaffold.")
    parser.add_argument("--hands", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def play_hand(env: KuhnPokerAECEnv, rng: np.random.Generator) -> dict[str, float]:
    returns = {agent: 0.0 for agent in AGENT_NAMES}
    env.reset()

    for agent in env.agent_iter(max_iter=10):
        obs, reward, termination, truncation, _ = env.last()
        returns[agent] += reward

        if termination or truncation:
            action = None
        elif agent == AGENT_NAMES[0]:
            action = simple_heuristic_action(
                private_card=env.private_cards[agent],
                public_history=env.history,
                action_mask=obs["action_mask"],
            )
        else:
            action = sample_random_legal_action(obs["action_mask"], rng=rng)

        env.step(action)

    return returns


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    env = KuhnPokerAECEnv()

    aggregate = {agent: 0.0 for agent in AGENT_NAMES}
    for _ in range(args.hands):
        result = play_hand(env, rng)
        for agent in AGENT_NAMES:
            aggregate[agent] += result[agent]

    print(f"Hands: {args.hands}")
    print(
        f"{AGENT_NAMES[0]} average return (heuristic): "
        f"{aggregate[AGENT_NAMES[0]] / args.hands:.3f}"
    )
    print(
        f"{AGENT_NAMES[1]} average return (random-legal): "
        f"{aggregate[AGENT_NAMES[1]] / args.hands:.3f}"
    )


if __name__ == "__main__":
    main()
