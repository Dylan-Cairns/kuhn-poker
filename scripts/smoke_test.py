"""Small end-to-end sanity check."""

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
from kuhn_poker.opponents import sample_random_legal_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run environment smoke test.")
    parser.add_argument("--hands", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def run_hand(env: KuhnPokerAECEnv, rng: np.random.Generator) -> dict[str, float]:
    returns = {agent: 0.0 for agent in AGENT_NAMES}
    env.reset()

    for agent in env.agent_iter(max_iter=10):
        obs, reward, termination, truncation, _ = env.last()
        returns[agent] += reward
        if termination or truncation:
            action = None
        else:
            if int(np.sum(obs["action_mask"])) < 1:
                raise RuntimeError(f"No legal action for live agent {agent}.")
            action = sample_random_legal_action(obs["action_mask"], rng=rng)
        env.step(action)

    total = sum(returns.values())
    if not np.isclose(total, 0.0):
        raise RuntimeError(f"Non zero-sum hand return detected: {returns}")
    return returns


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    env = KuhnPokerAECEnv()

    totals = {agent: 0.0 for agent in AGENT_NAMES}
    for _ in range(args.hands):
        hand_returns = run_hand(env, rng)
        for agent in AGENT_NAMES:
            totals[agent] += hand_returns[agent]

    print(f"Smoke test passed for {args.hands} hands.")
    print(
        f"Average return: {AGENT_NAMES[0]}={totals[AGENT_NAMES[0]] / args.hands:.3f}, "
        f"{AGENT_NAMES[1]}={totals[AGENT_NAMES[1]] / args.hands:.3f}"
    )


if __name__ == "__main__":
    main()
