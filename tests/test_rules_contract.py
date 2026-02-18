from __future__ import annotations

import numpy as np
import pytest

from kuhn_poker.constants import AGENT_NAMES, CARD_J, CARD_K, Action
from kuhn_poker.env import KuhnPokerAECEnv


def _run_hand(cards: tuple[int, int], actions: list[int]) -> tuple[dict[str, float], list[str]]:
    env = KuhnPokerAECEnv()
    env.reset(seed=0)
    env.private_cards = {
        AGENT_NAMES[0]: cards[0],
        AGENT_NAMES[1]: cards[1],
    }

    for action in actions:
        env.step(action)

    assert all(env.terminations.values())
    return env.rewards.copy(), env.history[:]


@pytest.mark.parametrize(
    "cards,actions,expected_history,expected_rewards",
    [
        (
            (CARD_K, CARD_J),
            [int(Action.CHECK_OR_CALL), int(Action.CHECK_OR_CALL)],
            ["check", "check"],
            {AGENT_NAMES[0]: 1.0, AGENT_NAMES[1]: -1.0},
        ),
        (
            (CARD_J, CARD_K),
            [int(Action.BET_OR_RAISE), int(Action.FOLD)],
            ["bet", "fold"],
            {AGENT_NAMES[0]: 1.0, AGENT_NAMES[1]: -1.0},
        ),
        (
            (CARD_J, CARD_K),
            [int(Action.BET_OR_RAISE), int(Action.CHECK_OR_CALL)],
            ["bet", "call"],
            {AGENT_NAMES[0]: -2.0, AGENT_NAMES[1]: 2.0},
        ),
        (
            (CARD_K, CARD_J),
            [
                int(Action.CHECK_OR_CALL),
                int(Action.BET_OR_RAISE),
                int(Action.CHECK_OR_CALL),
            ],
            ["check", "bet", "call"],
            {AGENT_NAMES[0]: 2.0, AGENT_NAMES[1]: -2.0},
        ),
        (
            (CARD_K, CARD_J),
            [
                int(Action.CHECK_OR_CALL),
                int(Action.BET_OR_RAISE),
                int(Action.FOLD),
            ],
            ["check", "bet", "fold"],
            {AGENT_NAMES[0]: -1.0, AGENT_NAMES[1]: 1.0},
        ),
    ],
)
def test_terminal_histories_and_payoffs(
    cards: tuple[int, int],
    actions: list[int],
    expected_history: list[str],
    expected_rewards: dict[str, float],
) -> None:
    rewards, history = _run_hand(cards=cards, actions=actions)
    assert history == expected_history
    assert rewards == expected_rewards
    assert np.isclose(sum(rewards.values()), 0.0)


def test_illegal_fold_before_bet_raises() -> None:
    env = KuhnPokerAECEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError, match="Illegal action"):
        env.step(int(Action.FOLD))


def test_legal_masks_match_betting_state() -> None:
    env = KuhnPokerAECEnv()
    env.reset(seed=0)

    obs = env.observe(AGENT_NAMES[0])
    assert np.array_equal(obs["action_mask"], np.array([1, 1, 0], dtype=np.int8))

    env.step(int(Action.CHECK_OR_CALL))
    obs = env.observe(AGENT_NAMES[1])
    assert np.array_equal(obs["action_mask"], np.array([1, 1, 0], dtype=np.int8))

    env.step(int(Action.BET_OR_RAISE))
    obs = env.observe(AGENT_NAMES[0])
    assert np.array_equal(obs["action_mask"], np.array([1, 0, 1], dtype=np.int8))
