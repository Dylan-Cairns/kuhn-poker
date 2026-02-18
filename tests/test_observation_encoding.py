from __future__ import annotations

import numpy as np

from kuhn_poker.constants import AGENT_NAMES, CARD_J, CARD_K, Action
from kuhn_poker.env import KuhnPokerAECEnv

CARD_SLICE = slice(0, 3)
HISTORY_SLICE = slice(3, 8)
ACTOR_SLICE = slice(8, 10)


def test_observation_layout_at_reset() -> None:
    env = KuhnPokerAECEnv()
    env.reset(seed=0)
    env.private_cards = {AGENT_NAMES[0]: CARD_K, AGENT_NAMES[1]: CARD_J}

    p0_obs = env.observe(AGENT_NAMES[0])
    p1_obs = env.observe(AGENT_NAMES[1])

    assert p0_obs["observation"].shape == (10,)
    assert np.array_equal(p0_obs["observation"][CARD_SLICE], np.array([0, 0, 1], dtype=np.int8))
    assert np.array_equal(p1_obs["observation"][CARD_SLICE], np.array([1, 0, 0], dtype=np.int8))
    assert np.array_equal(
        p0_obs["observation"][HISTORY_SLICE], np.array([1, 0, 0, 0, 0], dtype=np.int8)
    )
    assert np.array_equal(
        p0_obs["observation"][ACTOR_SLICE], np.array([1, 0], dtype=np.int8)
    )
    assert np.array_equal(p0_obs["action_mask"], np.array([1, 1, 0], dtype=np.int8))
    assert np.array_equal(p1_obs["action_mask"], np.array([0, 0, 0], dtype=np.int8))


def test_observation_layout_for_check_bet_response() -> None:
    env = KuhnPokerAECEnv()
    env.reset(seed=0)

    env.step(int(Action.CHECK_OR_CALL))
    p1_obs = env.observe(AGENT_NAMES[1])
    assert np.array_equal(
        p1_obs["observation"][HISTORY_SLICE], np.array([0, 1, 0, 0, 0], dtype=np.int8)
    )
    assert np.array_equal(p1_obs["observation"][ACTOR_SLICE], np.array([0, 1], dtype=np.int8))
    assert np.array_equal(p1_obs["action_mask"], np.array([1, 1, 0], dtype=np.int8))

    env.step(int(Action.BET))
    p0_obs = env.observe(AGENT_NAMES[0])
    assert np.array_equal(
        p0_obs["observation"][HISTORY_SLICE], np.array([0, 0, 0, 1, 0], dtype=np.int8)
    )
    assert np.array_equal(p0_obs["observation"][ACTOR_SLICE], np.array([1, 0], dtype=np.int8))
    assert np.array_equal(p0_obs["action_mask"], np.array([1, 0, 1], dtype=np.int8))


def test_terminal_observation_uses_terminal_history_and_no_actor() -> None:
    env = KuhnPokerAECEnv()
    env.reset(seed=0)

    env.step(int(Action.CHECK_OR_CALL))
    env.step(int(Action.BET))
    env.step(int(Action.FOLD))

    for agent in AGENT_NAMES:
        obs = env.observe(agent)
        assert np.array_equal(
            obs["observation"][HISTORY_SLICE], np.array([0, 0, 0, 0, 1], dtype=np.int8)
        )
        assert np.array_equal(obs["observation"][ACTOR_SLICE], np.array([0, 0], dtype=np.int8))
        assert np.array_equal(obs["action_mask"], np.array([0, 0, 0], dtype=np.int8))
