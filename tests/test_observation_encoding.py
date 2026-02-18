from __future__ import annotations

import numpy as np

from kuhn_poker.constants import AGENT_NAMES, CARD_J, CARD_K, Action
from kuhn_poker.env import KuhnPokerAECEnv
from kuhn_poker.generated.contract import (
    OBS_ACTOR_DIM,
    OBS_ACTOR_OFFSET,
    OBS_HISTORY_DIM,
    OBS_HISTORY_OFFSET,
    OBSERVATION_DIM,
    OBS_PRIVATE_CARD_DIM,
    OBS_PRIVATE_CARD_OFFSET,
)

CARD_SLICE = slice(OBS_PRIVATE_CARD_OFFSET, OBS_PRIVATE_CARD_OFFSET + OBS_PRIVATE_CARD_DIM)
HISTORY_SLICE = slice(OBS_HISTORY_OFFSET, OBS_HISTORY_OFFSET + OBS_HISTORY_DIM)
ACTOR_SLICE = slice(OBS_ACTOR_OFFSET, OBS_ACTOR_OFFSET + OBS_ACTOR_DIM)


def test_observation_layout_at_reset() -> None:
    env = KuhnPokerAECEnv()
    env.reset(seed=0)
    env.private_cards = {AGENT_NAMES[0]: CARD_K, AGENT_NAMES[1]: CARD_J}

    p0_obs = env.observe(AGENT_NAMES[0])
    p1_obs = env.observe(AGENT_NAMES[1])

    assert p0_obs["observation"].shape == (OBSERVATION_DIM,)
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
