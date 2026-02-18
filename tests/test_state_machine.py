from __future__ import annotations

from kuhn_poker.constants import AGENT_NAMES, Action
from kuhn_poker.env import HandPhase, KuhnPokerAECEnv


def test_reset_starts_at_p0_act_phase() -> None:
    env = KuhnPokerAECEnv()
    env.reset(seed=0)

    assert env.phase == HandPhase.P0_ACT
    assert env.agent_selection == AGENT_NAMES[0]


def test_check_bet_call_phase_path() -> None:
    env = KuhnPokerAECEnv()
    env.reset(seed=0)

    env.step(int(Action.CHECK_OR_CALL))
    assert env.phase == HandPhase.P1_ACT
    assert env.agent_selection == AGENT_NAMES[1]

    env.step(int(Action.BET_OR_RAISE))
    assert env.phase == HandPhase.P0_RESPONSE
    assert env.agent_selection == AGENT_NAMES[0]

    env.step(int(Action.CHECK_OR_CALL))
    assert env.phase == HandPhase.TERMINAL
    assert all(env.terminations.values())


def test_bet_fold_phase_path() -> None:
    env = KuhnPokerAECEnv()
    env.reset(seed=0)

    env.step(int(Action.BET_OR_RAISE))
    assert env.phase == HandPhase.P1_RESPONSE
    assert env.agent_selection == AGENT_NAMES[1]

    env.step(int(Action.FOLD))
    assert env.phase == HandPhase.TERMINAL
    assert all(env.terminations.values())
