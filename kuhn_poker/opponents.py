"""Baseline opponents for sanity checks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np

from kuhn_poker.constants import CARD_J, CARD_K, Action


def sample_random_legal_action(
    action_mask: np.ndarray, rng: Optional[np.random.Generator] = None
) -> int:
    """Sample uniformly from legal actions."""
    if rng is None:
        rng = np.random.default_rng()

    legal_actions = np.flatnonzero(action_mask)
    if len(legal_actions) == 0:
        raise ValueError("No legal actions available.")
    return int(rng.choice(legal_actions))


def simple_heuristic_action(
    private_card: int, public_history: Sequence[str], action_mask: np.ndarray
) -> int:
    """A tiny baseline strategy for quick checks."""
    legal_actions = np.flatnonzero(action_mask)
    if len(legal_actions) == 0:
        raise ValueError("No legal actions available.")

    facing_bet = tuple(public_history) in (("bet",), ("check", "bet"))

    if facing_bet:
        if private_card == CARD_J and action_mask[Action.FOLD] == 1:
            return int(Action.FOLD)
        if action_mask[Action.CHECK_OR_CALL] == 1:
            return int(Action.CHECK_OR_CALL)
        return int(legal_actions[0])

    if private_card == CARD_K and action_mask[Action.BET] == 1:
        return int(Action.BET)
    if action_mask[Action.CHECK_OR_CALL] == 1:
        return int(Action.CHECK_OR_CALL)
    return int(legal_actions[0])
