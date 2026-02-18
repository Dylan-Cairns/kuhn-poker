from __future__ import annotations

import numpy as np

from kuhn_poker.constants import CARD_J, CARD_K, Action
from kuhn_poker.opponents import sample_random_legal_action, simple_heuristic_action


def test_random_opponent_samples_legal_action() -> None:
    mask = np.array([1, 0, 1], dtype=np.int8)
    action = sample_random_legal_action(mask, np.random.default_rng(1))
    assert mask[action] == 1


def test_heuristic_plays_strong_and_weak_cards() -> None:
    passive_mask = np.array([1, 1, 0], dtype=np.int8)
    facing_bet_mask = np.array([1, 0, 1], dtype=np.int8)

    assert (
        simple_heuristic_action(CARD_K, [], passive_mask) == int(Action.BET)
    )
    assert (
        simple_heuristic_action(CARD_J, ["bet"], facing_bet_mask) == int(Action.FOLD)
    )
