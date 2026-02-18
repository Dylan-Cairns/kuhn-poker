"""Shared constants for Kuhn Poker environment and scripts."""

from enum import IntEnum
from typing import Final


class Action(IntEnum):
    CHECK_OR_CALL = 0
    BET = 1
    FOLD = 2


ACTION_DIM: Final[int] = 3
AGENT_NAMES: Final[tuple[str, str]] = ("player_0", "player_1")
CARD_LABELS: Final[tuple[str, str, str]] = ("J", "Q", "K")

CARD_J: Final[int] = 0
CARD_Q: Final[int] = 1
CARD_K: Final[int] = 2
