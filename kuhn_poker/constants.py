"""Shared constants for Kuhn Poker environment and scripts."""

from enum import IntEnum
from typing import Final

from kuhn_poker.generated.contract import (
    ACTION_DIM as _ACTION_DIM,
    ACTION_ID_BY_NAME,
    CARDS,
    CARD_INDEX_BY_LABEL,
    PLAYERS,
)


class Action(IntEnum):
    CHECK_OR_CALL = ACTION_ID_BY_NAME["CHECK_OR_CALL"]
    BET = ACTION_ID_BY_NAME["BET"]
    FOLD = ACTION_ID_BY_NAME["FOLD"]


ACTION_DIM: Final[int] = _ACTION_DIM
AGENT_NAMES: Final[tuple[str, str]] = (PLAYERS[0], PLAYERS[1])
CARD_LABELS: Final[tuple[str, str, str]] = (CARDS[0], CARDS[1], CARDS[2])

CARD_J: Final[int] = CARD_INDEX_BY_LABEL["J"]
CARD_Q: Final[int] = CARD_INDEX_BY_LABEL["Q"]
CARD_K: Final[int] = CARD_INDEX_BY_LABEL["K"]
