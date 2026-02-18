"""PettingZoo AEC environment scaffold for 2-player Kuhn Poker."""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import AECEnv

from kuhn_poker.constants import ACTION_DIM, AGENT_NAMES, CARD_LABELS, Action

_HISTORY_TO_INDEX = {
    (): 0,
    ("check",): 1,
    ("bet",): 2,
    ("check", "bet"): 3,
}
_TERMINAL_HISTORY_INDEX = 4
_PRIVATE_CARD_DIM = len(CARD_LABELS)
_HISTORY_DIM = 5
_ACTOR_DIM = len(AGENT_NAMES)
_OBS_SIZE = _PRIVATE_CARD_DIM + _HISTORY_DIM + _ACTOR_DIM

_PRIVATE_CARD_OFFSET = 0
_HISTORY_OFFSET = _PRIVATE_CARD_OFFSET + _PRIVATE_CARD_DIM
_ACTOR_OFFSET = _HISTORY_OFFSET + _HISTORY_DIM


class HandPhase(str, Enum):
    """Explicit phase machine for one Kuhn hand."""

    DEAL = "deal"
    P0_ACT = "p0_act"
    P1_ACT = "p1_act"
    P0_RESPONSE = "p0_response"
    P1_RESPONSE = "p1_response"
    TERMINAL = "terminal"


class KuhnPokerAECEnv(AECEnv):
    """Minimal AEC environment with one hand per episode."""

    metadata = {
        "name": "kuhn_poker_v0",
        "is_parallelizable": False,
        "render_modes": ["human"],
    }

    def __init__(self, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.possible_agents = list(AGENT_NAMES)
        self.agents: list[str] = []

        self._action_spaces = {
            agent: spaces.Discrete(ACTION_DIM) for agent in self.possible_agents
        }
        self._observation_spaces = {
            agent: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(_OBS_SIZE,), dtype=np.int8
                    ),
                    "action_mask": spaces.MultiBinary(ACTION_DIM),
                }
            )
            for agent in self.possible_agents
        }

        self.np_random, _ = seeding.np_random(None)

        self.private_cards: dict[str, int] = {}
        self.history: list[str] = []
        self.last_bettor: Optional[str] = None
        self.contributions: dict[str, int] = {}
        self.phase = HandPhase.DEAL

        self.rewards: dict[str, float] = {}
        self._cumulative_rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict[str, object]] = {}

        self.agent_selection = self.possible_agents[0]

    def observation_space(self, agent: str) -> spaces.Space:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self._action_spaces[agent]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> None:
        del options
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        dealt = self.np_random.permutation(len(CARD_LABELS))[:2]
        self.private_cards = {
            self.possible_agents[0]: int(dealt[0]),
            self.possible_agents[1]: int(dealt[1]),
        }
        self.contributions = {agent: 1 for agent in self.agents}
        self.history = []
        self.last_bettor = None

        self.phase = HandPhase.DEAL
        self._advance_from_deal()
        self.agent_selection = self.possible_agents[0]
        self._sync_infos()

    def observe(self, agent: str) -> dict[str, np.ndarray]:
        observation = np.zeros(_OBS_SIZE, dtype=np.int8)

        if agent in self.private_cards:
            card_index = self.private_cards[agent]
            observation[_PRIVATE_CARD_OFFSET + card_index] = 1

        history_index = self._history_index()
        observation[_HISTORY_OFFSET + history_index] = 1

        actor_index = self._current_actor_index()
        if actor_index is not None:
            observation[_ACTOR_OFFSET + actor_index] = 1

        return {
            "observation": observation,
            "action_mask": self._legal_action_mask(agent),
        }

    def step(self, action: Optional[int]) -> None:
        if self.terminations.get(self.agent_selection, False) or self.truncations.get(
            self.agent_selection, False
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        self._clear_rewards()
        self._cumulative_rewards[agent] = 0.0

        legal_mask = self._legal_action_mask(agent)
        if action is None:
            raise ValueError("Live agent must provide an action.")

        action = int(action)
        if action < 0 or action >= ACTION_DIM or legal_mask[action] == 0:
            raise ValueError(
                f"Illegal action {action} for agent {agent}. Legal mask: {legal_mask.tolist()}"
            )

        token = self._action_token(action)
        self._apply_action_effects(agent, token)
        winner = self._advance_phase(agent, token)
        if winner is not None:
            self._set_terminal_rewards(winner)
            self.terminations = {name: True for name in self.agents}

        self._sync_infos()
        self._accumulate_rewards()

    def render(self) -> None:
        if self.render_mode != "human":
            return
        print(
            f"phase={self.phase.value}, history={self.history}, "
            f"current={self.agent_selection}, cards={self.private_cards}, "
            f"contributions={self.contributions}"
        )

    def close(self) -> None:
        return

    def _legal_action_mask(self, agent: str) -> np.ndarray:
        if (
            agent != self.agent_selection
            or self.terminations.get(agent, False)
            or self.truncations.get(agent, False)
        ):
            return np.zeros(ACTION_DIM, dtype=np.int8)

        if self.phase in (HandPhase.P0_ACT, HandPhase.P1_ACT):
            return np.array([1, 1, 0], dtype=np.int8)
        if self.phase in (HandPhase.P0_RESPONSE, HandPhase.P1_RESPONSE):
            return np.array([1, 0, 1], dtype=np.int8)
        return np.zeros(ACTION_DIM, dtype=np.int8)

    def _action_token(self, action: int) -> str:
        facing_bet = self.phase in (HandPhase.P0_RESPONSE, HandPhase.P1_RESPONSE)
        if action == Action.CHECK_OR_CALL:
            return "call" if facing_bet else "check"
        if action == Action.BET_OR_RAISE:
            return "bet"
        return "fold"

    def _showdown_winner(self) -> str:
        p0, p1 = self.possible_agents
        return p0 if self.private_cards[p0] > self.private_cards[p1] else p1

    def _set_terminal_rewards(self, winner: Optional[str]) -> None:
        if winner is None:
            return
        loser = self.possible_agents[0] if winner == self.possible_agents[1] else self.possible_agents[1]
        pot = sum(self.contributions.values())
        self.rewards[winner] = float(pot - self.contributions[winner])
        self.rewards[loser] = float(-self.contributions[loser])

    def _next_agent(self, current_agent: str) -> str:
        if current_agent == self.possible_agents[0]:
            return self.possible_agents[1]
        return self.possible_agents[0]

    def _advance_from_deal(self) -> None:
        if self.phase != HandPhase.DEAL:
            raise RuntimeError(f"Cannot advance from non-deal phase: {self.phase}")
        self.phase = HandPhase.P0_ACT
        self.agent_selection = self.possible_agents[0]

    def _apply_action_effects(self, agent: str, token: str) -> None:
        if token in ("bet", "call"):
            self.contributions[agent] += 1
        if token == "bet":
            self.last_bettor = agent
        self.history.append(token)

    def _advance_phase(self, agent: str, token: str) -> Optional[str]:
        p0, p1 = self.possible_agents

        if self.phase == HandPhase.P0_ACT:
            if token == "check":
                self.phase = HandPhase.P1_ACT
                self.agent_selection = p1
                return None
            if token == "bet":
                self.phase = HandPhase.P1_RESPONSE
                self.agent_selection = p1
                return None

        elif self.phase == HandPhase.P1_ACT:
            if token == "check":
                self.phase = HandPhase.TERMINAL
                self.agent_selection = p0
                return self._showdown_winner()
            if token == "bet":
                self.phase = HandPhase.P0_RESPONSE
                self.agent_selection = p0
                return None

        elif self.phase in (HandPhase.P0_RESPONSE, HandPhase.P1_RESPONSE):
            self.phase = HandPhase.TERMINAL
            self.agent_selection = self._next_agent(agent)
            if token == "call":
                return self._showdown_winner()
            if token == "fold":
                return self.last_bettor

        raise RuntimeError(
            f"Invalid transition. phase={self.phase.value}, token={token}, agent={agent}"
        )

    def _sync_infos(self) -> None:
        self.infos = {
            agent: {
                "action_mask": self._legal_action_mask(agent),
                "phase": self.phase.value,
            }
            for agent in self.possible_agents
        }

    def _history_index(self) -> int:
        return _HISTORY_TO_INDEX.get(tuple(self.history), _TERMINAL_HISTORY_INDEX)

    def _current_actor_index(self) -> Optional[int]:
        if self.phase == HandPhase.TERMINAL:
            return None
        return 0 if self.agent_selection == self.possible_agents[0] else 1
