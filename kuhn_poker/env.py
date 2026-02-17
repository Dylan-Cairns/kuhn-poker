"""PettingZoo AEC environment scaffold for 2-player Kuhn Poker."""

from __future__ import annotations

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
_OBS_SIZE = len(CARD_LABELS) + 5


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

        self.rewards: dict[str, float] = {}
        self._cumulative_rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict[str, np.ndarray]] = {}

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

        self.agent_selection = self.possible_agents[0]
        self._sync_infos()

    def observe(self, agent: str) -> dict[str, np.ndarray]:
        observation = np.zeros(_OBS_SIZE, dtype=np.int8)

        if agent in self.private_cards:
            observation[self.private_cards[agent]] = 1

        history_index = _HISTORY_TO_INDEX.get(tuple(self.history), _TERMINAL_HISTORY_INDEX)
        observation[len(CARD_LABELS) + history_index] = 1

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
        if token in ("bet", "call"):
            self.contributions[agent] += 1
        if token == "bet":
            self.last_bettor = agent
        self.history.append(token)

        winner = self._resolve_terminal_winner()
        if winner is None:
            self.agent_selection = self._next_agent(agent)
        else:
            self._set_terminal_rewards(winner)
            self.terminations = {name: True for name in self.agents}
            self.agent_selection = self._next_agent(agent)

        self._sync_infos()
        self._accumulate_rewards()

    def render(self) -> None:
        if self.render_mode != "human":
            return
        print(
            f"history={self.history}, current={self.agent_selection}, "
            f"cards={self.private_cards}, contributions={self.contributions}"
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

        history = tuple(self.history)
        if history in ((), ("check",)):
            return np.array([1, 1, 0], dtype=np.int8)
        if history in (("bet",), ("check", "bet")):
            return np.array([1, 0, 1], dtype=np.int8)
        return np.zeros(ACTION_DIM, dtype=np.int8)

    def _action_token(self, action: int) -> str:
        history = tuple(self.history)
        facing_bet = history in (("bet",), ("check", "bet"))
        if action == Action.CHECK_OR_CALL:
            return "call" if facing_bet else "check"
        if action == Action.BET_OR_RAISE:
            return "bet"
        return "fold"

    def _resolve_terminal_winner(self) -> Optional[str]:
        history = tuple(self.history)
        if history in (("check", "check"), ("bet", "call"), ("check", "bet", "call")):
            return self._showdown_winner()
        if history in (("bet", "fold"), ("check", "bet", "fold")):
            return self.last_bettor
        return None

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

    def _sync_infos(self) -> None:
        self.infos = {
            agent: {"action_mask": self._legal_action_mask(agent)} for agent in self.possible_agents
        }
