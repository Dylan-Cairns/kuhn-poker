"""Wrappers/helpers for SB3 MaskablePPO training."""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from sb3_contrib.common.wrappers import ActionMasker

from kuhn_poker.env import KuhnPokerAECEnv

try:
    from pettingzoo.utils import BaseWrapper
except ImportError:  # pragma: no cover
    from pettingzoo.utils.wrappers import BaseWrapper  # type: ignore


def action_mask_from_observation(observation: dict) -> np.ndarray:
    """Extract action mask from observation dict."""
    return np.asarray(observation["action_mask"], dtype=np.int8)


class SB3ActionMaskWrapper(BaseWrapper, gym.Env):
    """Expose a PettingZoo AEC env as a Gymnasium env for parameter-sharing SB3."""

    def __init__(self, env: KuhnPokerAECEnv) -> None:
        super().__init__(env)
        first_agent = self.possible_agents[0]
        self.observation_space = super().observation_space(first_agent)["observation"]
        self.action_space = super().action_space(first_agent)

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)
        agent = self.agent_selection
        return self.observe(agent), dict(self.infos[agent])

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        current_agent = self.agent_selection
        super().step(action)
        next_agent = self.agent_selection
        return (
            self.observe(next_agent),
            float(self._cumulative_rewards[current_agent]),
            bool(self.terminations[current_agent]),
            bool(self.truncations[current_agent]),
            dict(self.infos[current_agent]),
        )

    def observe(self, agent: str) -> np.ndarray:
        return np.asarray(super().observe(agent)["observation"], dtype=np.int8)

    def action_mask(self) -> np.ndarray:
        return np.asarray(super().observe(self.agent_selection)["action_mask"], dtype=np.int8)


def mask_fn(env: SB3ActionMaskWrapper) -> np.ndarray:
    """ActionMasker callback."""
    return env.action_mask()


def make_masked_sb3_env(seed: int = 0):
    """Create the minimal SB3-compatible masked training environment."""
    env = KuhnPokerAECEnv()
    env = SB3ActionMaskWrapper(env)
    env.reset(seed=seed)
    return ActionMasker(env, mask_fn)
