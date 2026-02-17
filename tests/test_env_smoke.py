from __future__ import annotations

import numpy as np

from kuhn_poker.constants import AGENT_NAMES
from kuhn_poker.env import KuhnPokerAECEnv
from kuhn_poker.opponents import sample_random_legal_action


def test_env_random_play_is_zero_sum() -> None:
    env = KuhnPokerAECEnv()
    rng = np.random.default_rng(0)

    for _ in range(25):
        returns = {agent: 0.0 for agent in AGENT_NAMES}
        env.reset()
        for agent in env.agent_iter(max_iter=10):
            obs, reward, termination, truncation, _ = env.last()
            returns[agent] += reward
            if termination or truncation:
                action = None
            else:
                assert int(np.sum(obs["action_mask"])) > 0
                action = sample_random_legal_action(obs["action_mask"], rng)
            env.step(action)

        assert np.isclose(sum(returns.values()), 0.0)
