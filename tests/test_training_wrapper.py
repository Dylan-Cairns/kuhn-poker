from __future__ import annotations

import numpy as np

from kuhn_poker.wrappers import make_masked_sb3_env


def test_masked_sb3_env_reset_step_and_masks() -> None:
    env = make_masked_sb3_env(seed=0)
    obs, _ = env.reset(seed=0)

    assert obs.shape == (10,)
    mask = env.action_masks()
    assert mask.shape == (3,)
    assert int(np.sum(mask)) >= 1

    action = int(np.flatnonzero(mask)[0])
    next_obs, reward, terminated, truncated, _ = env.step(action)
    assert next_obs.shape == (10,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    if not terminated and not truncated:
        next_mask = env.action_masks()
        assert next_mask.shape == (3,)
        assert int(np.sum(next_mask)) >= 1

    env.close()
