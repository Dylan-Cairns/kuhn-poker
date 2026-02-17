"""Small wrappers/helpers for training integration."""

from __future__ import annotations

import numpy as np


def action_mask_from_observation(observation: dict) -> np.ndarray:
    """Extract action mask from observation dict."""
    return np.asarray(observation["action_mask"], dtype=np.int8)


def to_sb3_training_env(aec_env):
    """Placeholder for AEC -> SB3 training env conversion."""
    del aec_env
    raise NotImplementedError(
        "Roadmap item 4: choose the least-friction AEC/wrapper path for MaskablePPO."
    )
