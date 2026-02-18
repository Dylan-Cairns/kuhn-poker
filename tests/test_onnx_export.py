from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO

from kuhn_poker.onnx_export import MaskablePolicyExportModule, export_maskable_ppo_to_onnx
from kuhn_poker.wrappers import make_masked_sb3_env

onnx = pytest.importorskip("onnx")
onnxruntime = pytest.importorskip("onnxruntime")


def _build_untrained_model(seed: int = 0) -> tuple[MaskablePPO, object]:
    env = make_masked_sb3_env(seed=seed)
    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        seed=seed,
        n_steps=64,
        batch_size=64,
    )
    return model, env


def _sample_obs_and_masks(num_samples: int = 8) -> list[tuple[np.ndarray, np.ndarray]]:
    env = make_masked_sb3_env(seed=1)
    samples: list[tuple[np.ndarray, np.ndarray]] = []
    observation, _ = env.reset(seed=1)

    while len(samples) < num_samples:
        action_mask = env.action_masks().astype(np.float32)
        samples.append((observation.astype(np.float32), action_mask))

        legal_actions = np.flatnonzero(action_mask)
        action = int(legal_actions[0])
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            observation, _ = env.reset()

    env.close()
    return samples


def test_export_creates_valid_onnx_model(tmp_path: Path) -> None:
    model, env = _build_untrained_model(seed=2)

    onnx_path = tmp_path / "kuhn_policy.onnx"
    export_maskable_ppo_to_onnx(model=model, onnx_path=onnx_path)

    assert onnx_path.exists()
    loaded = onnx.load(str(onnx_path))
    onnx.checker.check_model(loaded)
    env.close()


def test_exported_onnx_matches_torch_outputs(tmp_path: Path) -> None:
    model, env = _build_untrained_model(seed=3)
    onnx_path = tmp_path / "kuhn_policy.onnx"
    export_maskable_ppo_to_onnx(model=model, onnx_path=onnx_path)

    session = onnxruntime.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    export_module = MaskablePolicyExportModule(model).eval()
    samples = _sample_obs_and_masks(num_samples=10)

    for observation, action_mask in samples:
        observation_batch = observation.reshape(1, -1).astype(np.float32)
        action_mask_batch = action_mask.reshape(1, -1).astype(np.float32)

        with torch.no_grad():
            torch_logits, torch_value = export_module(
                torch.from_numpy(observation_batch),
                torch.from_numpy(action_mask_batch),
            )
        ort_logits, ort_value = session.run(
            None,
            {"observation": observation_batch, "action_mask": action_mask_batch},
        )

        assert np.allclose(ort_logits, torch_logits.numpy(), atol=1e-4, rtol=1e-4)
        assert np.allclose(ort_value, torch_value.numpy(), atol=1e-4, rtol=1e-4)

    env.close()
