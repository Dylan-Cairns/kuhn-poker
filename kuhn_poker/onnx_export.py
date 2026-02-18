"""Utilities for exporting MaskablePPO policies to ONNX."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import torch
from sb3_contrib import MaskablePPO

from kuhn_poker.generated.contract import (
    ONNX_INPUT_ACTION_MASK_NAME,
    ONNX_INPUT_OBSERVATION_NAME,
    ONNX_OUTPUT_MASKED_LOGITS_NAME,
    ONNX_OUTPUT_VALUE_NAME,
)

DEFAULT_ONNX_OPSET: Final[int] = 17
_ILLEGAL_LOGIT: Final[float] = -1e9


class MaskablePolicyExportModule(torch.nn.Module):
    """Torch module that exposes masked logits and value for ONNX export."""

    def __init__(self, model: MaskablePPO) -> None:
        super().__init__()
        self.policy = model.policy

    def forward(
        self, observation: torch.Tensor, action_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.policy.extract_features(observation)
        if isinstance(features, tuple):
            pi_features, vf_features = features
            latent_pi = self.policy.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.policy.mlp_extractor.forward_critic(vf_features)
        else:
            latent_pi, latent_vf = self.policy.mlp_extractor(features)

        logits = self.policy.action_net(latent_pi)

        mask = action_mask.to(dtype=logits.dtype)
        illegal_logits = torch.full_like(logits, _ILLEGAL_LOGIT)
        masked_logits = torch.where(mask > 0.5, logits, illegal_logits)

        value = self.policy.value_net(latent_vf)
        return masked_logits, value


def export_maskable_ppo_to_onnx(
    model: MaskablePPO,
    onnx_path: Path,
    opset_version: int = DEFAULT_ONNX_OPSET,
) -> Path:
    """Export a trained MaskablePPO checkpoint to ONNX."""
    observation_shape = model.observation_space.shape
    if observation_shape is None:
        raise ValueError("Model observation space shape is undefined.")
    if len(observation_shape) != 1:
        raise ValueError(
            f"Expected flat 1D observations for export, got shape={observation_shape}."
        )
    if not hasattr(model.action_space, "n"):
        raise ValueError("Expected discrete action space for export.")

    obs_dim = int(observation_shape[0])
    action_dim = int(model.action_space.n)

    export_module = MaskablePolicyExportModule(model).eval().to("cpu")
    dummy_observation = torch.zeros((1, obs_dim), dtype=torch.float32)
    dummy_action_mask = torch.ones((1, action_dim), dtype=torch.float32)

    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            export_module,
            (dummy_observation, dummy_action_mask),
            str(onnx_path),
            dynamo=False,
            input_names=[ONNX_INPUT_OBSERVATION_NAME, ONNX_INPUT_ACTION_MASK_NAME],
            output_names=[ONNX_OUTPUT_MASKED_LOGITS_NAME, ONNX_OUTPUT_VALUE_NAME],
            dynamic_axes={
                ONNX_INPUT_OBSERVATION_NAME: {0: "batch"},
                ONNX_INPUT_ACTION_MASK_NAME: {0: "batch"},
                ONNX_OUTPUT_MASKED_LOGITS_NAME: {0: "batch"},
                ONNX_OUTPUT_VALUE_NAME: {0: "batch"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )

    return onnx_path
