"""Export a trained MaskablePPO checkpoint to ONNX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sb3_contrib import MaskablePPO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kuhn_poker.onnx_export import DEFAULT_ONNX_OPSET, export_maskable_ppo_to_onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MaskablePPO checkpoint to ONNX.")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("checkpoints/maskable_ppo_kuhn.zip"),
        help="Path to trained .zip checkpoint from scripts/train.py",
    )
    parser.add_argument(
        "--onnx-out",
        type=Path,
        default=Path("models/kuhn_policy.onnx"),
        help="Output ONNX path.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_ONNX_OPSET,
        help="ONNX opset version.",
    )
    return parser.parse_args()


def resolve_checkpoint_path(checkpoint_path: Path) -> Path:
    if checkpoint_path.exists():
        return checkpoint_path
    if checkpoint_path.suffix != ".zip":
        zipped = checkpoint_path.with_suffix(".zip")
        if zipped.exists():
            return zipped
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_checkpoint_path(args.checkpoint_path)
    model = MaskablePPO.load(checkpoint_path, device="cpu")
    onnx_path = export_maskable_ppo_to_onnx(
        model=model,
        onnx_path=args.onnx_out,
        opset_version=args.opset,
    )

    print(f"Exported ONNX model: {onnx_path}")

    try:
        import onnx
    except ImportError:
        print("Optional validation skipped (install with: pip install -e .[onnx]).")
        return

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX validation passed.")


if __name__ == "__main__":
    main()
