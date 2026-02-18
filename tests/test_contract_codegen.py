from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from kuhn_poker.generated import contract as generated


def test_generated_bindings_are_in_sync() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "scripts/generate_contract_bindings.py",
        "--check",
    ]
    result = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        "Contract bindings are out of sync.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_generated_contract_core_values_match_source() -> None:
    contract_path = Path(__file__).resolve().parents[1] / "contracts" / "kuhn.v1.json"
    with contract_path.open("r", encoding="utf-8") as f:
        source = json.load(f)

    assert tuple(source["entities"]["players"]) == generated.PLAYERS
    assert tuple(source["entities"]["cards"]) == generated.CARDS
    assert tuple(source["entities"]["phases"]) == generated.PHASES
    assert len(source["actions"]) == generated.ACTION_DIM
    assert source["observation"]["size"] == generated.OBSERVATION_DIM
    assert source["observation"]["terminal_history_index"] == generated.OBS_TERMINAL_HISTORY_INDEX
