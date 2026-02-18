"""Generate Python and TypeScript bindings from the canonical game contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover - optional tooling dependency
    Draft202012Validator = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT_PATH = REPO_ROOT / "contracts" / "kuhn.v1.json"
DEFAULT_SCHEMA_PATH = REPO_ROOT / "contracts" / "schema" / "game_contract.schema.json"
DEFAULT_PY_OUT = REPO_ROOT / "kuhn_poker" / "generated" / "contract.py"
DEFAULT_TS_OUT = REPO_ROOT / "web" / "src" / "game" / "generated" / "contract.ts"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _validate_schema(contract: dict[str, Any], schema: dict[str, Any]) -> None:
    if Draft202012Validator is None:
        _assert(
            isinstance(schema, dict) and schema.get("type") == "object",
            "Schema file is malformed.",
        )
        required = schema.get("required", [])
        _assert(isinstance(required, list), "Schema 'required' must be a list.")
        for key in required:
            _assert(key in contract, f"Contract is missing required top-level key: {key}")
        return

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(contract), key=lambda error: list(error.path))
    if not errors:
        return

    rendered = []
    for error in errors:
        path = ".".join(str(part) for part in error.path)
        location = path if path else "<root>"
        rendered.append(f"- {location}: {error.message}")
    raise ValueError("Contract schema validation failed:\n" + "\n".join(rendered))


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_semantics(contract: dict[str, Any]) -> None:
    entities = contract["entities"]
    turn_model = contract["turn_model"]
    actions = sorted(contract["actions"], key=lambda item: item["id"])
    legal_masks = contract["legal_masks_by_phase"]
    observation = contract["observation"]

    phases = set(entities["phases"])
    players = entities["players"]
    cards = entities["cards"]
    public_actions = set(entities["public_actions"])
    action_ids = [int(item["id"]) for item in actions]

    _assert(action_ids == list(range(len(action_ids))), "Action ids must be contiguous from 0.")

    action_names = [item["name"] for item in actions]
    _assert(len(set(action_names)) == len(action_names), "Action names must be unique.")

    _assert(len(players) == 2, "Current project requires exactly 2 players.")
    _assert(turn_model["initial_actor"] in players, "Initial actor must be a declared player.")
    _assert(turn_model["initial_phase"] in phases, "Initial phase must be declared.")
    _assert(turn_model["terminal_phase"] in phases, "Terminal phase must be declared.")

    open_phases = set(turn_model["open_action_phases"])
    response_phases = set(turn_model["response_action_phases"])
    _assert(open_phases.issubset(phases), "Open action phases must be declared phases.")
    _assert(
        response_phases.issubset(phases), "Response action phases must be declared phases."
    )
    _assert(
        open_phases.isdisjoint(response_phases),
        "Open action phases and response action phases must be disjoint.",
    )

    _assert(set(legal_masks.keys()) == phases, "legal_masks_by_phase must cover every phase.")
    action_dim = len(actions)
    for phase, mask in legal_masks.items():
        _assert(
            len(mask) == action_dim,
            f"Phase '{phase}' mask length {len(mask)} != action_dim {action_dim}.",
        )
        _assert(
            all(bit in (0, 1) for bit in mask),
            f"Phase '{phase}' has invalid mask values (must be 0/1).",
        )

    for action in actions:
        open_label = action["labels"]["open"]
        response_label = action["labels"]["response"]
        _assert(
            open_label in public_actions,
            f"Action '{action['name']}' open label '{open_label}' must be a public action token.",
        )
        _assert(
            response_label in public_actions,
            f"Action '{action['name']}' response label '{response_label}' must be a public action token.",
        )

    segments = sorted(observation["segments"], key=lambda item: item["offset"])
    expected_offset = 0
    for segment in segments:
        offset = int(segment["offset"])
        size = int(segment["size"])
        _assert(
            offset == expected_offset,
            f"Observation segment '{segment['name']}' has offset {offset}, expected {expected_offset}.",
        )
        expected_offset += size
    _assert(
        expected_offset == int(observation["size"]),
        "Observation segment sizes/offsets do not match observation.size.",
    )

    segment_by_name = {segment["name"]: segment for segment in segments}
    for required_segment in (
        "private_card_one_hot",
        "public_history_one_hot",
        "current_actor_one_hot",
    ):
        _assert(
            required_segment in segment_by_name,
            f"Missing required observation segment '{required_segment}'.",
        )

    private_card_size = int(segment_by_name["private_card_one_hot"]["size"])
    actor_size = int(segment_by_name["current_actor_one_hot"]["size"])
    _assert(
        private_card_size == len(cards),
        "private_card_one_hot size must equal number of cards.",
    )
    _assert(actor_size == len(players), "current_actor_one_hot size must equal number of players.")

    history_segment_size = int(segment_by_name["public_history_one_hot"]["size"])
    history_buckets = observation["history_buckets"]
    _assert(
        history_segment_size == len(history_buckets),
        "public_history_one_hot size must equal number of history buckets.",
    )

    history_indices = [int(bucket["index"]) for bucket in history_buckets]
    _assert(
        sorted(history_indices) == list(range(len(history_buckets))),
        "History bucket indices must be contiguous from 0.",
    )

    terminal_history_index = int(observation["terminal_history_index"])
    _assert(
        terminal_history_index in history_indices,
        "terminal_history_index must be one of the history bucket indices.",
    )

    sequence_keys: set[tuple[str, ...]] = set()
    for bucket in history_buckets:
        sequence = bucket["sequence"]
        if sequence is None:
            continue
        key = tuple(sequence)
        _assert(key not in sequence_keys, f"Duplicate history sequence: {key}")
        sequence_keys.add(key)


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _ts_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=False)


def _render_python(contract: dict[str, Any], contract_path: str) -> str:
    entities = contract["entities"]
    turn_model = contract["turn_model"]
    actions = sorted(contract["actions"], key=lambda item: item["id"])
    legal_masks = contract["legal_masks_by_phase"]
    observation = contract["observation"]
    onnx = contract["onnx"]

    segment_by_name = {segment["name"]: segment for segment in observation["segments"]}
    history_buckets = sorted(observation["history_buckets"], key=lambda item: item["index"])
    history_sequence_map = {
        tuple(bucket["sequence"]): int(bucket["index"])
        for bucket in history_buckets
        if bucket["sequence"] is not None
    }

    action_id_by_name = {item["name"]: int(item["id"]) for item in actions}
    action_name_by_id = {int(item["id"]): item["name"] for item in actions}
    action_open_label_by_id = {
        int(item["id"]): item["labels"]["open"] for item in actions
    }
    action_response_label_by_id = {
        int(item["id"]): item["labels"]["response"] for item in actions
    }

    history_bucket_tuples = tuple(
        (int(bucket["index"]), tuple(bucket["sequence"]) if bucket["sequence"] is not None else None)
        for bucket in history_buckets
    )

    legal_masks_by_phase = {
        phase: tuple(int(bit) for bit in legal_masks[phase])
        for phase in entities["phases"]
    }

    lines = [
        '"""Auto-generated contract constants.',
        "",
        f"Source: {contract_path}",
        "Generated by: scripts/generate_contract_bindings.py",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import Final",
        "",
        f"CONTRACT_NAME: Final[str] = {contract['contract_name']!r}",
        f"CONTRACT_VERSION: Final[str] = {contract['version']!r}",
        "",
        f"PLAYERS: Final[tuple[str, ...]] = {tuple(entities['players'])!r}",
        f"CARDS: Final[tuple[str, ...]] = {tuple(entities['cards'])!r}",
        f"PUBLIC_ACTIONS: Final[tuple[str, ...]] = {tuple(entities['public_actions'])!r}",
        f"PHASES: Final[tuple[str, ...]] = {tuple(entities['phases'])!r}",
        "",
        f"INITIAL_PHASE: Final[str] = {turn_model['initial_phase']!r}",
        f"INITIAL_ACTOR: Final[str] = {turn_model['initial_actor']!r}",
        f"TERMINAL_PHASE: Final[str] = {turn_model['terminal_phase']!r}",
        f"OPEN_ACTION_PHASES: Final[tuple[str, ...]] = {tuple(turn_model['open_action_phases'])!r}",
        "RESPONSE_ACTION_PHASES: Final[tuple[str, ...]] = "
        f"{tuple(turn_model['response_action_phases'])!r}",
        "",
        f"ACTION_DIM: Final[int] = {len(actions)}",
        f"ACTION_ID_BY_NAME: Final[dict[str, int]] = {action_id_by_name!r}",
        f"ACTION_NAME_BY_ID: Final[dict[int, str]] = {action_name_by_id!r}",
        f"ACTION_OPEN_LABEL_BY_ID: Final[dict[int, str]] = {action_open_label_by_id!r}",
        "ACTION_RESPONSE_LABEL_BY_ID: Final[dict[int, str]] = "
        f"{action_response_label_by_id!r}",
        f"LEGAL_MASK_BY_PHASE: Final[dict[str, tuple[int, ...]]] = {legal_masks_by_phase!r}",
        "",
        f"OBSERVATION_DIM: Final[int] = {int(observation['size'])}",
        "OBS_PRIVATE_CARD_OFFSET: Final[int] = "
        f"{int(segment_by_name['private_card_one_hot']['offset'])}",
        "OBS_PRIVATE_CARD_DIM: Final[int] = "
        f"{int(segment_by_name['private_card_one_hot']['size'])}",
        "OBS_HISTORY_OFFSET: Final[int] = "
        f"{int(segment_by_name['public_history_one_hot']['offset'])}",
        "OBS_HISTORY_DIM: Final[int] = "
        f"{int(segment_by_name['public_history_one_hot']['size'])}",
        "OBS_ACTOR_OFFSET: Final[int] = "
        f"{int(segment_by_name['current_actor_one_hot']['offset'])}",
        "OBS_ACTOR_DIM: Final[int] = "
        f"{int(segment_by_name['current_actor_one_hot']['size'])}",
        "OBS_HISTORY_BUCKETS: Final[tuple[tuple[int, tuple[str, ...] | None], ...]] = "
        f"{history_bucket_tuples!r}",
        f"OBS_HISTORY_INDEX_BY_SEQUENCE: Final[dict[tuple[str, ...], int]] = {history_sequence_map!r}",
        f"OBS_TERMINAL_HISTORY_INDEX: Final[int] = {int(observation['terminal_history_index'])}",
        "",
        "CARD_INDEX_BY_LABEL: Final[dict[str, int]] = {label: index for index, label in enumerate(CARDS)}",
        "PLAYER_INDEX_BY_ID: Final[dict[str, int]] = {player: index for index, player in enumerate(PLAYERS)}",
        "",
        "ONNX_INPUT_OBSERVATION_NAME: Final[str] = "
        f"{onnx['input_names']['observation']!r}",
        f"ONNX_INPUT_ACTION_MASK_NAME: Final[str] = {onnx['input_names']['action_mask']!r}",
        "ONNX_OUTPUT_MASKED_LOGITS_NAME: Final[str] = "
        f"{onnx['output_names']['masked_logits']!r}",
        f"ONNX_OUTPUT_VALUE_NAME: Final[str] = {onnx['output_names']['value']!r}",
        f"ONNX_VALUE_DIM: Final[int] = {int(onnx['value_dim'])}",
        "",
    ]
    return "\n".join(lines)


def _render_typescript(contract: dict[str, Any], contract_path: str) -> str:
    entities = contract["entities"]
    turn_model = contract["turn_model"]
    actions = sorted(contract["actions"], key=lambda item: item["id"])
    legal_masks = contract["legal_masks_by_phase"]
    observation = contract["observation"]
    onnx = contract["onnx"]

    segment_by_name = {segment["name"]: segment for segment in observation["segments"]}
    history_buckets = sorted(observation["history_buckets"], key=lambda item: item["index"])

    action_dim = len(actions)
    action_mask_type = ", ".join("number" for _ in range(action_dim))
    action_name_union = " | ".join(f'"{action["name"]}"' for action in actions)

    action_id_constants = "\n".join(
        f"export const ACTION_{action['name']} = {int(action['id'])} as const"
        for action in actions
    )

    action_id_by_name = {action["name"]: int(action["id"]) for action in actions}
    action_open_label_by_id = {
        str(int(action["id"])): action["labels"]["open"] for action in actions
    }
    action_response_label_by_id = {
        str(int(action["id"])): action["labels"]["response"] for action in actions
    }
    history_key_to_index = {
        "|".join(bucket["sequence"]): int(bucket["index"])
        for bucket in history_buckets
        if bucket["sequence"] is not None
    }
    card_index_by_label = {
        card: index for index, card in enumerate(entities["cards"])
    }
    player_index_by_id = {
        player: index for index, player in enumerate(entities["players"])
    }

    legal_masks_by_phase = {
        phase: [int(bit) for bit in legal_masks[phase]]
        for phase in entities["phases"]
    }

    lines = [
        "/* Auto-generated contract constants.",
        f" * Source: {contract_path}",
        " * Generated by: scripts/generate_contract_bindings.py",
        " */",
        "",
        f"export const CONTRACT_NAME = {contract['contract_name']!r} as const",
        f"export const CONTRACT_VERSION = {contract['version']!r} as const",
        "",
        f"export const PLAYERS = {_ts_json(entities['players'])} as const",
        "export type PlayerId = (typeof PLAYERS)[number]",
        "",
        f"export const CARDS = {_ts_json(entities['cards'])} as const",
        "export type Card = (typeof CARDS)[number]",
        "",
        f"export const PUBLIC_ACTIONS = {_ts_json(entities['public_actions'])} as const",
        "export type PublicAction = (typeof PUBLIC_ACTIONS)[number]",
        "",
        f"export const PHASES = {_ts_json(entities['phases'])} as const",
        "export type Phase = (typeof PHASES)[number]",
        "",
        f"export const INITIAL_PHASE = {turn_model['initial_phase']!r} as const",
        f"export const INITIAL_ACTOR = {turn_model['initial_actor']!r} as const",
        f"export const TERMINAL_PHASE = {turn_model['terminal_phase']!r} as const",
        "export const OPEN_ACTION_PHASES = "
        f"{_ts_json(turn_model['open_action_phases'])} as const",
        "export const RESPONSE_ACTION_PHASES = "
        f"{_ts_json(turn_model['response_action_phases'])} as const",
        "",
        f"export const ACTIONS = {_ts_json(actions)} as const",
        "export type ActionName = (typeof ACTIONS)[number][\"name\"]",
        "export type ActionId = (typeof ACTIONS)[number][\"id\"]",
        f"export type ActionMask = readonly [{action_mask_type}]",
        f"export const ACTION_DIM = {action_dim} as const",
        action_id_constants,
        "",
        f"export const ACTION_NAME_BY_ID = {_ts_json({str(a['id']): a['name'] for a in actions})} as const",
        f"export const ACTION_ID_BY_NAME = {_ts_json(action_id_by_name)} as const",
        "export const ACTION_OPEN_LABEL_BY_ID = "
        f"{_ts_json(action_open_label_by_id)} as const",
        "export const ACTION_RESPONSE_LABEL_BY_ID = "
        f"{_ts_json(action_response_label_by_id)} as const",
        "",
        "export const LEGAL_MASK_BY_PHASE = "
        f"{_ts_json(legal_masks_by_phase)} as const satisfies Record<Phase, ActionMask>",
        "export const NO_LEGAL_ACTION_MASK = LEGAL_MASK_BY_PHASE[TERMINAL_PHASE]",
        "",
        f"export const OBSERVATION_DIM = {int(observation['size'])} as const",
        "export const OBS_PRIVATE_CARD_OFFSET = "
        f"{int(segment_by_name['private_card_one_hot']['offset'])} as const",
        "export const OBS_PRIVATE_CARD_DIM = "
        f"{int(segment_by_name['private_card_one_hot']['size'])} as const",
        "export const OBS_HISTORY_OFFSET = "
        f"{int(segment_by_name['public_history_one_hot']['offset'])} as const",
        "export const OBS_HISTORY_DIM = "
        f"{int(segment_by_name['public_history_one_hot']['size'])} as const",
        "export const OBS_ACTOR_OFFSET = "
        f"{int(segment_by_name['current_actor_one_hot']['offset'])} as const",
        "export const OBS_ACTOR_DIM = "
        f"{int(segment_by_name['current_actor_one_hot']['size'])} as const",
        "export const OBS_HISTORY_BUCKETS = "
        f"{_ts_json(history_buckets)} as const",
        "export const OBS_HISTORY_SEQUENCE_TO_INDEX = "
        f"{_ts_json(history_key_to_index)} as const",
        "export const OBS_TERMINAL_HISTORY_INDEX = "
        f"{int(observation['terminal_history_index'])} as const",
        "",
        f"export const CARD_INDEX_BY_LABEL = {_ts_json(card_index_by_label)} as const",
        f"export const PLAYER_INDEX_BY_ID = {_ts_json(player_index_by_id)} as const",
        "",
        "export const ONNX_INPUT_OBSERVATION_NAME = "
        f"{onnx['input_names']['observation']!r} as const",
        "export const ONNX_INPUT_ACTION_MASK_NAME = "
        f"{onnx['input_names']['action_mask']!r} as const",
        "export const ONNX_OUTPUT_MASKED_LOGITS_NAME = "
        f"{onnx['output_names']['masked_logits']!r} as const",
        "export const ONNX_OUTPUT_VALUE_NAME = "
        f"{onnx['output_names']['value']!r} as const",
        f"export const ONNX_VALUE_DIM = {int(onnx['value_dim'])} as const",
        "",
        f"export type ActionSymbol = {action_name_union}",
        "",
    ]
    return "\n".join(lines)


def _read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(content)


def _check_or_write(path: Path, content: str, check: bool) -> bool:
    if path.exists():
        current = _read_text(path)
    else:
        current = ""

    if current == content:
        return False

    if check:
        return True

    _write_text(path, content)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Python/TypeScript bindings from contracts/kuhn.v1.json."
    )
    parser.add_argument(
        "--contract-path",
        type=Path,
        default=DEFAULT_CONTRACT_PATH,
        help="Path to source contract JSON.",
    )
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
        help="Path to JSON schema for contract validation.",
    )
    parser.add_argument(
        "--python-out",
        type=Path,
        default=DEFAULT_PY_OUT,
        help="Generated Python output path.",
    )
    parser.add_argument(
        "--ts-out",
        type=Path,
        default=DEFAULT_TS_OUT,
        help="Generated TypeScript output path.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check mode: fail if generated outputs are out of date.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract_path = args.contract_path.resolve()
    schema_path = args.schema_path.resolve()

    contract = _load_json(contract_path)
    schema = _load_json(schema_path)

    _validate_schema(contract, schema)
    _validate_semantics(contract)

    displayed_contract_path = _display_path(contract_path)
    py_content = _render_python(contract, displayed_contract_path)
    ts_content = _render_typescript(contract, displayed_contract_path)

    py_changed = _check_or_write(args.python_out, py_content, check=args.check)
    ts_changed = _check_or_write(args.ts_out, ts_content, check=args.check)

    if args.check:
        if py_changed or ts_changed:
            pending = []
            if py_changed:
                pending.append(str(args.python_out))
            if ts_changed:
                pending.append(str(args.ts_out))
            raise SystemExit(
                "Generated bindings are out of date. Run:\n"
                "python scripts/generate_contract_bindings.py\n"
                "Files needing updates:\n- "
                + "\n- ".join(pending)
            )
        print("Generated bindings are up to date.")
        return

    if not py_changed and not ts_changed:
        print("No binding changes detected.")
        return

    print("Generated contract bindings:")
    print(f"- Python: {args.python_out}")
    print(f"- TypeScript: {args.ts_out}")


if __name__ == "__main__":
    main()
