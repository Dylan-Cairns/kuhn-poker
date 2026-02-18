# Web Inference Contract

This document locks the browser-side inference contract for Kuhn Poker.

Canonical source:

- `contracts/kuhn.v1.json` (schema-first source of truth)
- Generated bindings:
  - `kuhn_poker/generated/contract.py`
  - `web/src/game/generated/contract.ts`

## Model Interface (ONNX)

The exported ONNX policy model uses:

- Input `observation`: `float32[batch, 10]`
- Input `action_mask`: `float32[batch, 3]`
- Output `masked_logits`: `float32[batch, 3]`
- Output `value`: `float32[batch, 1]`

Notes:

- `masked_logits` already applies invalid-action masking in the exported graph.
- Invalid actions should have very negative logits (for example `-1e9`).
- Frontend must still enforce legal mask checks before choosing an action.

## Observation Encoding

Observation vector length is `10`, matching Python env:

- `[0:3]` private card one-hot (`J, Q, K`)
- `[3:8]` public history one-hot:
  - index `3`: `[]`
  - index `4`: `["check"]`
  - index `5`: `["bet"]`
  - index `6`: `["check", "bet"]`
  - index `7`: terminal bucket
- `[8:10]` current actor one-hot (`player_0`, `player_1`)

## Action IDs and Masks

Action IDs are fixed:

- `0 = CHECK_OR_CALL`
- `1 = BET`
- `2 = FOLD`

Legal action mask semantics:

- Acting in `p0_act` or `p1_act`: `[1, 1, 0]`
- Acting in `p0_response` or `p1_response`: `[1, 0, 1]`
- Non-acting or terminal: `[0, 0, 0]`

## Bot Decision Mode

UI exposes a bot mode toggle with two options:

- `deterministic`: choose legal action with max masked logit
- `stochastic`: sample from softmax over legal masked logits

If multiple legal actions tie in deterministic mode, choose the smallest action ID.

## Frontend Parity Requirement

The TypeScript engine must match Python behavior for:

- phase transitions and legal masks
- terminal rewards (zero-sum net chip change)
- observation encoding and action IDs
