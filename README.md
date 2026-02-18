# Kuhn Poker RL (MVP)

Minimal scaffolding for a 2-player Kuhn Poker learning project with:
- Python
- PettingZoo AEC environment
- Gymnasium spaces
- Stable-Baselines3 + `sb3-contrib` (`MaskablePPO`) integration path

This repo is intentionally set up as a small, clean starting point.

Requires Python 3.10+.

## Quickstart

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

Install ONNX export dependencies when needed:

```bash
pip install -e .[onnx]
```

Run basic checks:

```bash
python scripts/smoke_test.py
pytest -q
```

Install and run web engine tests:

```bash
cd web
npm install
npm test
```

Run a short masked PPO training smoke:

```bash
python scripts/train.py --total-timesteps 2048 --n-steps 128 --batch-size 64
```

Export a trained checkpoint to ONNX:

```bash
python scripts/export_onnx.py --checkpoint-path checkpoints/maskable_ppo_kuhn.zip --onnx-out models/kuhn_policy.onnx
```

Play against a trained checkpoint in the CLI:

```bash
python scripts/play_cli.py --model-path checkpoints/maskable_ppo_kuhn.zip
```

Play as second player:

```bash
python scripts/play_cli.py --model-path checkpoints/maskable_ppo_kuhn.zip --human-seat 1
```

## Current Layout

- `kuhn_poker/env.py`: PettingZoo AEC environment
- `kuhn_poker/opponents.py`: baseline opponents (random legal + simple heuristic)
- `kuhn_poker/wrappers.py`: helper functions for conversion/masking integration
- `scripts/train.py`: training entrypoint scaffold
- `scripts/export_onnx.py`: checkpoint to ONNX export entrypoint
- `scripts/eval.py`: evaluation entrypoint scaffold
- `scripts/smoke_test.py`: end-to-end sanity check
- `tests/`: smoke tests for environment/opponents
- `docs/kuhn_rules.md`: exact Kuhn rules contract implemented by the environment
- `docs/web_inference_contract.md`: locked browser ONNX I/O + action selection contract
- `web/src/game/engine.ts`: frontend Kuhn rules/state machine engine
- `web/src/game/engine.test.ts`: frontend parity/unit tests

## Next Steps

1. Evaluate trained checkpoints against random/heuristic baselines.
2. Add simple logging/plots and iterate on hyperparameters.

Roadmap status:
- Item 1 (rules): complete
- Item 2 (AEC state machine): complete
- Item 3 (observation/action/mask encoding): complete
- Item 4 (MaskablePPO training loop): complete
