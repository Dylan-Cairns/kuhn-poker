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

Run basic checks:

```bash
python scripts/smoke_test.py
pytest -q
```

## Current Layout

- `kuhn_poker/env.py`: PettingZoo AEC environment
- `kuhn_poker/opponents.py`: baseline opponents (random legal + simple heuristic)
- `kuhn_poker/wrappers.py`: helper functions for conversion/masking integration
- `scripts/train.py`: training entrypoint scaffold
- `scripts/eval.py`: evaluation entrypoint scaffold
- `scripts/smoke_test.py`: end-to-end sanity check
- `tests/`: smoke tests for environment/opponents
- `docs/kuhn_rules.md`: exact Kuhn rules contract implemented by the environment

## Next Steps

1. Finalize the explicit PettingZoo AEC state machine phases.
2. Add SB3 + MaskablePPO training loop for shared-policy self-play.
3. Expand evaluation and logging.
