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

Run the web app locally:

```bash
cd web
npm run dev
```

Build static assets (for GitHub Pages):

```bash
cd web
npm run build
```

Run a short masked PPO training smoke:

```bash
python scripts/train.py --total-timesteps 2048 --n-steps 128 --batch-size 64
```

Export a trained checkpoint to ONNX:

```bash
python scripts/export_onnx.py --checkpoint-path checkpoints/maskable_ppo_kuhn.zip --onnx-out models/kuhn_policy.onnx
```

Copy the exported model into the web public folder:

```bash
copy models\kuhn_policy.onnx web\public\models\kuhn_policy.onnx
```

## Deploy to GitHub Pages

This repo includes a workflow at `.github/workflows/deploy_pages.yml` that builds `web/` and deploys `web/dist` to GitHub Pages.

One-time setup in GitHub:

1. Go to `Settings -> Pages`.
2. Under `Build and deployment`, set `Source` to `GitHub Actions`.

Deploy flow:

1. Commit and push your web changes (including `web/public/models/kuhn_policy.onnx`).
2. The `Deploy Web App to GitHub Pages` workflow runs automatically on pushes to `main` that touch `web/**`.
3. Your site is published at:
   - `https://<your-github-username>.github.io/<your-repo-name>/`

You can also trigger deployment manually from `Actions -> Deploy Web App to GitHub Pages -> Run workflow`.

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
- `web/src/game/onnx_adapter.ts`: browser ONNX Runtime adapter for policy inference
- `web/src/game/onnx_adapter.test.ts`: adapter contract/unit tests
- `web/src/App.tsx`: single-page React UI for human-vs-bot play
- `web/src/main.tsx`: React app entrypoint
- `web/public/models/`: static ONNX model location for browser inference

## Next Steps

1. Evaluate trained checkpoints against random/heuristic baselines.
2. Add simple logging/plots and iterate on hyperparameters.

Roadmap status:
- Item 1 (rules): complete
- Item 2 (AEC state machine): complete
- Item 3 (observation/action/mask encoding): complete
- Item 4 (MaskablePPO training loop): complete
