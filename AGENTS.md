# AGENTS.md

This file defines how agents should work in this repository.

## Project Goal

Build a minimum-viable reinforcement learning bot for 2-player Kuhn Poker with:
- Python
- PettingZoo AEC (turn-based) environment
- Gymnasium-style spaces and wrappers as needed
- Stable-Baselines3 + `sb3_contrib.MaskablePPO`

Primary optimization target is implementation clarity and a clean learning project, not peak performance.

## Locked Decisions

- Episode scope: one hand per episode
- Training setup: shared policy self-play (same policy controls both players)
- Action IDs are fixed:
  - `0 = CHECK_OR_CALL`
  - `1 = BET_OR_RAISE`
  - `2 = FOLD`
- Environment must provide an invalid-action mask each turn

Do not change these without explicitly updating this file, tests, and README in the same change.

## Kuhn Rules Contract

Use the standard 3-card Kuhn deck: `J, Q, K` (one card per player, one unseen).

- Two players, one private card each
- Both players ante 1 chip at hand start
- Single betting round
- No raises beyond one bet

Terminal outcomes:
- Check-check: showdown, high card wins pot
- Bet-fold: bettor wins pot
- Bet-call: showdown, high card wins pot

Reward convention:
- Zero-sum per hand
- Return net chip change per player from the hand outcome

Information constraints:
- Each player observes only private card + public action history
- No access to opponent private card during play

## State Machine Contract (AEC)

Implement environment flow as:
- `deal`
- `p0_act`
- `p1_act`
- optional response state after a bet
- `terminal`

Keep this minimal and explicit. Prefer a small phase enum and compact public history representation.

## Observation and Action Contract

- Fixed discrete action space of size 3 using IDs above
- Per-turn legal action mask included in observation `dict` (or in `info`, with a wrapper that makes it available to MaskablePPO)
- Observation should be a compact fixed-size encoding of:
  - private card
  - public betting history
  - current actor/turn if needed for shared-policy disambiguation

Do not change action IDs after training starts; this breaks checkpoints and evaluations.

## Repo Layout Contract

Target layout:
- `kuhn_poker/env.py` - PettingZoo AEC environment
- `kuhn_poker/wrappers.py` - compatibility wrappers for SB3/masking
- `kuhn_poker/opponents.py` - random-legal and heuristic opponents
- `scripts/train.py` - MaskablePPO training entrypoint
- `scripts/eval.py` - evaluation entrypoint
- `scripts/smoke_test.py` - end-to-end sanity run
- `tests/` - environment and integration checks

If layout changes, keep README and command examples aligned.

## Standard Commands

Agents should keep these working (update as implementation evolves):
- Install deps: `pip install -e .` or `pip install -r requirements.txt`
- Train: `python scripts/train.py`
- Eval: `python scripts/eval.py`
- Sanity run: `python scripts/smoke_test.py`
- Tests: `pytest -q`

## Validation Checklist

Before marking work complete:
- Environment `reset()` and `step()` run without errors
- Action mask always has at least one legal action
- Illegal actions are never selected when masking is enabled
- Episode terminates correctly for all legal action paths
- Rewards are finite and zero-sum per hand
- A short MaskablePPO run completes end-to-end

## Definition of Done

A task is done when:
- Code changes are implemented
- Relevant tests/sanity checks pass
- README and this file are updated if contracts changed
- Any new assumptions are documented in code or docs
