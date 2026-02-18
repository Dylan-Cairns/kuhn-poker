"""Play Kuhn Poker against a trained MaskablePPO bot in the terminal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from sb3_contrib import MaskablePPO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kuhn_poker.constants import AGENT_NAMES, CARD_LABELS, Action
from kuhn_poker.env import HandPhase, KuhnPokerAECEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Kuhn Poker vs a trained bot.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("checkpoints/maskable_ppo_kuhn.zip"),
        help="Path to .zip checkpoint from scripts/train.py",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--human-seat",
        type=int,
        choices=(0, 1),
        default=0,
        help="Seat index for the human player (0 acts first, 1 acts second).",
    )
    parser.add_argument(
        "--hands",
        type=int,
        default=0,
        help="Number of hands to play (0 means unlimited until quit).",
    )
    parser.add_argument(
        "--stochastic-bot",
        action="store_true",
        help="Use stochastic policy sampling instead of deterministic actions.",
    )
    return parser.parse_args()


def resolve_model_path(model_path: Path) -> Path:
    if model_path.exists():
        return model_path
    if model_path.suffix != ".zip":
        zipped = model_path.with_suffix(".zip")
        if zipped.exists():
            return zipped
    raise FileNotFoundError(
        f"Model checkpoint not found: {model_path}. "
        "Train first with scripts/train.py or pass --model-path."
    )


def format_history(history: list[str]) -> str:
    return " ".join(history) if history else "(start)"


def card_label(card_idx: int) -> str:
    return CARD_LABELS[card_idx]


def is_response_phase(phase: HandPhase) -> bool:
    return phase in (HandPhase.P0_RESPONSE, HandPhase.P1_RESPONSE)


def action_label(action: int, phase: HandPhase) -> str:
    if action == int(Action.CHECK_OR_CALL):
        return "call" if is_response_phase(phase) else "check"
    if action == int(Action.BET):
        return "bet"
    return "fold"


def legal_action_prompt(mask: np.ndarray, phase: HandPhase) -> str:
    options: list[str] = []
    if mask[int(Action.CHECK_OR_CALL)] == 1:
        options.append(f"[c] {action_label(int(Action.CHECK_OR_CALL), phase)}")
    if mask[int(Action.BET)] == 1:
        options.append("[b] bet")
    if mask[int(Action.FOLD)] == 1:
        options.append("[f] fold")
    options.append("[h] help")
    options.append("[q] quit")
    return ", ".join(options)


def print_help() -> None:
    print("Commands:")
    print("  c -> check/call (context-dependent)")
    print("  b -> bet")
    print("  f -> fold")
    print("  h -> help")
    print("  q -> quit")


def prompt_human_action(mask: np.ndarray, phase: HandPhase) -> Optional[int]:
    while True:
        print(f"Legal actions: {legal_action_prompt(mask, phase)}")
        raw = input("Your action: ").strip().lower()

        if raw == "h":
            print_help()
            continue
        if raw == "q":
            return None
        if raw == "c" and mask[int(Action.CHECK_OR_CALL)] == 1:
            return int(Action.CHECK_OR_CALL)
        if raw == "b" and mask[int(Action.BET)] == 1:
            return int(Action.BET)
        if raw == "f" and mask[int(Action.FOLD)] == 1:
            return int(Action.FOLD)
        print("Invalid or illegal action for this state. Try again.")


def play_hand(
    env: KuhnPokerAECEnv,
    model: MaskablePPO,
    deterministic_bot: bool,
    human_agent: str,
    bot_agent: str,
) -> tuple[bool, dict[str, float]]:
    returns = {agent: 0.0 for agent in AGENT_NAMES}
    env.reset()

    print("")
    print(f"Your card: {card_label(env.private_cards[human_agent])}")

    for agent in env.agent_iter(max_iter=10):
        obs, reward, termination, truncation, _ = env.last()
        returns[agent] += reward

        if termination or truncation:
            action = None
        elif agent == human_agent:
            phase = env.phase
            print(f"History: {format_history(env.history)}")
            action = prompt_human_action(obs["action_mask"], phase)
            if action is None:
                return True, returns
            print(f"You chose: {action_label(action, phase)}")
        else:
            phase = env.phase
            action, _ = model.predict(
                obs["observation"],
                action_masks=obs["action_mask"],
                deterministic=deterministic_bot,
            )
            action = int(action)
            print(f"Bot chose: {action_label(action, phase)}")

        env.step(action)

    print(
        "Reveal: "
        f"you={card_label(env.private_cards[human_agent])}, "
        f"bot={card_label(env.private_cards[bot_agent])}"
    )
    print(f"Final history: {format_history(env.history)}")
    print(
        f"Hand return: you={returns[human_agent]:+.1f}, "
        f"bot={returns[bot_agent]:+.1f}"
    )
    return False, returns


def prompt_continue() -> bool:
    while True:
        raw = input("Press Enter or [n] for next hand, [q] to quit: ").strip().lower()
        if raw in ("", "n"):
            return True
        if raw == "q":
            return False
        print("Invalid command. Use Enter, n, or q.")


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model_path)
    human_agent = AGENT_NAMES[args.human_seat]
    bot_agent = AGENT_NAMES[1 - args.human_seat]

    model = MaskablePPO.load(model_path)
    env = KuhnPokerAECEnv()
    env.reset(seed=args.seed)

    print("Kuhn Poker CLI")
    print(f"You are {human_agent}. Bot is {bot_agent}.")
    print(f"Loaded model: {model_path}")
    print_help()

    totals = {agent: 0.0 for agent in AGENT_NAMES}
    hand_count = 0
    completed_hands = 0
    quit_requested = False

    while not quit_requested:
        if args.hands > 0 and hand_count >= args.hands:
            break

        hand_count += 1
        print("")
        print(f"=== Hand {hand_count} ===")
        quit_requested, hand_returns = play_hand(
            env=env,
            model=model,
            deterministic_bot=not args.stochastic_bot,
            human_agent=human_agent,
            bot_agent=bot_agent,
        )

        if not quit_requested:
            completed_hands += 1
            for agent in AGENT_NAMES:
                totals[agent] += hand_returns[agent]
            print(
                f"Session score: you={totals[human_agent]:+.1f}, "
                f"bot={totals[bot_agent]:+.1f}"
            )
            if args.hands == 0:
                quit_requested = not prompt_continue()

    env.close()
    if completed_hands > 0:
        print("")
        print(
            f"Final score after {completed_hands} completed hand(s): "
            f"you={totals[human_agent]:+.1f}, "
            f"bot={totals[bot_agent]:+.1f}"
        )
    print("Session ended.")


if __name__ == "__main__":
    main()
