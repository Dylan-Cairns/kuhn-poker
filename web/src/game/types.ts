export {
  ACTION_BET,
  ACTION_CHECK_OR_CALL,
  ACTION_DIM,
  ACTION_FOLD,
  CARDS,
  PHASES,
  PLAYERS
} from "./generated/contract"

export type {
  ActionId,
  ActionMask,
  Card,
  Phase,
  PlayerId,
  PublicAction
} from "./generated/contract"

import type { Card, Phase, PlayerId, PublicAction } from "./generated/contract"

export type BotDecisionMode = "deterministic" | "stochastic";

export interface HandState {
  readonly phase: Phase;
  readonly actor: PlayerId | null;
  readonly privateCards: Readonly<Record<PlayerId, Card>>;
  readonly history: readonly PublicAction[];
  readonly contributions: Readonly<Record<PlayerId, number>>;
  readonly lastBettor: PlayerId | null;
  readonly terminalReturns: Readonly<Record<PlayerId, number>> | null;
}
