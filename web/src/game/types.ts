export const ACTION_CHECK_OR_CALL = 0 as const;
export const ACTION_BET = 1 as const;
export const ACTION_FOLD = 2 as const;

export type ActionId =
  | typeof ACTION_CHECK_OR_CALL
  | typeof ACTION_BET
  | typeof ACTION_FOLD;

export type PlayerId = "player_0" | "player_1";
export type Card = "J" | "Q" | "K";
export type PublicAction = "check" | "bet" | "call" | "fold";

export type Phase =
  | "deal"
  | "p0_act"
  | "p1_act"
  | "p0_response"
  | "p1_response"
  | "terminal";

export type ActionMask = readonly [number, number, number];

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

export const PLAYERS: readonly [PlayerId, PlayerId] = ["player_0", "player_1"];
export const CARDS: readonly [Card, Card, Card] = ["J", "Q", "K"];
