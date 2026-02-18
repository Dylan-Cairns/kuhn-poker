import {
  ACTION_BET,
  ACTION_DIM,
  ACTION_CHECK_OR_CALL,
  ACTION_FOLD,
  CARDS,
  type ActionId,
  type ActionMask,
  type BotDecisionMode,
  type Card,
  type HandState,
  type PlayerId,
  type PublicAction
} from "./types";
import {
  ACTION_OPEN_LABEL_BY_ID,
  ACTION_RESPONSE_LABEL_BY_ID,
  CARD_INDEX_BY_LABEL,
  INITIAL_ACTOR,
  INITIAL_PHASE,
  LEGAL_MASK_BY_PHASE,
  NO_LEGAL_ACTION_MASK,
  OBS_ACTOR_OFFSET,
  OBS_HISTORY_OFFSET,
  OBS_HISTORY_SEQUENCE_TO_INDEX,
  OBS_PRIVATE_CARD_OFFSET,
  OBS_TERMINAL_HISTORY_INDEX,
  OBSERVATION_DIM,
  PLAYER_INDEX_BY_ID,
  RESPONSE_ACTION_PHASES
} from "./generated/contract";

const ACTION_OPEN_LABELS = ACTION_OPEN_LABEL_BY_ID as unknown as Record<number, PublicAction>;
const ACTION_RESPONSE_LABELS = ACTION_RESPONSE_LABEL_BY_ID as unknown as Record<
  number,
  PublicAction
>;
const HISTORY_KEY_TO_INDEX = OBS_HISTORY_SEQUENCE_TO_INDEX as unknown as Record<string, number>;

function nextPlayer(player: PlayerId): PlayerId {
  return player === "player_0" ? "player_1" : "player_0";
}

function assert(condition: boolean, message: string): asserts condition {
  if (!condition) {
    throw new Error(message);
  }
}

export function createInitialState(privateCards: Record<PlayerId, Card>): HandState {
  return {
    phase: INITIAL_PHASE,
    actor: INITIAL_ACTOR,
    privateCards: { ...privateCards },
    history: [],
    contributions: { player_0: 1, player_1: 1 },
    lastBettor: null,
    terminalReturns: null
  };
}

export function dealRandomCards(rng: () => number = Math.random): Record<PlayerId, Card> {
  const deck = [...CARDS];
  for (let i = deck.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = deck[i];
    deck[i] = deck[j];
    deck[j] = tmp;
  }
  return { player_0: deck[0], player_1: deck[1] };
}

export function createRandomInitialState(rng: () => number = Math.random): HandState {
  return createInitialState(dealRandomCards(rng));
}

export function legalActionMask(state: HandState, viewer: PlayerId): ActionMask {
  if (state.terminalReturns !== null || state.actor !== viewer) {
    return [...NO_LEGAL_ACTION_MASK] as ActionMask;
  }
  return [...LEGAL_MASK_BY_PHASE[state.phase]] as ActionMask;
}

function actionToken(action: ActionId, phase: HandState["phase"]): PublicAction {
  if (RESPONSE_ACTION_PHASES.includes(phase as (typeof RESPONSE_ACTION_PHASES)[number])) {
    return ACTION_RESPONSE_LABELS[action];
  }
  return ACTION_OPEN_LABELS[action];
}

function showdownWinner(privateCards: Record<PlayerId, Card>): PlayerId {
  const p0 = CARD_INDEX_BY_LABEL[privateCards.player_0];
  const p1 = CARD_INDEX_BY_LABEL[privateCards.player_1];
  return p0 > p1 ? "player_0" : "player_1";
}

function terminalReturns(
  winner: PlayerId,
  contributions: Record<PlayerId, number>
): Record<PlayerId, number> {
  const loser = nextPlayer(winner);
  const pot = contributions.player_0 + contributions.player_1;
  return {
    [winner]: pot - contributions[winner],
    [loser]: -contributions[loser]
  } as Record<PlayerId, number>;
}

function historyIndex(history: readonly PublicAction[]): number {
  const key = history.join("|");
  const bucketIndex = HISTORY_KEY_TO_INDEX[key];
  return bucketIndex ?? OBS_TERMINAL_HISTORY_INDEX;
}

export function buildObservation(state: HandState, viewer: PlayerId): number[] {
  const obs = new Array<number>(OBSERVATION_DIM).fill(0);
  obs[OBS_PRIVATE_CARD_OFFSET + CARD_INDEX_BY_LABEL[state.privateCards[viewer]]] = 1;
  obs[OBS_HISTORY_OFFSET + historyIndex(state.history)] = 1;

  if (state.actor !== null) {
    obs[OBS_ACTOR_OFFSET + PLAYER_INDEX_BY_ID[state.actor]] = 1;
  }
  return obs;
}

export function applyAction(state: HandState, action: ActionId): HandState {
  assert(state.actor !== null, "Cannot act in terminal state.");
  const actor = state.actor;
  const mask = legalActionMask(state, actor);
  assert(mask[action] === 1, `Illegal action ${action} for phase ${state.phase}.`);

  const token = actionToken(action, state.phase);
  const nextHistory = [...state.history, token];
  const nextContrib = { ...state.contributions };
  const nextLastBettor = token === "bet" ? actor : state.lastBettor;

  if (token === "bet" || token === "call") {
    nextContrib[actor] += 1;
  }

  if (state.phase === "p0_act") {
    if (token === "check") {
      return {
        ...state,
        phase: "p1_act",
        actor: "player_1",
        history: nextHistory,
        contributions: nextContrib,
        lastBettor: nextLastBettor
      };
    }
    if (token === "bet") {
      return {
        ...state,
        phase: "p1_response",
        actor: "player_1",
        history: nextHistory,
        contributions: nextContrib,
        lastBettor: nextLastBettor
      };
    }
  }

  if (state.phase === "p1_act") {
    if (token === "check") {
      const winner = showdownWinner(state.privateCards);
      return {
        ...state,
        phase: "terminal",
        actor: null,
        history: nextHistory,
        contributions: nextContrib,
        lastBettor: nextLastBettor,
        terminalReturns: terminalReturns(winner, nextContrib)
      };
    }
    if (token === "bet") {
      return {
        ...state,
        phase: "p0_response",
        actor: "player_0",
        history: nextHistory,
        contributions: nextContrib,
        lastBettor: nextLastBettor
      };
    }
  }

  if (state.phase === "p0_response" || state.phase === "p1_response") {
    const winner =
      token === "call"
        ? showdownWinner(state.privateCards)
        : (() => {
            assert(nextLastBettor !== null, "Fold resolution requires last bettor.");
            return nextLastBettor;
          })();
    return {
      ...state,
      phase: "terminal",
      actor: null,
      history: nextHistory,
      contributions: nextContrib,
      lastBettor: nextLastBettor,
      terminalReturns: terminalReturns(winner, nextContrib)
    };
  }

  throw new Error(`Unhandled transition: phase=${state.phase}, token=${token}.`);
}

export function chooseBotActionFromMaskedLogits(
  maskedLogits: readonly number[],
  mask: ActionMask,
  mode: BotDecisionMode,
  rng: () => number = Math.random
): ActionId {
  const legalIds = Array.from({ length: ACTION_DIM }, (_, actionId) => actionId).filter(
    (actionId) => mask[actionId as ActionId] === 1
  ) as ActionId[];
  assert(legalIds.length > 0, "No legal actions available.");

  if (mode === "deterministic") {
    let best = legalIds[0];
    let bestLogit = maskedLogits[best];
    for (const action of legalIds.slice(1)) {
      const score = maskedLogits[action];
      if (score > bestLogit || (score === bestLogit && action < best)) {
        best = action;
        bestLogit = score;
      }
    }
    return best;
  }

  const legalLogits = legalIds.map((id) => maskedLogits[id]);
  const maxLogit = Math.max(...legalLogits);
  const exp = legalLogits.map((value) => Math.exp(value - maxLogit));
  const total = exp.reduce((sum, value) => sum + value, 0);
  const threshold = rng() * total;

  let cumulative = 0;
  for (let i = 0; i < legalIds.length; i += 1) {
    cumulative += exp[i];
    if (threshold <= cumulative) {
      return legalIds[i];
    }
  }
  return legalIds[legalIds.length - 1];
}
