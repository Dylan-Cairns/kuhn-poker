import {
  ACTION_BET,
  ACTION_CHECK_OR_CALL,
  ACTION_FOLD,
  CARDS,
  PLAYERS,
  type ActionId,
  type ActionMask,
  type BotDecisionMode,
  type Card,
  type HandState,
  type PlayerId,
  type PublicAction
} from "./types";

const CARD_INDEX: Record<Card, number> = { J: 0, Q: 1, K: 2 };

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
    phase: "p0_act",
    actor: "player_0",
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
    return [0, 0, 0];
  }
  if (state.phase === "p0_act" || state.phase === "p1_act") {
    return [1, 1, 0];
  }
  if (state.phase === "p0_response" || state.phase === "p1_response") {
    return [1, 0, 1];
  }
  return [0, 0, 0];
}

function actionToken(action: ActionId, phase: HandState["phase"]): PublicAction {
  const response = phase === "p0_response" || phase === "p1_response";
  if (action === ACTION_CHECK_OR_CALL) {
    return response ? "call" : "check";
  }
  if (action === ACTION_BET) {
    return "bet";
  }
  return "fold";
}

function showdownWinner(privateCards: Record<PlayerId, Card>): PlayerId {
  const p0 = CARD_INDEX[privateCards.player_0];
  const p1 = CARD_INDEX[privateCards.player_1];
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
  if (history.length === 0) {
    return 0;
  }
  if (history.length === 1 && history[0] === "check") {
    return 1;
  }
  if (history.length === 1 && history[0] === "bet") {
    return 2;
  }
  if (history.length === 2 && history[0] === "check" && history[1] === "bet") {
    return 3;
  }
  return 4;
}

export function buildObservation(state: HandState, viewer: PlayerId): number[] {
  const obs = new Array<number>(10).fill(0);
  obs[CARD_INDEX[state.privateCards[viewer]]] = 1;
  obs[3 + historyIndex(state.history)] = 1;

  if (state.actor !== null) {
    obs[8 + (state.actor === "player_0" ? 0 : 1)] = 1;
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
  const legalIds = [0, 1, 2].filter((a) => mask[a as ActionId] === 1) as ActionId[];
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
