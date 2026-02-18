import { describe, expect, it } from "vitest";

import {
  ACTION_BET,
  ACTION_CHECK_OR_CALL,
  ACTION_FOLD,
  type HandState
} from "./types";
import {
  applyAction,
  buildObservation,
  chooseBotActionFromMaskedLogits,
  createInitialState,
  legalActionMask
} from "./engine";

function stateWithCards(p0: "J" | "Q" | "K", p1: "J" | "Q" | "K"): HandState {
  return createInitialState({ player_0: p0, player_1: p1 });
}

describe("engine", () => {
  it("matches legal masks by phase and actor", () => {
    let state = stateWithCards("Q", "K");
    expect(legalActionMask(state, "player_0")).toEqual([1, 1, 0]);
    expect(legalActionMask(state, "player_1")).toEqual([0, 0, 0]);

    state = applyAction(state, ACTION_CHECK_OR_CALL);
    expect(state.phase).toBe("p1_act");
    expect(legalActionMask(state, "player_1")).toEqual([1, 1, 0]);

    state = applyAction(state, ACTION_BET);
    expect(state.phase).toBe("p0_response");
    expect(legalActionMask(state, "player_0")).toEqual([1, 0, 1]);
  });

  it("resolves terminal returns for check-check showdown", () => {
    let state = stateWithCards("K", "J");
    state = applyAction(state, ACTION_CHECK_OR_CALL);
    state = applyAction(state, ACTION_CHECK_OR_CALL);

    expect(state.phase).toBe("terminal");
    expect(state.history).toEqual(["check", "check"]);
    expect(state.terminalReturns).toEqual({ player_0: 1, player_1: -1 });
  });

  it("resolves bet-fold and bet-call returns", () => {
    let foldState = stateWithCards("J", "K");
    foldState = applyAction(foldState, ACTION_BET);
    foldState = applyAction(foldState, ACTION_FOLD);

    expect(foldState.history).toEqual(["bet", "fold"]);
    expect(foldState.terminalReturns).toEqual({ player_0: 1, player_1: -1 });

    let callState = stateWithCards("J", "K");
    callState = applyAction(callState, ACTION_BET);
    callState = applyAction(callState, ACTION_CHECK_OR_CALL);

    expect(callState.history).toEqual(["bet", "call"]);
    expect(callState.terminalReturns).toEqual({ player_0: -2, player_1: 2 });
  });

  it("builds observation encoding compatible with Python contract", () => {
    let state = stateWithCards("K", "J");

    expect(buildObservation(state, "player_0")).toEqual([0, 0, 1, 1, 0, 0, 0, 0, 1, 0]);
    expect(buildObservation(state, "player_1")).toEqual([1, 0, 0, 1, 0, 0, 0, 0, 1, 0]);

    state = applyAction(state, ACTION_CHECK_OR_CALL);
    expect(buildObservation(state, "player_1")).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  });

  it("chooses deterministic action by max legal logit", () => {
    const action = chooseBotActionFromMaskedLogits(
      [-0.4, 2.0, -1e9],
      [1, 1, 0],
      "deterministic"
    );
    expect(action).toBe(ACTION_BET);
  });

  it("samples stochastic action from legal logits only", () => {
    const logits = [0.0, -1e9, 0.0];
    const mask = [1, 0, 1] as const;

    const actionLow = chooseBotActionFromMaskedLogits(logits, mask, "stochastic", () => 0.1);
    const actionHigh = chooseBotActionFromMaskedLogits(logits, mask, "stochastic", () => 0.9);

    expect(actionLow).toBe(ACTION_CHECK_OR_CALL);
    expect(actionHigh).toBe(ACTION_FOLD);
  });
});
