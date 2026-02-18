import { useCallback, useEffect, useMemo, useRef, useState } from "react"

import { applyAction, buildObservation, createRandomInitialState, legalActionMask } from "./game/engine"
import { OnnxPolicyAdapter } from "./game/onnx_adapter"
import robotImage from "./assets/robot.png"
import {
  ACTION_BET,
  ACTION_CHECK_OR_CALL,
  ACTION_FOLD,
  type ActionId,
  type ActionMask,
  type BotDecisionMode,
  type HandState,
  type Phase,
  type PlayerId
} from "./game/types"

const DEFAULT_MODEL_URL = "models/kuhn_policy.onnx"
const ACTION_IDS: readonly ActionId[] = [ACTION_CHECK_OR_CALL, ACTION_BET, ACTION_FOLD]

type LoadStatus = "idle" | "loading" | "ready" | "error"

function opponentOf(player: PlayerId): PlayerId {
  return player === "player_0" ? "player_1" : "player_0"
}

function displayPlayer(player: PlayerId): string {
  return player === "player_0" ? "Player 0" : "Player 1"
}

function formatChipDelta(value: number): string {
  return value >= 0 ? `+${value}` : `${value}`
}

function actionLabel(action: ActionId, phase: Phase): string {
  if (action === ACTION_CHECK_OR_CALL) {
    return phase === "p0_response" || phase === "p1_response" ? "Call" : "Check"
  }
  if (action === ACTION_BET) {
    return "Bet"
  }
  return "Fold"
}

function actionCode(action: ActionId): string {
  if (action === ACTION_CHECK_OR_CALL) {
    return "c"
  }
  if (action === ACTION_BET) {
    return "b"
  }
  return "f"
}

function pickRandomLegalAction(mask: ActionMask, rng: () => number = Math.random): ActionId {
  const legal = ACTION_IDS.filter((id) => mask[id] === 1)
  if (legal.length === 0) {
    throw new Error("No legal action available for fallback policy.")
  }
  const index = Math.floor(rng() * legal.length)
  return legal[index]
}

function formatHistory(state: HandState): string {
  return state.history.length === 0 ? "(none)" : state.history.join(" -> ")
}

function formatReturns(state: HandState, humanSeat: PlayerId): string {
  if (state.terminalReturns === null) {
    return "(pending)"
  }
  const botSeat = opponentOf(humanSeat)
  return `You ${formatChipDelta(state.terminalReturns[humanSeat])}, Bot ${formatChipDelta(state.terminalReturns[botSeat])}`
}

export default function App(): JSX.Element {
  const [humanSeat, setHumanSeat] = useState<PlayerId>("player_0")
  const botSeat = opponentOf(humanSeat)

  const [botMode, setBotMode] = useState<BotDecisionMode>("deterministic")
  const [showHotkeys, setShowHotkeys] = useState(true)
  const [state, setState] = useState<HandState>(() => createRandomInitialState())
  const [handNumber, setHandNumber] = useState(1)
  const [score, setScore] = useState<Record<PlayerId, number>>({ player_0: 0, player_1: 0 })
  const [lastScoredHand, setLastScoredHand] = useState<number | null>(null)

  const [modelStatus, setModelStatus] = useState<LoadStatus>("idle")
  const [modelStatusText, setModelStatusText] = useState("Model not loaded. Using random fallback.")
  const [adapter, setAdapter] = useState<OnnxPolicyAdapter | null>(null)
  const loadRequestRef = useRef(0)

  const isTerminal = state.phase === "terminal"
  const humanCanAct = !isTerminal && state.actor === humanSeat
  const humanMask = legalActionMask(state, humanSeat)
  const botMask = legalActionMask(state, botSeat)
  const legalHumanActions = ACTION_IDS.filter((id) => humanMask[id] === 1)

  const loadModel = useCallback(async (): Promise<void> => {
    const requestId = loadRequestRef.current + 1
    loadRequestRef.current = requestId
    setModelStatus("loading")
    setModelStatusText("Loading model ...")

    try {
      const loadedAdapter = await OnnxPolicyAdapter.create({ modelSource: DEFAULT_MODEL_URL })
      if (requestId !== loadRequestRef.current) {
        return
      }
      setAdapter(loadedAdapter)
      setModelStatus("ready")
      setModelStatusText("Model ready.")
    } catch (error) {
      if (requestId !== loadRequestRef.current) {
        return
      }
      const detail = error instanceof Error ? error.message : String(error)
      setAdapter(null)
      setModelStatus("error")
      setModelStatusText(`Model load failed (${detail}). Using random fallback policy.`)
    }
  }, [])

  useEffect(() => {
    void loadModel()
  }, [loadModel])

  useEffect(() => {
    const returns = state.terminalReturns
    if (state.phase !== "terminal" || returns === null) {
      return
    }
    if (lastScoredHand === handNumber) {
      return
    }
    setScore((current) => ({
      player_0: current.player_0 + returns.player_0,
      player_1: current.player_1 + returns.player_1
    }))
    setLastScoredHand(handNumber)
  }, [handNumber, lastScoredHand, state])

  useEffect(() => {
    if (state.phase === "terminal" || state.actor !== botSeat) {
      return
    }

    let cancelled = false
    const runBotTurn = async (): Promise<void> => {
      const mask = botMask
      let action: ActionId

      if (adapter !== null) {
        try {
          const observation = buildObservation(state, botSeat)
          action = await adapter.chooseAction(observation, mask, botMode)
        } catch (error) {
          const detail = error instanceof Error ? error.message : String(error)
          setAdapter(null)
          setModelStatus("error")
          setModelStatusText(`Inference failed (${detail}). Falling back to random legal policy.`)
          action = pickRandomLegalAction(mask)
        }
      } else {
        action = pickRandomLegalAction(mask)
      }

      await new Promise((resolve) => setTimeout(resolve, 250))
      if (cancelled) {
        return
      }
      setState((current) => {
        if (current.phase === "terminal" || current.actor !== botSeat) {
          return current
        }
        return applyAction(current, action)
      })
    }

    void runBotTurn()
    return () => {
      cancelled = true
    }
  }, [adapter, botMask, botMode, botSeat, state])

  const startNextHand = useCallback((): void => {
    setHandNumber((current) => current + 1)
    setLastScoredHand(null)
    setState(createRandomInitialState())
  }, [])

  const resetMatch = useCallback((nextSeat?: PlayerId): void => {
    if (nextSeat !== undefined) {
      setHumanSeat(nextSeat)
    }
    setScore({ player_0: 0, player_1: 0 })
    setHandNumber(1)
    setLastScoredHand(null)
    setState(createRandomInitialState())
  }, [])

  const onHumanAction = useCallback(
    (action: ActionId): void => {
      if (!humanCanAct || humanMask[action] !== 1) {
        return
      }
      setState((current) => applyAction(current, action))
    },
    [humanCanAct, humanMask]
  )

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent): void => {
      const target = event.target as HTMLElement | null
      if (
        target !== null &&
        (target.tagName === "INPUT" ||
          target.tagName === "SELECT" ||
          target.tagName === "TEXTAREA" ||
          target.isContentEditable)
      ) {
        return
      }

      const key = event.key.toLowerCase()
      if (key === "0") {
        event.preventDefault()
        resetMatch("player_0")
        return
      }
      if (key === "1") {
        event.preventDefault()
        resetMatch("player_1")
        return
      }
      if (key === "c") {
        event.preventDefault()
        onHumanAction(ACTION_CHECK_OR_CALL)
        return
      }
      if (key === "b") {
        event.preventDefault()
        onHumanAction(ACTION_BET)
        return
      }
      if (key === "f") {
        event.preventDefault()
        onHumanAction(ACTION_FOLD)
        return
      }
      if (key === "n") {
        event.preventDefault()
        if (isTerminal) {
          startNextHand()
        }
        return
      }
      if (key === "r") {
        event.preventDefault()
        resetMatch()
        return
      }
      if (key === "h") {
        event.preventDefault()
        setShowHotkeys((current) => !current)
      }
    }

    window.addEventListener("keydown", onKeyDown)
    return () => {
      window.removeEventListener("keydown", onKeyDown)
    }
  }, [isTerminal, onHumanAction, resetMatch, startNextHand])

  const botPolicyLabel = adapter === null ? "random legal fallback" : botMode
  const humanCard = state.privateCards[humanSeat]
  const botCard = state.phase === "terminal" ? state.privateCards[botSeat] : "?"

  const publicPot = state.contributions.player_0 + state.contributions.player_1
  const humanLabel = displayPlayer(humanSeat)
  const botLabel = displayPlayer(botSeat)
  const scoreText = `You: ${score[humanSeat]} | Bot: ${score[botSeat]}`
  const lastHandText = useMemo(() => {
    if (state.phase !== "terminal") {
      return "Hand in progress."
    }
    return `Hand ${handNumber} complete. ${formatReturns(state, humanSeat)}`
  }, [handNumber, humanSeat, state])

  return (
    <main className="app-shell">
      <section className="panel controls side-panel">
        <h1>Kuhn Poker Bot</h1>
        <p className="subhead">Play Kuhn Poker against a reinforcement learning trained bot</p>

        <div className="control-group">
          <span className="label">Play as</span>
          <div className="segmented">
            <button
              type="button"
              className={humanSeat === "player_0" ? "active" : ""}
              onClick={() => resetMatch("player_0")}
            >
              Player 0 [0]
            </button>
            <button
              type="button"
              className={humanSeat === "player_1" ? "active" : ""}
              onClick={() => resetMatch("player_1")}
            >
              Player 1 [1]
            </button>
          </div>
        </div>

        <div className="control-group">
          <label htmlFor="bot-mode" className="label">
            Bot mode
          </label>
          <select
            id="bot-mode"
            value={botMode}
            onChange={(event) => setBotMode(event.target.value as BotDecisionMode)}
          >
            <option value="deterministic">deterministic</option>
            <option value="stochastic">stochastic</option>
          </select>
        </div>

        <div className="control-group">
          <p className="label">Model status</p>
          <p className={`status ${modelStatus}`}>{modelStatusText}</p>
        </div>

        <div className="control-group">
          <p className="label">Keyboard</p>
          {showHotkeys ? (
            <ul className="hotkeys">
              <li>
                <code>0</code> / <code>1</code> change player
              </li>
              <li>
                <code>c</code> check/call, <code>b</code> bet, <code>f</code> fold
              </li>
              <li>
                <code>n</code> next hand, <code>r</code> reset match
              </li>
              <li>
                <code>h</code> toggle help
              </li>
            </ul>
          ) : (
            <p className="status">Press h to show keyboard help.</p>
          )}
        </div>
      </section>

      <section className="panel table-panel">
        <div className="table-header">
          <h2>Hand {handNumber}</h2>
          <button type="button" onClick={() => resetMatch()}>
            Reset match (r)
          </button>
        </div>

        <div className="facts">
          <p>
            <span className="label">Score</span>
            <span>{scoreText}</span>
          </p>
          <p>
            <span className="label">Bot policy</span>
            <span>{botPolicyLabel}</span>
          </p>
          <p>
            <span className="label">Pot</span>
            <span>{publicPot}</span>
          </p>
          <p>
            <span className="label">History</span>
            <span>{formatHistory(state)}</span>
          </p>
        </div>

        <div className="cards">
          <article>
            <h3>{humanLabel} (you)</h3>
            <p className="card">{humanCard}</p>
          </article>
          <article>
            <h3>{botLabel} (bot)</h3>
            <p className="card">{botCard}</p>
          </article>
        </div>

        <div className="action-area">
          {humanCanAct ? (
            <div className="actions">
              <p className="label">Your legal actions</p>
              <div className="action-buttons">
                {legalHumanActions.map((action) => (
                  <button key={action} type="button" onClick={() => onHumanAction(action)}>
                    {actionCode(action)}: {actionLabel(action, state.phase)}
                  </button>
                ))}
              </div>
            </div>
          ) : isTerminal ? (
            <div className="terminal">
              <p>{lastHandText}</p>
              <button type="button" onClick={startNextHand}>
                Next hand (n)
              </button>
            </div>
          ) : (
            <p className="waiting">Bot is acting...</p>
          )}
        </div>
      </section>

      <aside className="panel robot-panel side-panel">
        <div className="robot-frame">
          <img src={robotImage} alt="Robot playing Kuhn Poker" />
        </div>
      </aside>
    </main>
  )
}
