import { chooseBotActionFromMaskedLogits } from "./engine"
import type { ActionId, ActionMask, BotDecisionMode } from "./types"
import ortWasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url"
import ortWasmMjsUrl from "onnxruntime-web/ort-wasm-simd-threaded.mjs?url"

const OBSERVATION_DIM = 10
const ACTION_DIM = 3
const OBSERVATION_INPUT = "observation"
const ACTION_MASK_INPUT = "action_mask"
const MASKED_LOGITS_OUTPUT = "masked_logits"
const VALUE_OUTPUT = "value"

export type OnnxModelSource = string | ArrayBuffer | Uint8Array

export interface OrtTensorLike {
  readonly data: ArrayLike<number>
  readonly dims: readonly number[]
}

export interface OrtSessionLike {
  run(feeds: Record<string, OrtTensorLike>): Promise<Record<string, OrtTensorLike>>
}

export interface OrtRuntimeLike {
  readonly Tensor: new (
    type: "float32",
    data: Float32Array,
    dims: readonly number[]
  ) => OrtTensorLike
  readonly InferenceSession: {
    create(modelSource: OnnxModelSource, options?: unknown): Promise<OrtSessionLike>
  }
  readonly env?: {
    wasm?: {
      wasmPaths?: string | { wasm?: string; mjs?: string }
      numThreads?: number
    }
  }
}

export interface OnnxPolicyAdapterOptions {
  readonly modelSource: OnnxModelSource
  readonly runtimeLoader?: () => Promise<OrtRuntimeLike>
  readonly sessionOptions?: unknown
}

export interface OnnxInferenceResult {
  readonly maskedLogits: Float32Array
  readonly value: Float32Array
}

function assert(condition: boolean, message: string): asserts condition {
  if (!condition) {
    throw new Error(message)
  }
}

function toFloat32Array(values: ArrayLike<number>): Float32Array {
  const out = new Float32Array(values.length)
  for (let i = 0; i < values.length; i += 1) {
    out[i] = values[i]
  }
  return out
}

function requireOutput(
  outputs: Record<string, OrtTensorLike>,
  outputName: string
): OrtTensorLike {
  const output = outputs[outputName]
  assert(output !== undefined, `ONNX output missing: "${outputName}".`)
  return output
}

function validateInputSizes(observation: readonly number[], actionMask: ActionMask): void {
  assert(
    observation.length === OBSERVATION_DIM,
    `Observation must have length ${OBSERVATION_DIM}, got ${observation.length}.`
  )
  assert(
    actionMask.length === ACTION_DIM,
    `Action mask must have length ${ACTION_DIM}, got ${actionMask.length}.`
  )
  const legalCount = actionMask.reduce((sum, value) => sum + (value > 0.5 ? 1 : 0), 0)
  assert(legalCount > 0, "Action mask has no legal actions.")
}

export async function loadOrtRuntime(): Promise<OrtRuntimeLike> {
  const runtime = (await import("onnxruntime-web/wasm")) as OrtRuntimeLike
  if (runtime.env?.wasm !== undefined) {
    runtime.env.wasm.wasmPaths = {
      wasm: ortWasmUrl,
      mjs: ortWasmMjsUrl
    }
  }
  return runtime
}

export class OnnxPolicyAdapter {
  private readonly runtime: OrtRuntimeLike
  private readonly session: OrtSessionLike

  private constructor(runtime: OrtRuntimeLike, session: OrtSessionLike) {
    this.runtime = runtime
    this.session = session
  }

  static async create(options: OnnxPolicyAdapterOptions): Promise<OnnxPolicyAdapter> {
    const runtimeLoader = options.runtimeLoader ?? loadOrtRuntime
    const runtime = await runtimeLoader()
    const session = await runtime.InferenceSession.create(
      options.modelSource,
      options.sessionOptions
    )
    return new OnnxPolicyAdapter(runtime, session)
  }

  async infer(
    observation: readonly number[],
    actionMask: ActionMask
  ): Promise<OnnxInferenceResult> {
    validateInputSizes(observation, actionMask)

    const observationTensor = new this.runtime.Tensor(
      "float32",
      Float32Array.from(observation),
      [1, OBSERVATION_DIM]
    )
    const actionMaskTensor = new this.runtime.Tensor(
      "float32",
      Float32Array.from(actionMask),
      [1, ACTION_DIM]
    )

    const outputs = await this.session.run({
      [OBSERVATION_INPUT]: observationTensor,
      [ACTION_MASK_INPUT]: actionMaskTensor
    })

    const maskedLogitsOutput = requireOutput(outputs, MASKED_LOGITS_OUTPUT)
    const valueOutput = requireOutput(outputs, VALUE_OUTPUT)

    const maskedLogits = toFloat32Array(maskedLogitsOutput.data)
    const value = toFloat32Array(valueOutput.data)

    assert(
      maskedLogits.length === ACTION_DIM,
      `Expected ${ACTION_DIM} masked logits values, got ${maskedLogits.length}.`
    )
    assert(value.length === 1, `Expected value output length 1, got ${value.length}.`)

    return { maskedLogits, value }
  }

  async chooseAction(
    observation: readonly number[],
    actionMask: ActionMask,
    mode: BotDecisionMode,
    rng: () => number = Math.random
  ): Promise<ActionId> {
    const result = await this.infer(observation, actionMask)
    return chooseBotActionFromMaskedLogits(Array.from(result.maskedLogits), actionMask, mode, rng)
  }
}
