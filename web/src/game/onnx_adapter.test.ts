import { describe, expect, it } from "vitest"

import { ACTION_BET, ACTION_CHECK_OR_CALL, ACTION_FOLD } from "./types"
import {
  OnnxPolicyAdapter,
  type OnnxModelSource,
  type OrtRuntimeLike,
  type OrtSessionLike,
  type OrtTensorLike
} from "./onnx_adapter"

class FakeTensor implements OrtTensorLike {
  readonly data: Float32Array
  readonly dims: readonly number[]
  readonly type: string

  constructor(type: "float32", data: Float32Array, dims: readonly number[]) {
    this.type = type
    this.data = data
    this.dims = [...dims]
  }
}

class FakeSession implements OrtSessionLike {
  readonly outputs: Record<string, OrtTensorLike>
  lastFeeds: Record<string, OrtTensorLike> | null = null

  constructor(outputs: Record<string, OrtTensorLike>) {
    this.outputs = outputs
  }

  async run(feeds: Record<string, OrtTensorLike>): Promise<Record<string, OrtTensorLike>> {
    this.lastFeeds = feeds
    return this.outputs
  }
}

function tensor(values: readonly number[], dims: readonly number[]): OrtTensorLike {
  return new FakeTensor("float32", Float32Array.from(values), dims)
}

function makeRuntime(
  outputs: Record<string, OrtTensorLike>
): { runtime: OrtRuntimeLike; session: FakeSession } {
  const session = new FakeSession(outputs)
  const runtime: OrtRuntimeLike = {
    Tensor: FakeTensor,
    InferenceSession: {
      create: async (_modelSource: OnnxModelSource) => session
    }
  }
  return { runtime, session }
}

const validObservation = [1, 0, 0, 1, 0, 0, 0, 0, 1, 0] as const

describe("onnx adapter", () => {
  it("runs inference with contract names and tensor shapes", async () => {
    const { runtime, session } = makeRuntime({
      masked_logits: tensor([0.25, 1.5, -1e9], [1, 3]),
      value: tensor([0.42], [1, 1])
    })
    const adapter = await OnnxPolicyAdapter.create({
      modelSource: new Uint8Array([0, 1, 2]),
      runtimeLoader: async () => runtime
    })

    const result = await adapter.infer(validObservation, [1, 1, 0])

    expect(Array.from(result.maskedLogits)).toEqual([0.25, 1.5, -1e9])
    expect(result.value[0]).toBeCloseTo(0.42)
    expect(session.lastFeeds).not.toBeNull()
    expect(session.lastFeeds?.observation.dims).toEqual([1, 10])
    expect(session.lastFeeds?.action_mask.dims).toEqual([1, 3])
    expect(Array.from(session.lastFeeds?.action_mask.data ?? [])).toEqual([1, 1, 0])
  })

  it("chooses deterministic action from ONNX masked logits", async () => {
    const { runtime } = makeRuntime({
      masked_logits: tensor([-0.5, 0.7, -1e9], [1, 3]),
      value: tensor([0.0], [1, 1])
    })
    const adapter = await OnnxPolicyAdapter.create({
      modelSource: new Uint8Array([5]),
      runtimeLoader: async () => runtime
    })

    const action = await adapter.chooseAction(validObservation, [1, 1, 0], "deterministic")
    expect(action).toBe(ACTION_BET)
  })

  it("samples stochastic action from legal logits only", async () => {
    const { runtime } = makeRuntime({
      masked_logits: tensor([0.0, -1e9, 0.0], [1, 3]),
      value: tensor([0.0], [1, 1])
    })
    const adapter = await OnnxPolicyAdapter.create({
      modelSource: new Uint8Array([9]),
      runtimeLoader: async () => runtime
    })

    const low = await adapter.chooseAction(validObservation, [1, 0, 1], "stochastic", () => 0.1)
    const high = await adapter.chooseAction(validObservation, [1, 0, 1], "stochastic", () => 0.9)

    expect(low).toBe(ACTION_CHECK_OR_CALL)
    expect(high).toBe(ACTION_FOLD)
  })

  it("rejects invalid observation length", async () => {
    const { runtime } = makeRuntime({
      masked_logits: tensor([0.0, 0.0, 0.0], [1, 3]),
      value: tensor([0.0], [1, 1])
    })
    const adapter = await OnnxPolicyAdapter.create({
      modelSource: new Uint8Array([1]),
      runtimeLoader: async () => runtime
    })

    await expect(adapter.infer([1, 0, 0], [1, 1, 0])).rejects.toThrow(
      "Observation must have length 10"
    )
  })

  it("rejects masks without legal actions", async () => {
    const { runtime } = makeRuntime({
      masked_logits: tensor([0.0, 0.0, 0.0], [1, 3]),
      value: tensor([0.0], [1, 1])
    })
    const adapter = await OnnxPolicyAdapter.create({
      modelSource: new Uint8Array([2]),
      runtimeLoader: async () => runtime
    })

    await expect(adapter.infer(validObservation, [0, 0, 0])).rejects.toThrow(
      "Action mask has no legal actions"
    )
  })

  it("fails fast when masked_logits output is missing", async () => {
    const { runtime } = makeRuntime({
      value: tensor([0.0], [1, 1])
    })
    const adapter = await OnnxPolicyAdapter.create({
      modelSource: new Uint8Array([3]),
      runtimeLoader: async () => runtime
    })

    await expect(adapter.infer(validObservation, [1, 1, 0])).rejects.toThrow(
      'ONNX output missing: "masked_logits".'
    )
  })
})
