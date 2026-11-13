// The four networks of the hierarchy, run with onnxruntime-web (public/models/*.onnx, from tools/export_onnx.py).
// `ort` is the onnxruntime-web module and `model` the bytes or the URL of an ONNX file.

const SESSION_OPTIONS = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
export const LOW_LEVEL_HIDDEN = 512; // Memory size of the jump and crawl policies.
export const HIGH_LEVEL_HIDDEN = 256; // Memory size of the high-level policy.

/**
 * A recurrent actor (GRU): the two low-level policies (40 inputs, 7 actions) and the high-level policy (464 inputs,
 * 3 actions). It keeps its memory between calls; `reset()` clears it at the start of an episode.
 */
export class RecurrentPolicy {
  static async create(ort, model, hiddenSize) {
    const session = await ort.InferenceSession.create(model, SESSION_OPTIONS);
    return new RecurrentPolicy(ort, session, hiddenSize);
  }

  constructor(ort, session, hiddenSize) {
    this.ort = ort;
    this.session = session;
    this.hiddenSize = hiddenSize;
    [this.observationName, this.hiddenName] = session.inputNames;
    [this.actionName, this.nextHiddenName] = session.outputNames;
    this.reset();
  }

  reset() {
    this.hidden = new this.ort.Tensor('float32', new Float32Array(this.hiddenSize), [1, 1, this.hiddenSize]);
  }

  /** @param observation Float32Array; @returns the deterministic action as a Float32Array. */
  async act(observation) {
    const input = new this.ort.Tensor('float32', observation, [1, observation.length]);
    const output = await this.session.run({ [this.observationName]: input, [this.hiddenName]: this.hidden });
    this.hidden = output[this.nextHiddenName];
    return output[this.actionName].data;
  }
}

/** The frozen DeFM depth encoder: a 60 x 80 depth image in metres (0 = no return) -> 440 features. */
export class DepthEncoder {
  static async create(ort, model) {
    return new DepthEncoder(ort, await ort.InferenceSession.create(model, SESSION_OPTIONS));
  }

  constructor(ort, session) {
    this.ort = ort;
    this.session = session;
  }

  /** @param image Float32Array of 4800 depths, row by row; @returns Float32Array of 440 features. */
  async encode(image) {
    const output = await this.session.run({ depth: new this.ort.Tensor('float32', image, [1, 60, 80]) });
    return output.features.data;
  }
}
