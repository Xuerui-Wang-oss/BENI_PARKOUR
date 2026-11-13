// Run the exported ONNX networks with onnxruntime-web (the runtime the browser uses) on the test vectors of
// export_onnx.py, compare with the PyTorch outputs, and time each network.
//   node tools/check_onnx.mjs [threads]
import fs from 'node:fs';
import * as ort from 'onnxruntime-web';

const threads = Number(process.argv[2] ?? 1);
ort.env.wasm.numThreads = threads;
const root = new URL('..', import.meta.url);
const vectors = JSON.parse(fs.readFileSync(new URL('tools/ref/onnx_vectors.json', root)));
const load = (name) => ort.InferenceSession.create(
  fs.readFileSync(new URL(`public/models/${name}.onnx`, root)),
  { executionProviders: ['wasm'], graphOptimizationLevel: 'all' },
);
const maxGap = (a, b) => a.reduce((worst, v, i) => Math.max(worst, Math.abs(v - b[i])), 0);
let failed = false;

for (const name of ['jump', 'crawl', 'policy']) {
  const session = await load(name);
  const { obs, actions, hidden_size: hiddenSize } = vectors[name];
  let hidden = new ort.Tensor('float32', new Float32Array(hiddenSize), [1, 1, hiddenSize]);
  let worst = 0;
  const start = performance.now();
  for (let t = 0; t < obs.length; t++) {
    const out = await session.run({
      [session.inputNames[0]]: new ort.Tensor('float32', Float32Array.from(obs[t]), [1, obs[t].length]),
      [session.inputNames[1]]: hidden,
    });
    hidden = out[session.outputNames[1]];
    worst = Math.max(worst, maxGap(Array.from(out[session.outputNames[0]].data), actions[t]));
  }
  const each = (performance.now() - start) / obs.length;
  console.log(`${name}: inputs ${session.inputNames}, outputs ${session.outputNames}; ${obs.length} steps with memory, `
    + `largest action difference ${worst.toExponential(2)}, ${each.toFixed(2)} ms per step`);
  failed ||= worst > 1e-3;
}

{
  const session = await load('defm');
  const { depth, features } = vectors.defm;
  // The encoder is 60 layers deep with a GroupNorm after each convolution, which amplifies the rounding differences
  // between two runtimes, most on an image without any structure (the last one is all zeros). The mean difference
  // is the criterion here; what such differences do to the policy's actions is measured on rollouts.
  let worst = 0;
  let worstMean = 0;
  const run = (image) => session.run({ depth: new ort.Tensor('float32', Float32Array.from(image), [1, 60, 80]) });
  for (let i = 0; i < depth.length; i++) {
    const out = Array.from((await run(depth[i])).features.data);
    worst = Math.max(worst, maxGap(out, features[i]));
    worstMean = Math.max(worstMean, out.reduce((sum, v, k) => sum + Math.abs(v - features[i][k]), 0) / out.length);
  }
  const repeats = 20;
  const start = performance.now();
  for (let i = 0; i < repeats; i++) await run(depth[i % depth.length]);
  const each = (performance.now() - start) / repeats;
  console.log(`defm: ${depth.length} images, feature difference: mean ${worstMean.toExponential(2)}, `
    + `largest ${worst.toExponential(2)} (features are about 0.4 in size); `
    + `${each.toFixed(1)} ms per image with ${threads} thread(s), ${(each * 20 / 10).toFixed(0)}% of real time at 20 Hz`);
  failed ||= worstMean > 2e-3;
}
if (failed) throw new Error('an ONNX network differs from PyTorch');
console.log('all networks match PyTorch');
