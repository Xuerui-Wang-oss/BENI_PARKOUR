// Check the perception and the networks against the recorded Python episodes, on recorded inputs:
//   depth      the ray-cast image from the recorded state, against the image mjlab rendered
//   encoder    DeFM in onnxruntime-web on the recorded image, against the recorded features
//   policies   the two low-level policies on the recorded observations of a whole episode (with memory), and the
//              high-level policy on the recorded state numbers with (a) the recorded features, (b) the features of
//              the ONNX encoder on the recorded image, (c) the features of the ONNX encoder on the ray-cast image.
//              (b) and (c) show what the differences in perception do to the high-level action.
//   node tools/check_perception.mjs [env id ...]
import fs from 'node:fs';
import * as ort from 'onnxruntime-web';
import { DECIMATION, Robot } from '../src/sim/robot.js';
import { SUB_STEPS } from '../src/sim/hierarchy.js';
import { DepthCamera } from '../src/sim/depth.js';
import { DepthEncoder, HIGH_LEVEL_HIDDEN, LOW_LEVEL_HIDDEN, RecurrentPolicy } from '../src/sim/policies.js';
import { ROOT, env2SceneXml, loadMujoco, loadRollout, maxAbsDiff, rolloutIds } from './lib.mjs';

ort.env.wasm.numThreads = 1;
const ids = process.argv.length > 2 ? process.argv.slice(2).map(Number) : rolloutIds();
const model = (name) => fs.readFileSync(new URL(`public/models/${name}.onnx`, ROOT));
const mj = await loadMujoco();
const robot = new Robot(mj, env2SceneXml());
const camera = new DepthCamera(robot);
const encoder = await DepthEncoder.create(ort, model('defm'));
const crawl = await RecurrentPolicy.create(ort, model('crawl'), LOW_LEVEL_HIDDEN);
const jump = await RecurrentPolicy.create(ort, model('jump'), LOW_LEVEL_HIDDEN);
const highLevel = { recorded: null, onnx: null, rays: null };
for (const key of Object.keys(highLevel)) highLevel[key] = await RecurrentPolicy.create(ort, model('policy'), HIGH_LEVEL_HIDDEN);

const summary = (values) => {
  const sorted = Float64Array.from(values).sort();
  const at = (q) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
  return `median ${at(0.5).toExponential(1)}, 99% ${at(0.99).toExponential(1)}, max ${at(1).toExponential(1)}`;
};
const mean = (values) => values.reduce((a, b) => a + b, 0) / values.length;

for (const id of ids) {
  const rollout = loadRollout(id);
  const { info, steps } = rollout;
  const highSteps = Math.ceil(steps / SUB_STEPS);
  console.log(`\nenv ${id} (row ${info.row}, column ${info.column}): ${steps} control steps, ${highSteps} high-level steps`);

  for (const [name, policy, observations, actions] of [['crawl', crawl, 'crawl_obs', 'crawl_action'], ['jump', jump, 'jump_obs', 'jump_action']]) {
    policy.reset();
    const gaps = [];
    for (let t = 0; t < steps; t++) gaps.push(maxAbsDiff(await policy.act(rollout.row(observations, t)), rollout.row(actions, t)));
    console.log(`  ${name} policy, action difference over the episode: ${summary(gaps)}`);
  }

  const pixelGaps = [];
  let offPixels = 0;
  const featureGaps = { onnx: [], rays: [] };
  const actionGaps = { recorded: [], onnx: [], rays: [] };
  const triggers = { recorded: 0, onnx: 0, rays: 0 };
  let pythonTriggers = 0;
  for (const policy of Object.values(highLevel)) policy.reset();
  for (let h = 0; h < highSteps; h++) {
    const p = h * SUB_STEPS * DECIMATION; // The physics step that the high-level step starts with.
    robot.setState(p ? rollout.row('qpos', p - 1) : rollout.row('qpos0', 0), p ? rollout.row('qvel', p - 1) : rollout.row('qvel0', 0));
    const reference = rollout.row('depth', h);
    const image = Float32Array.from(camera.render());
    for (let i = 0; i < image.length; i++) {
      const gap = Math.abs(image[i] - reference[i]);
      pixelGaps.push(gap);
      if (gap > 0.01) offPixels++;
    }
    const features = {
      recorded: rollout.row('features', h),
      onnx: Float32Array.from(await encoder.encode(Float32Array.from(reference))),
      rays: Float32Array.from(await encoder.encode(image)),
    };
    const state = rollout.row('hl_obs', h);
    const python = rollout.row('hl_action', h);
    pythonTriggers += python[2] > 1;
    for (const key of Object.keys(highLevel)) {
      if (key !== 'recorded') featureGaps[key].push(mean(Array.from(features[key], (v, i) => Math.abs(v - features.recorded[i]))));
      const input = new Float32Array(464);
      input.set(state, 0);
      input.set(features[key], 24);
      const action = await highLevel[key].act(input);
      actionGaps[key].push(maxAbsDiff(action, python));
      triggers[key] += (action[2] > 1) === (python[2] > 1);
    }
  }
  console.log(`  depth image, per pixel (m): ${summary(pixelGaps)}; ${(100 * offPixels / pixelGaps.length).toFixed(3)}% of pixels differ by more than 1 cm`);
  console.log(`  encoder features, mean difference per image: ONNX on the recorded image ${summary(featureGaps.onnx)}`);
  console.log(`                                               ONNX on the ray-cast image ${summary(featureGaps.rays)}`);
  for (const [key, label] of [['recorded', 'recorded features'], ['onnx', 'ONNX encoder, recorded image'], ['rays', 'ONNX encoder, ray-cast image']]) {
    console.log(`  high-level action with ${label}: ${summary(actionGaps[key])}; trigger equal on ${triggers[key]} of ${highSteps} steps`);
  }
  console.log(`  (Python pressed the trigger on ${pythonTriggers} high-level steps)`);
}
