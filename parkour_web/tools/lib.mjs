// Shared helpers of the Node checks: load the recorded Python episodes and build the env2 world in MuJoCo WASM.
import fs from 'node:fs';
import load from '@mujoco/mujoco';
import { LANE_LENGTH, boxesXml, laneBoxes } from '../src/sim/terrain.js';
import { sceneXml } from '../src/sim/robot.js';

export const ROOT = new URL('..', import.meta.url);
const read = (path) => fs.readFileSync(new URL(path, ROOT));

export const loadMujoco = () => load();

/** The 80 env2 lanes that mjlab compiled: [{row, column, origin, obstacles, geoms}]. */
export const lanes = () => JSON.parse(read('tools/ref/env2_lanes.json'));

/**
 * Scene MJCF with every env2 lane at its place in the Python world. The rows follow each other along x, so the
 * lanes of rows 0-3 are continued by the next row and those of the last row by nothing; `tail` adds a flat lane
 * behind every lane of the last row.
 */
export function env2SceneXml({ tail = false } = {}) {
  const all = lanes();
  let terrain = all.map((lane) => boxesXml(lane.geoms, lane.origin)).join('');
  if (tail) {
    const lastRow = Math.max(...all.map((lane) => lane.row));
    for (const lane of all.filter((l) => l.row === lastRow)) {
      terrain += boxesXml(laneBoxes([], LANE_LENGTH), [lane.origin[0] + LANE_LENGTH, lane.origin[1], lane.origin[2]]);
    }
  }
  return sceneXml(read('public/robot/beni.xml').toString(), terrain);
}

/** Ids of the recorded episodes in tools/ref/rollouts. */
export function rolloutIds() {
  return fs.readdirSync(new URL('tools/ref/rollouts/', ROOT))
    .filter((name) => /^env\d+\.json$/.test(name)).map((name) => Number(name.slice(3, -5))).sort((a, b) => a - b);
}

/**
 * One recorded episode. `info` is its JSON header; `rows(name)` is the number of rows of an array and
 * `row(name, i)` row i as a Float32Array view (flattened if the array has more than two dimensions).
 */
export function loadRollout(id) {
  const info = JSON.parse(read(`tools/ref/rollouts/env${id}.json`));
  const bytes = read(`tools/ref/rollouts/env${id}.bin`);
  const all = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  const width = (name) => info.arrays[name].shape.slice(1).reduce((a, b) => a * b, 1);
  // The recording runs to the end of the high-level step in which the episode ended, so its last control steps can
  // belong to the next episode. `steps` counts the control steps of the recorded episode itself.
  const done = info.arrays.done;
  let steps = done.shape[0];
  for (let t = 0; t < done.shape[0]; t++) {
    if (all[done.offset + t] > 0) { steps = t + 1; break; }
  }
  return {
    info,
    steps,
    rows: (name) => info.arrays[name].shape[0],
    row: (name, i) => {
      const start = info.arrays[name].offset + i * width(name);
      return all.subarray(start, start + width(name));
    },
    state: (i) => Object.fromEntries(info.state_keys.map((key, k) => [key, all[info.arrays.state.offset + i * info.state_keys.length + k]])),
  };
}

export const maxAbsDiff = (a, b) => {
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
  return worst;
};
