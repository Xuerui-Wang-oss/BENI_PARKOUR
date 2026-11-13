// Run the whole JavaScript stack in closed loop on the env2 lanes, as python -m parkour_rl.evaluate does: MuJoCo
// WASM physics, ray-cast depth, the four ONNX networks, deterministic actions, a new random spawn and goal for
// every episode. Prints the counts of parkour_rl.evaluate, to compare with Python.
//   node tools/run_episodes.mjs [episodes per lane = 1] [seed = 0] [shard = 0/1] [out.json]
// A shard "i/n" runs every n-th lane starting at lane i, so that several processes can share the work; with an
// output file the episodes are also written as JSON, and `node tools/run_episodes.mjs report a.json b.json ...`
// prints the report of several files together.
// Environment: ROWS=4 (or 0,1) runs only the lanes of these rows; TAIL=1 adds a flat lane behind the lanes of the
// last row, which in the Python world end in nothing (see env2SceneXml).
import fs from 'node:fs';
import * as ort from 'onnxruntime-web';
import { Robot, seededRandom } from '../src/sim/robot.js';
import { DepthCamera } from '../src/sim/depth.js';
import { DepthEncoder, HIGH_LEVEL_HIDDEN, LOW_LEVEL_HIDDEN, RecurrentPolicy } from '../src/sim/policies.js';
import { Episode } from '../src/sim/episode.js';
import { ROOT, env2SceneXml, lanes, loadMujoco } from './lib.mjs';

const TRAIN_ROWS = 4; // Rows 0-3 are the training layouts, row 4 the test layout.
const MIN_RUN_UP = 1.5;
const SPAWN_MIN_X = -0.5;
const SPAWN_SPREAD_Y = 0.3;

function report(episodes) {
  const line = (list) => {
    const goal = list.filter((e) => e.outcome.includes('goal_reached')).length;
    const complete = list.filter((e) => e.passed === e.obstacles).length;
    const passed = list.reduce((sum, e) => sum + e.passed, 0);
    const reached = list.reduce((sum, e) => sum + Math.min(e.passed + 1, e.obstacles), 0);
    const pct = (a, b) => `${a} (${(100 * a / Math.max(b, 1)).toFixed(0)}%)`;
    return `${list.length} episodes, goal ${pct(goal, list.length)}, all obstacles passed ${pct(complete, list.length)}, `
      + `obstacles passed ${passed}/${reached} (${(100 * passed / Math.max(reached, 1)).toFixed(0)}%)`;
  };
  for (let row = 0; row <= TRAIN_ROWS; row++) console.log(`row ${row}: ${line(episodes.filter((e) => e.row === row))}`);
  console.log(`training layouts (rows 0-3): ${line(episodes.filter((e) => e.row < TRAIN_ROWS))}`);
  console.log(`test layout (row 4):        ${line(episodes.filter((e) => e.row === TRAIN_ROWS))}`);
  console.log(`all lanes: ${line(episodes)}`);
  const byKind = {};
  for (const e of episodes) {
    e.kinds.slice(0, e.passed + 1).forEach((kind, i) => {
      byKind[kind] ??= [0, 0];
      byKind[kind][0] += i < e.passed;
      byKind[kind][1] += 1;
    });
  }
  console.log(`passed / got to, per kind: ${Object.keys(byKind).sort().map((k) => `${k} ${byKind[k][0]}/${byKind[k][1]}`).join(', ')}`);
  const ends = {};
  for (const e of episodes) ends[e.outcome] = (ends[e.outcome] ?? 0) + 1;
  console.log('episodes ended with:', Object.fromEntries(Object.entries(ends).sort((a, b) => b[1] - a[1])));
  const seconds = episodes.reduce((sum, e) => sum + e.seconds, 0);
  const simulated = episodes.reduce((sum, e) => sum + e.steps * 0.01, 0);
  console.log(`simulated ${simulated.toFixed(0)} s in ${seconds.toFixed(0)} s: ${(simulated / seconds).toFixed(2)} x real time (one thread)`);
}

if (process.argv[2] === 'report') {
  report(process.argv.slice(3).flatMap((file) => JSON.parse(fs.readFileSync(file))));
  process.exit(0);
}

const perLane = Number(process.argv[2] ?? 1);
const seed = Number(process.argv[3] ?? 0);
const [shard, shards] = (process.argv[4] ?? '0/1').split('/').map(Number);
const output = process.argv[5];

ort.env.wasm.numThreads = 1;
const model = (name) => fs.readFileSync(new URL(`public/models/${name}.onnx`, ROOT));
const mj = await loadMujoco();
const rows = process.env.ROWS ? process.env.ROWS.split(',').map(Number) : null;
const allLanes = lanes().filter((lane) => !rows || rows.includes(lane.row));
const networks = {
  encoder: await DepthEncoder.create(ort, model('defm')),
  policy: await RecurrentPolicy.create(ort, model('policy'), HIGH_LEVEL_HIDDEN),
  crawl: await RecurrentPolicy.create(ort, model('crawl'), LOW_LEVEL_HIDDEN),
  jump: await RecurrentPolicy.create(ort, model('jump'), LOW_LEVEL_HIDDEN),
};
const xml = env2SceneXml({ tail: process.env.TAIL === '1' });
const episodes = [];
for (let index = shard; index < allLanes.length; index += shards) {
  const lane = allLanes[index];
  for (let episode = 0; episode < perLane; episode++) {
    const random = seededRandom(seed * 1000003 + index * 1009 + episode);
    // Lane x is measured from the lane origin; the obstacle positions of the reference are measured from the lane
    // start, 1 m before it.
    const firstStart = lane.obstacles[0].x_start - 1;
    const lastEnd = lane.obstacles.at(-1).x_end - 1;
    const spawnX = SPAWN_MIN_X + (firstStart - MIN_RUN_UP - SPAWN_MIN_X) * random();
    const spawnY = SPAWN_SPREAD_Y * (2 * random() - 1);
    const goalX = lastEnd + 2 + 2 * random();
    const robot = new Robot(mj, xml, random);
    robot.reset({ x: lane.origin[0] + spawnX, y: lane.origin[1] + spawnY });
    const run = new Episode(robot, new DepthCamera(robot), networks, { origin: lane.origin, spawnX, goalX });
    const start = performance.now();
    let result;
    do result = await run.step(); while (!result.done);
    const x = run.hierarchy.position();
    episodes.push({
      row: lane.row, column: lane.column, outcome: result.outcome, steps: run.controlSteps,
      kinds: lane.obstacles.map((o) => o.kind), obstacles: lane.obstacles.length,
      passed: lane.obstacles.filter((o) => x > o.x_end - 1).length, jumps: run.hierarchy.jumps,
      x, goalX, spawnX, seconds: (performance.now() - start) / 1000,
    });
    robot.data.delete();
    robot.model.delete();
    console.log(`lane row ${lane.row} column ${lane.column} episode ${episode}: ${result.outcome} after `
      + `${(run.controlSteps * 0.01).toFixed(1)} s, ${episodes.at(-1).passed}/${lane.obstacles.length} obstacles, `
      + `${run.hierarchy.jumps} jumps`);
  }
}
if (output) fs.writeFileSync(output, JSON.stringify(episodes));
report(episodes);
