// The simulation worker: MuJoCo WASM, the four ONNX networks and the parkour hierarchy, off the main thread.
//
// Messages in:   {type: 'init'}                      load everything
//                {type: 'start', course, run, hold}  build the lane of `course` and run an episode on it
//                {type: 'pause', paused}, {type: 'stop'}
// Messages out:  {type: 'progress', loaded, total}   bytes of the downloads
//                {type: 'ready', threads}
//                {type: 'frame', time, bodies, status}  after every control step: simulated time, body poses,
//                                                    status; with `depth` when the policy looked at a new image
//                {type: 'done', outcome, ...}        the episode ended
//                {type: 'error', message}
import loadMujoco from '@mujoco/mujoco';
import * as ort from 'onnxruntime-web/wasm';
import { CONTROL_DT, H_REF, Robot, sceneXml } from './sim/robot.js';
import { DepthCamera } from './sim/depth.js';
import { DepthEncoder, HIGH_LEVEL_HIDDEN, LOW_LEVEL_HIDDEN, RecurrentPolicy } from './sim/policies.js';
import { Episode } from './sim/episode.js';
import { boxesXml, laneBoxes } from './sim/terrain.js';
import { BODIES } from './protocol.js';

const BASE = import.meta.env.BASE_URL;
const DOWNLOADS = ['models/defm.onnx', 'models/policy.onnx', 'models/crawl.onnx', 'models/jump.onnx', 'robot/beni.xml'];

let mj = null;
let robotXml = '';
let networks = null;
let robot = null;
let episode = null;
let course = null;
let bodyIds = [];
let runId = 0; // Incremented to stop the running loop.
let paused = false;

/** Download `paths` under the page's base URL, reporting the bytes received so far. */
async function download(paths) {
  const responses = await Promise.all(paths.map((path) => fetch(BASE + path)));
  const total = responses.reduce((sum, r) => sum + Number(r.headers.get('content-length') ?? 0), 0);
  let loaded = 0;
  return Promise.all(responses.map(async (response, i) => {
    if (!response.ok) throw new Error(`${paths[i]}: HTTP ${response.status}`);
    const chunks = [];
    const reader = response.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      postMessage({ type: 'progress', loaded, total });
    }
    const bytes = new Uint8Array(chunks.reduce((sum, c) => sum + c.length, 0));
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.length;
    }
    return bytes;
  }));
}

async function init() {
  // One thread. With more, onnxruntime-web starts helper workers, and creating a session then never returns when
  // the library itself runs inside this worker (seen with 4 threads on a cross-origin isolated page, Vite dev
  // server, 2026-11-12). One thread runs the depth encoder in about 23 ms per image on an Apple M5.
  const threads = 1;
  ort.env.wasm.numThreads = threads;
  const [files, module] = await Promise.all([download(DOWNLOADS), loadMujoco()]);
  mj = module;
  const [defm, policy, crawl, jump, xml] = files;
  robotXml = new TextDecoder().decode(xml);
  networks = {
    encoder: await DepthEncoder.create(ort, defm),
    policy: await RecurrentPolicy.create(ort, policy, HIGH_LEVEL_HIDDEN),
    crawl: await RecurrentPolicy.create(ort, crawl, LOW_LEVEL_HIDDEN),
    jump: await RecurrentPolicy.create(ort, jump, LOW_LEVEL_HIDDEN),
  };
  postMessage({ type: 'ready', threads });
}

/** Yield to the event loop without the timer clamp, so that messages are handled between control steps. */
const channel = new MessageChannel();
const tick = () => new Promise((resolve) => {
  channel.port1.onmessage = resolve;
  channel.port2.postMessage(0);
});
const sleep = (ms) => new Promise((resolve) => { setTimeout(resolve, ms); });

function frame(withDepth, realTime) {
  const { data } = robot;
  const { hierarchy } = episode;
  const bodies = new Float32Array(7 * BODIES.length);
  bodyIds.forEach((id, i) => {
    bodies.set(data.xpos.subarray(3 * id, 3 * id + 3), 7 * i);
    bodies.set(data.xquat.subarray(4 * id, 4 * id + 4), 7 * i + 3);
  });
  const { jump } = hierarchy;
  const x = hierarchy.position();
  const time = episode.controlSteps * CONTROL_DT; // Simulated seconds since the start of the episode.
  const message = {
    type: 'frame', run: course.run, time, bodies,
    status: {
      time,
      x,
      speedCommand: hierarchy.twist[0],
      heightCommand: hierarchy.heightCommand,
      // What drives the robot: the jump policy from an accepted request until it has landed, else the crawl policy.
      mode: hierarchy.inJump ? (jump.preparing ? 'prepare' : 'jump')
        : hierarchy.heightCommand < H_REF - 0.008 ? 'crouch' : 'drive',
      trigger: hierarchy.pending > 0,
      // What the high-level policy asked for in its latest step: speed (m/s), body height (m), jump trigger.
      policy: { speed: hierarchy.vx, height: hierarchy.height, trigger: episode.highLevelAction[2] > 1 },
      jumps: hierarchy.jumps,
      passed: course.obstacles.filter((o) => x > o.xEnd - 1).length,
      realTime,
    },
  };
  const transfer = [bodies.buffer];
  if (withDepth) {
    message.depth = Float32Array.from(episode.depth);
    transfer.push(message.depth.buffer);
  }
  postMessage(message, transfer);
}

async function run(id) {
  // Wall-clock time at which the simulation was at step 0, were it running in real time since.
  let origin = performance.now();
  let windowStart = origin;
  let windowSteps = 0;
  let realTime = 1;
  while (id === runId) {
    if (paused) {
      await sleep(50);
      origin = performance.now() - episode.controlSteps * CONTROL_DT * 1000;
      windowStart = performance.now();
      windowSteps = 0;
      continue;
    }
    const newImage = episode.controlSteps % 5 === 0;
    const result = await episode.step();
    if (id !== runId) return;
    windowSteps += 1;
    const now = performance.now();
    if (now - windowStart > 500) {
      realTime = (windowSteps * CONTROL_DT * 1000) / (now - windowStart);
      windowStart = now;
      windowSteps = 0;
    }
    frame(newImage, realTime);
    if (result.done) {
      const x = episode.hierarchy.position();
      postMessage({
        type: 'done', run: course.run, outcome: result.outcome, time: episode.controlSteps * CONTROL_DT, x,
        passed: course.obstacles.filter((o) => x > o.xEnd - 1).length, jumps: episode.hierarchy.jumps,
      });
      return;
    }
    const ahead = origin + episode.controlSteps * CONTROL_DT * 1000 - now;
    if (ahead > 1) {
      await sleep(ahead);
    } else {
      if (ahead < -200) origin = now - episode.controlSteps * CONTROL_DT * 1000; // Slower than real time.
      await tick();
    }
  }
}

/**
 * Build the lane of `next` ({obstacles, length, goalX, spawn: {x, y}}) and start an episode on it. `number` is the
 * page's number of this run, sent back with every frame. With `hold` the robot waits on the grid until the page
 * sends {type: 'pause', paused: false}.
 */
function start(next, number, hold) {
  runId += 1;
  paused = Boolean(hold);
  course = { ...next, run: number };
  if (robot) {
    robot.data.delete();
    robot.model.delete();
  }
  robot = new Robot(mj, sceneXml(robotXml, boxesXml(laneBoxes(course.obstacles, course.length))));
  robot.reset({ x: course.spawn.x, y: course.spawn.y });
  bodyIds = BODIES.map((name) => mj.mj_name2id(robot.model, 1, `robot/${name}`));
  episode = new Episode(robot, new DepthCamera(robot), networks, {
    origin: [0, 0, 0], spawnX: course.spawn.x, goalX: course.goalX,
  });
  frame(false, 1); // The robot at the start, before the first step.
  run(runId).catch((error) => postMessage({ type: 'error', message: String(error?.stack ?? error) }));
}

self.onmessage = ({ data }) => {
  if (data.type === 'init') init().catch((error) => postMessage({ type: 'error', message: String(error?.stack ?? error) }));
  else if (data.type === 'start') start(data.course, data.run, data.hold);
  else if (data.type === 'pause') paused = data.paused;
  else if (data.type === 'stop') runId += 1;
};
