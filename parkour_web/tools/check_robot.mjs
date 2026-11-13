// Check the JavaScript robot (src/sim/robot.js) against the recorded Python episodes, with the simulation state
// taken from the recording at every step, so each part is tested by itself:
//   observation   the low-level observation computed from a recorded state
//   motors        the motor torques computed from a recorded state, action and command delay
//   physics       one physics step of MuJoCo C from a recorded state with the recorded torques, against the next
//                 recorded state of mujoco_warp; and the drift over longer stretches without correction
//   contacts      the contact sensors after that step
//   node tools/check_robot.mjs [env id ...]
import { DECIMATION, Robot } from '../src/sim/robot.js';
import { env2SceneXml, loadMujoco, loadRollout, maxAbsDiff, rolloutIds } from './lib.mjs';

const ids = process.argv.length > 2 ? process.argv.slice(2).map(Number) : rolloutIds();
const mj = await loadMujoco();
let start = performance.now();
const robot = new Robot(mj, env2SceneXml());
console.log(`world of 80 lanes compiled in ${(performance.now() - start).toFixed(0)} ms, ${robot.model.ngeom} geoms`);

const quantile = (values, q) => values.slice().sort((a, b) => a - b)[Math.min(values.length - 1, Math.floor(q * values.length))];
const summary = (values) => `median ${quantile(values, 0.5).toExponential(1)}, 99% ${quantile(values, 0.99).toExponential(1)}, `
  + `max ${Math.max(...values).toExponential(1)}`;

for (const id of ids) {
  const rollout = loadRollout(id);
  const steps = rollout.steps;
  const physicsSteps = steps * DECIMATION;
  console.log(`\nenv ${id} (row ${rollout.info.row}, column ${rollout.info.column}): ${steps} control steps, `
    + `outcome ${rollout.info.outcome}`);
  // State before physics step p: the recorded state after step p - 1, or the start of the episode.
  const before = (p) => (p === 0 ? [rollout.row('qpos0', 0), rollout.row('qvel0', 0)]
    : [rollout.row('qpos', p - 1), rollout.row('qvel', p - 1)]);

  // Observation: dynamic entries (gyro, gravity, joint positions and velocities, head gyro) from the recorded state.
  const observationGaps = [];
  for (let t = 0; t < steps; t++) {
    robot.setState(...before(t * DECIMATION));
    const reference = rollout.row('crawl_obs', t);
    const jumpReference = rollout.row('jump_obs', t);
    if (t) robot.lastAction.set(rollout.row('action', t - 1));
    else robot.lastAction.fill(0);
    const observation = robot.observation(reference.subarray(6, 9), reference[30], jumpReference.subarray(36, 40));
    // The crawl policy gets zeros as jump command; everything else must be the recorded observation.
    const expected = Float32Array.from(reference);
    expected.set(jumpReference.subarray(36, 40), 36);
    observationGaps.push(maxAbsDiff(observation, expected));
  }
  console.log(`  observation (40 numbers) from recorded states: ${summary(observationGaps)}`);

  // Motors: torques from the recorded state, action and delay.
  robot.clearMemory();
  const torqueGaps = [];
  for (let p = 0; p < physicsSteps; p++) {
    const [qpos, qvel] = before(p);
    robot.data.qpos.set(qpos);
    robot.data.qvel.set(qvel);
    robot.lastAction.set(rollout.row('action', Math.floor(p / DECIMATION)));
    robot.applyMotors(rollout.row('lag', p));
    torqueGaps.push(maxAbsDiff(robot.data.ctrl, rollout.row('ctrl', p)));
  }
  console.log(`  motor torques (N m): ${summary(torqueGaps)}`);

  // Physics: one step of MuJoCo C from each recorded state, and the contact sensors after it.
  const positionGaps = [];
  const jointGaps = [];
  const velocityGaps = [];
  const wheelForceGaps = [];
  let bodyContactAgree = 0;
  let wheelContactAgree = 0;
  for (let p = 0; p < physicsSteps; p++) {
    const [qpos, qvel] = before(p);
    robot.data.qpos.set(qpos);
    robot.data.qvel.set(qvel);
    robot.data.ctrl.set(rollout.row('ctrl', p));
    mj.mj_step(robot.model, robot.data);
    robot.readContacts();
    const next = rollout.row('qpos', p);
    positionGaps.push(maxAbsDiff(robot.data.qpos.subarray(0, 3), next.subarray(0, 3)));
    jointGaps.push(maxAbsDiff(robot.data.qpos.subarray(7), next.subarray(7)));
    velocityGaps.push(maxAbsDiff(robot.data.qvel, rollout.row('qvel', p)));
    const wheel = rollout.row('wheel_force', p);
    const body = rollout.row('body_force', p);
    for (let w = 0; w < 2; w++) {
      const reference = wheel.subarray(3 * w, 3 * w + 3);
      wheelForceGaps.push(maxAbsDiff(robot.wheelForce[w], reference));
      wheelContactAgree += (Math.hypot(...robot.wheelForce[w]) > 0.1) === (Math.hypot(...reference) > 0.1);
    }
    for (let b = 0; b < 6; b++) {
      bodyContactAgree += (robot.bodyForce[b] > 0.1) === (Math.hypot(...body.subarray(3 * b, 3 * b + 3)) > 0.1);
    }
  }
  console.log(`  one physics step, base position (m): ${summary(positionGaps)}`);
  console.log(`  one physics step, joint angles (rad): ${summary(jointGaps)}`);
  console.log(`  one physics step, velocities: ${summary(velocityGaps)}`);
  console.log(`  wheel contact force (N): ${summary(wheelForceGaps)}; contact flag equal on `
    + `${(100 * wheelContactAgree / (2 * physicsSteps)).toFixed(2)}% of wheel samples, body contact flag on `
    + `${(100 * bodyContactAgree / (6 * physicsSteps)).toFixed(2)}%`);

  // Drift: MuJoCo C run for a stretch with the recorded torques, without correction.
  for (const stretch of [0.1, 0.5]) {
    const length = Math.round(stretch / 0.002);
    const drift = [];
    for (let first = 0; first + length <= physicsSteps; first += length) {
      robot.setState(...before(first));
      for (let p = first; p < first + length; p++) {
        robot.data.ctrl.set(rollout.row('ctrl', p));
        mj.mj_step(robot.model, robot.data);
      }
      drift.push(maxAbsDiff(robot.data.qpos.subarray(0, 3), rollout.row('qpos', first + length - 1).subarray(0, 3)));
    }
    console.log(`  drift of the base after ${stretch} s of recorded torques (m): ${summary(drift)}`);
  }
}
