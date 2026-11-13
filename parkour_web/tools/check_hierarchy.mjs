// Check the JavaScript hierarchy (src/sim/hierarchy.js, jump.js) against the recorded Python episodes.
//
// The high-level actions, the applied low-level actions, the simulation states and the contact forces come from the
// recording, so this tests the logic by itself: commands, trigger latch, jump state machine, the observation each
// low-level policy gets, which policy drives, the high-level observation, and how the episode ends. Every control
// step is compared.
//   node tools/check_hierarchy.mjs [env id ...]
import { DECIMATION, Robot } from '../src/sim/robot.js';
import { Hierarchy, SUB_STEPS } from '../src/sim/hierarchy.js';
import { env2SceneXml, loadMujoco, loadRollout, maxAbsDiff, rolloutIds } from './lib.mjs';

const ids = process.argv.length > 2 ? process.argv.slice(2).map(Number) : rolloutIds();
const mj = await loadMujoco();
const robot = new Robot(mj, env2SceneXml());
const norm3 = (v, i) => Math.hypot(v[3 * i], v[3 * i + 1], v[3 * i + 2]);
let failures = 0;

for (const id of ids) {
  const rollout = loadRollout(id);
  const { info, steps } = rollout;
  robot.clearMemory();
  robot.setState(rollout.row('qpos0', 0), rollout.row('qvel0', 0));
  const hierarchy = new Hierarchy(robot, { origin: info.origin, spawnX: info.spawn_x, goalX: info.goal_x });
  hierarchy.timeBudget = info.time_budget;

  // Largest difference per quantity, and the control step where it occurred.
  const worst = {};
  const note = (name, gap, t) => {
    if (!(name in worst) || gap > worst[name].gap) worst[name] = { gap, t };
  };
  let result = null;
  for (let t = 0; t < steps; t++) {
    if (t % SUB_STEPS === 0) {
      const h = t / SUB_STEPS;
      note('high-level observation', maxAbsDiff(hierarchy.observation(), rollout.row('hl_obs', h)), t);
      hierarchy.decodeAction(rollout.row('hl_action', h));
    }
    const { crawlObservation, jumpObservation } = hierarchy.beforePhysics();
    note('crawl observation', maxAbsDiff(crawlObservation, rollout.row('crawl_obs', t)), t);
    note('jump observation', maxAbsDiff(jumpObservation, rollout.row('jump_obs', t)), t);
    hierarchy.selectAction(null, null);

    // Physics from the recording: every physics step starts from the recorded state with the recorded torques, and
    // the contact sensors are the recorded ones.
    robot.lastAction.set(rollout.row('action', t));
    robot.bodyForcePeak.fill(0);
    for (let k = 0; k < DECIMATION; k++) {
      const p = t * DECIMATION + k;
      robot.data.qpos.set(p ? rollout.row('qpos', p - 1) : rollout.row('qpos0', 0));
      robot.data.qvel.set(p ? rollout.row('qvel', p - 1) : rollout.row('qvel0', 0));
      robot.data.ctrl.set(rollout.row('ctrl', p));
      mj.mj_step(robot.model, robot.data);
      const body = rollout.row('body_force', p);
      for (let b = 0; b < 6; b++) {
        robot.bodyForce[b] = norm3(body, b);
        robot.bodyForcePeak[b] = Math.max(robot.bodyForcePeak[b], robot.bodyForce[b]);
      }
      const wheel = rollout.row('wheel_force', p);
      robot.wheelForce.forEach((force, w) => force.set(wheel.subarray(3 * w, 3 * w + 3)));
    }
    const last = t * DECIMATION + DECIMATION - 1;
    robot.data.qpos.set(rollout.row('qpos', last));
    robot.data.qvel.set(rollout.row('qvel', last));
    result = hierarchy.afterPhysics();

    const s = rollout.state(t);
    const { jump } = hierarchy;
    const m = result.measurement;
    const pose = robot.basePose();
    const ours = {
      ready_steps: jump.readySteps, yaw_ready_steps: jump.yawReadySteps, cool_down: jump.coolDown,
      jump_flag: jump.flag, jump_count: jump.count, preparing: jump.preparing, tracking: jump.tracking,
      gate: jump.gate, lifted: jump.lifted, land_s: jump.landS, age: jump.age, heading_active: jump.headingActive,
      command_hold: jump.commandHold, failed: jump.failed, ready_hold: jump.readyHold, pending: hierarchy.pending,
      trigger_high: hierarchy.triggerHigh, in_jump: hierarchy.inJump, jump_ready: m.jumpReady, air: m.air,
      contact_left: m.contacts[0], contact_right: m.contacts[1], body: m.body, load: m.load, height: m.height,
      tilt: m.tilt, error: m.error, margin: m.margin, twist_vx: hierarchy.twist[0], twist_vy: hierarchy.twist[1],
      twist_wz: hierarchy.twist[2], height_command: hierarchy.heightCommand,
      goal_distance: info.goal_x - hierarchy.position(), position_x: hierarchy.position(),
      lateral_offset: pose.position[1] - info.origin[1], heading: Robot.heading(pose.quat),
    };
    // On the last control step Python reset the env before the state was read.
    if (t < steps - 1) for (const key of info.state_keys) note(key, Math.abs(Number(ours[key]) - s[key]), t);
  }

  const outcome = result.outcome === info.outcome && result.done;
  // The tilt is an arc cosine near 1; Python computes it in 32-bit floats, which resolve it to about 0.02 degrees.
  const bad = Object.entries(worst).filter(([name, { gap }]) => gap > (name === 'tilt' ? 0.05 : 2e-4));
  failures += bad.length + (outcome ? 0 : 1);
  console.log(`env ${id} (row ${info.row}, column ${info.column}): ${steps} control steps, ${hierarchy.jumps} jumps; `
    + `ended with ${result.outcome || 'nothing'} (Python: ${info.outcome}); `
    + `${Object.keys(worst).length} quantities compared, ${bad.length} differ`);
  for (const [name, { gap, t }] of bad) console.log(`    ${name}: largest difference ${gap.toExponential(2)} at control step ${t}`);
  const largest = Object.entries(worst).filter(([name]) => name.includes('observation'))
    .map(([name, { gap }]) => `${name} ${gap.toExponential(1)}`).join(', ');
  console.log(`    largest differences: ${largest}`);
}
if (failures) throw new Error(`${failures} differences`);
console.log('hierarchy logic matches the recorded episodes');
