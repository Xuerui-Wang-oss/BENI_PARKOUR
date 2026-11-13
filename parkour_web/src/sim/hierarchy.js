// The parkour hierarchy around one robot: a port of python/parkour_rl/hierarchy.py (ParkourVecEnvWrapper) and of the
// goal, the terminations and the high-level observation of the env (python/parkour_rl/env.py, goal.py, reward.py).
//
// The high-level policy acts every SUB_STEPS control steps (20 Hz) with three numbers: forward speed, base height and
// a jump trigger. On every control step (100 Hz) the hierarchy writes the commands, requests a pending jump, builds
// the observation of the two frozen low-level policies, and applies the action of the jump policy while a jump is in
// progress and of the crawl policy otherwise.
//
// One control step, in the order of the Python env:
//   const { crawlObservation, jumpObservation } = hierarchy.beforePhysics();
//   const action = hierarchy.selectAction(crawlAction, jumpAction);   // the outputs of the two policies
//   robot.step(action);
//   const { done, outcome } = hierarchy.afterPhysics();

import { CONTACT_FORCE, CONTROL_DT, H_LOW, H_REF, Robot } from './robot.js';
import { JumpCommand } from './jump.js';

export const SUB_STEPS = 5;
export const HL_OBSERVATION_DIM = 24;

const f = Math.fround;
const clamp = (x, low, high) => Math.min(Math.max(x, low), high);
const radians = (degrees) => (degrees * Math.PI) / 180;

// The (vx, height) action: one unit is 0.4 m/s and 0.02 m, clamped so that it maps to 0-1.5 m/s and H_LOW-H_REF.
const VX_CENTER = 0.8;
const VX_SCALE = 0.4;
const HEIGHT_SCALE = 0.02;
const ACTION_LOW = [f((0 - VX_CENTER) / VX_SCALE), f((H_LOW - H_REF) / HEIGHT_SCALE)];
const ACTION_HIGH = [f((1.5 - VX_CENTER) / VX_SCALE), 0];
const TRIGGER_THRESHOLD = 1;
const LATCH_STEPS = 60; // Control steps a trigger stays pending.
const PRESETTLE_STEPS = 20; // Control steps at the start of a latch without a request.
// Heading hold and lane keeping: yaw rate back to the centre line of the lane and along it.
const HEADING_GAIN = 2;
const LATERAL_GAIN = 0.5;
const LATERAL_HEADING_LIMIT = 0.2;
const YAW_LIMIT = 1;
const YAW_DEADBAND = 0.05;
const SETTLE_S = f(0.4); // The jump policy drives until this long after touchdown.
const JUMP_HEIGHT_RANGE = [0.1, 0.12]; // Height commands the jump policy was trained with.

// Success: at the goal (from GOAL_TOLERANCE before it to GOAL_OVERSHOOT past it), slow, upright and without body
// contact for GOAL_HOLD_STEPS control steps in a row.
const GOAL_TOLERANCE = 0.3;
export const GOAL_OVERSHOOT = 1;
const GOAL_MAX_SPEED = 0.2;
const GOAL_MAX_TILT = radians(20);
const GOAL_HOLD_STEPS = 50;
const FELL_TILT = radians(60);
const FELL_STEPS = 30;
const PIT_HEIGHT = -0.05;
const LANE_BEHIND = 0.5;
const STUCK_DISTANCE = 0.1;
const STUCK_WINDOW_S = 3;
const CONTACT_STEPS = 50;
export const EPISODE_LENGTH_S = 64;
const TIME_BASE_S = 4;
const MIN_AVERAGE_SPEED = 0.6;
const GOAL_DISTANCE_CLIP = [-GOAL_OVERSHOOT, 2]; // Range of the goal distance that the policy observes (m).

/** Time budget of an episode (s) from the spawn and the goal in lane x. */
export const timeBudget = (spawnX, goalX) => TIME_BASE_S + (goalX - spawnX) / MIN_AVERAGE_SPEED;

export class Hierarchy {
  /**
   * @param robot a `Robot` that was just reset to the spawn pose.
   * @param origin lane origin [x, y, z] in the world: lane x and the lateral offset are measured from it.
   * @param spawnX, goalX lane x of the spawn and of the goal (m).
   */
  constructor(robot, { origin = [0, 0, 0], spawnX = 0, goalX }) {
    this.robot = robot;
    this.origin = origin;
    this.spawnX = spawnX;
    this.goalX = goalX;
    this.timeBudget = timeBudget(spawnX, goalX);
    this.jump = new JumpCommand();
    this.twist = [0, 0, 0]; // vx, vy, yaw rate command.
    this.heightCommand = H_REF;
    this.action = [0, 0]; // Clamped (vx, height) action of the current high-level step.
    this.vx = 0;
    this.height = H_REF;
    this.triggerHigh = false;
    this.pending = 0; // Control steps left in the trigger latch.
    this.triggerFailed = false;
    this.inJump = false; // The jump policy drives in the current control step.
    this.episodeLength = 0; // Control steps.
    this.goalSteps = 0;
    this.fellSteps = 0;
    this.contactSteps = 0;
    this.markX = spawnX;
    this.markStep = 0;
    this.jumps = 0; // Accepted requests.
  }

  /** Lane x of the robot (m). */
  position() {
    return this.robot.basePose().position[0] - this.origin[0];
  }

  /** Start of a high-level step: `action` is [vx, height, trigger] from the high-level policy. */
  decodeAction(action) {
    this.action = [clamp(action[0], ACTION_LOW[0], ACTION_HIGH[0]), clamp(action[1], ACTION_LOW[1], ACTION_HIGH[1])];
    const trigger = action[2] > TRIGGER_THRESHOLD;
    // Ignored while a jump is being tracked and during the cool-down.
    const blocked = this.jump.tracking || this.jump.coolDown > 0;
    if (trigger && !blocked && !this.triggerHigh && this.pending === 0) this.pending = LATCH_STEPS;
    this.triggerHigh = trigger;
    this.vx = f(VX_CENTER + VX_SCALE * this.action[0]);
    this.height = f(H_REF + HEIGHT_SCALE * this.action[1]);
  }

  /** The jump policy drives: from an accepted request until SETTLE_S after touchdown, unless the jump failed. */
  jumpInProgress() {
    const jump = this.jump;
    const notSettled = jump.tracking && jump.gate && (jump.landS < 0 || jump.age < f(jump.landS + SETTLE_S));
    return jump.preparing || jump.flag || notSettled;
  }

  /** Yaw-rate command (rad/s) that turns the robot back to the centre line of its lane and along it. */
  headingHold() {
    const { position, quat } = this.robot.basePose();
    const offset = position[1] - this.origin[1];
    const heading = clamp(-LATERAL_GAIN * offset, -LATERAL_HEADING_LIMIT, LATERAL_HEADING_LIMIT);
    const yawRate = clamp(HEADING_GAIN * (heading - Robot.heading(quat)), -YAW_LIMIT, YAW_LIMIT);
    return Math.abs(yawRate) < YAW_DEADBAND ? 0 : yawRate;
  }

  /** Write the commands of this control step, request a pending jump and build the low-level observations. */
  beforePhysics() {
    const jump = this.jump;
    // Commands. During a jump the twist stays as the jump command pinned it, and the yaw rate is exactly 0 before
    // and during a jump; the hold steers again as soon as the crawl policy drives after touchdown.
    const pending = this.pending > 0;
    const locked = jump.preparing || jump.commandHold;
    if (!locked) this.twist[0] = this.vx;
    this.twist[1] = 0;
    const landedOnCrawl = jump.landS >= 0 && !this.jumpInProgress();
    const steering = !pending && (!locked || landedOnCrawl);
    this.twist[2] = steering ? this.headingHold() : 0;
    this.heightCommand = pending ? H_REF : this.height;

    // The latch: no request for its first PRESETTLE_STEPS control steps, then one per control step until accepted.
    this.triggerFailed = false;
    if (this.pending > LATCH_STEPS - PRESETTLE_STEPS) {
      this.pending -= 1;
    } else if (this.pending > 0) {
      const refused = jump.request(this.twist) !== 0;
      this.pending = refused ? this.pending - 1 : 0;
      this.triggerFailed = refused && this.pending === 0;
      if (!refused) this.jumps += 1;
    }

    const observation = this.robot.observation(this.twist, this.heightCommand, jump.command);
    const crawlObservation = Float32Array.from(observation);
    crawlObservation.fill(0, 36, 40); // The crawl policy was trained without a jump command.
    const jumpObservation = Float32Array.from(observation);
    jumpObservation[30] = clamp(jumpObservation[30], JUMP_HEIGHT_RANGE[0], JUMP_HEIGHT_RANGE[1]);
    return { crawlObservation, jumpObservation };
  }

  /** The action to apply: of the jump policy while a jump is in progress, of the crawl policy otherwise. */
  selectAction(crawlAction, jumpAction) {
    this.inJump = this.jumpInProgress();
    return this.inJump ? jumpAction : crawlAction;
  }

  /**
   * After the physics of a control step: terminations, jump measurement, then the commands advance.
   * @returns {done, outcome, measurement}: `outcome` lists the termination terms that fired, joined by "+".
   */
  afterPhysics() {
    const { robot, jump } = this;
    this.episodeLength += 1;
    const m = robot.measure(this.origin);

    const x = m.position[0] - this.origin[0];
    const distance = this.goalX - x;
    const speed = Math.hypot(m.velocity.linear[0], m.velocity.linear[1]);
    const tilt = Math.atan2(Math.hypot(m.gravity[0], m.gravity[1]), -m.gravity[2]);
    const contact = robot.bodyForcePeak.some((force) => force > CONTACT_FORCE);
    const stopped = distance < GOAL_TOLERANCE && distance > -GOAL_OVERSHOOT && speed < GOAL_MAX_SPEED
      && tilt < GOAL_MAX_TILT && !contact;
    this.goalSteps = stopped ? this.goalSteps + 1 : 0;
    this.fellSteps = Math.abs(Math.acos(clamp(-m.gravity[2], -1, 1))) > FELL_TILT ? this.fellSteps + 1 : 0;
    this.contactSteps = contact ? this.contactSteps + 1 : 0;
    if (x > this.markX + STUCK_DISTANCE) {
      this.markX = x;
      this.markStep = this.episodeLength;
    }
    const elapsed = this.episodeLength * CONTROL_DT;
    const terms = {
      time_out: elapsed >= EPISODE_LENGTH_S,
      out_of_time: elapsed >= this.timeBudget,
      goal_reached: this.goalSteps >= GOAL_HOLD_STEPS,
      fell: this.fellSteps > FELL_STEPS,
      in_pit: m.height < PIT_HEIGHT,
      out_of_lane: x < this.spawnX - LANE_BEHIND || x > this.goalX + GOAL_OVERSHOOT,
      stuck: (this.episodeLength - this.markStep) * CONTROL_DT > STUCK_WINDOW_S,
      body_contact: this.contactSteps > CONTACT_STEPS,
    };
    const fired = Object.keys(terms).filter((name) => terms[name]);
    const done = fired.length > 0;

    jump.observe(m, this.twist[2], done);
    robot.forward();
    // The twist command term: held at the request's value through a jump, and no yaw rate after any jump until the
    // next control step writes one.
    if (jump.commandHold) this.twist = jump.requestTwist.slice();
    if (jump.headingActive) this.twist[2] = 0;
    jump.update();
    return { done, outcome: fired.join('+'), measurement: m };
  }

  /** The 24 state numbers of the high-level policy; call after the last control step of a high-level step. */
  observation() {
    const { robot, jump } = this;
    const low = robot.observation(this.twist, this.heightCommand, jump.command);
    const o = new Float32Array(HL_OBSERVATION_DIM);
    o.set(low.subarray(0, 6), 0); // Gyro, gravity direction.
    o.set(low.subarray(9, 21), 6); // Joint positions (5) and velocities (7).
    o.set(this.action, 18);
    // Jump status: the jump policy is driving, a trigger is pending, a trigger would now start a jump on time.
    const ready = jump.readySteps > 0 && jump.coolDown === 0 && !jump.tracking
      && Math.abs(robot.baseVelocity().angular[2]) <= 0.3;
    o.set([this.inJump ? 1 : 0, this.pending > 0 ? 1 : 0, ready ? 1 : 0], 20);
    o[23] = clamp(this.goalX - this.position(), GOAL_DISTANCE_CLIP[0], GOAL_DISTANCE_CLIP[1]);
    return o;
  }
}
