// The BENI robot in MuJoCo: model, motors, sensors, and the low-level observation.
//
// It reproduces what mjlab does around the physics for one robot (python/beni_rl_crawl/beni_robot.py, actuator.py and
// python/beni_rl/beni_velocity_flat_env_cfg.py): the motors are torque actuators driven by a PD law computed here on
// every physics step from delayed targets, and the contact sensors are read after every physics step.
//
// A control step is DECIMATION physics steps of 2 ms. After the physics steps of a control step the kinematic
// quantities of MuJoCo (body poses and velocities) still belong to the state before the last physics step; mjlab
// evaluates terminations and the jump measurement on them and only then calls forward(). `measure()` is meant to be
// called at that point, `observation()` after `forward()`.

export const DECIMATION = 5;
export const CONTROL_DT = 0.01;
export const ACTION_DIM = 7;
export const OBSERVATION_DIM = 40;

const PREFIX = 'robot/';
// Order of the 7 actions.
export const ACTION_JOINTS = ['L1', 'R1', 'L2', 'R2', 'LW', 'RW', 'head_yaw_joint'];
const POSITION_OBS_JOINTS = ['L1', 'R1', 'head_yaw_joint', 'L2', 'R2'];
const VELOCITY_OBS_JOINTS = ['L1', 'R1', 'head_yaw_joint', 'L2', 'R2', 'LW', 'RW'];
export const DEFAULT_JOINT_POS = { L1: 0.3, R1: 0.3, L2: 0.5, R2: 0.5, head_yaw_joint: 0, head_roll_joint: 0, LW: 0, RW: 0 };
const ACTION_SCALE = { L1: 0.25, R1: 0.25, L2: 0.25, R2: 0.25, LW: 20, RW: 20, head_yaw_joint: 0.5 };
export const SPAWN_HEIGHT = 0.14;

// Standing posture and height that the jump readiness is measured against (python/beni_rl/recovery.py).
export const Q_REF = { L1: 0.2848744094, R1: 0.289914161, L2: 0.4957503676, R2: 0.5020063519 };
export const H_REF = 0.1101198718;
export const H_LOW = 0.067;

// Motor groups: kp, kd, torque limit of the PD law, and whether the command is delayed. A knee adds its passive
// spring to the clipped PD torque and clips the sum at SIM_LIMIT.
const GROUPS = [
  { joints: ['L1', 'R1'], kp: 0.5, kd: 0.02, limit: 1.375, delayed: true },
  { joints: ['L2', 'R2'], kp: 0.5, kd: 0.02, limit: 3.6, simLimit: 4.6, spring: true, delayed: true },
  { joints: ['LW', 'RW'], kp: 0, kd: 0.004, limit: 0.6, velocity: true, delayed: true },
  { joints: ['head_yaw_joint'], kp: 3, kd: 0.05, limit: 0.1, delayed: true },
  { joints: ['head_roll_joint'], kp: 2, kd: 0, limit: 1, delayed: false },
];
// Command delay of a delayed group in physics steps: drawn anew on every physics step, uniform in MIN_LAG..MAX_LAG.
export const MIN_LAG = 2;
export const MAX_LAG = 5;

const BODY_CONTACT_BODIES = ['base_link', 'head', 'L1_Link', 'L2_Link', 'R1_Link', 'R2_Link'];
const WHEEL_BODIES = ['LW_Link', 'RW_Link'];
export const CONTACT_FORCE = 0.1; // N: a larger net force counts as contact.
const WHEEL_RADIUS = 0.038;
const WHEEL_HALF_WIDTH = 0.0105;

const OBJ = { body: 1, joint: 3, geom: 5, camera: 7, actuator: 19, sensor: 20 };
const clamp = (x, low, high) => Math.min(Math.max(x, low), high);
const kneeSpring = (q) => 0.22 * Math.sin(q - 1) + 0.15 * Math.sin(2 * (q - 1));

/** Insert the terrain geoms into the robot MJCF (public/robot/beni.xml). */
export function sceneXml(robotXml, terrainXml) {
  if (!robotXml.includes('<!--TERRAIN-->')) throw new Error('the robot XML has no terrain marker');
  return robotXml.replace('<!--TERRAIN-->', terrainXml);
}

export class Robot {
  /**
   * @param mj the MuJoCo WASM module.
   * @param xml scene MJCF from `sceneXml`.
   * @param random () => number in [0, 1), draws the motor delays.
   */
  constructor(mj, xml, random = Math.random) {
    this.mj = mj;
    this.random = random;
    this.model = mj.MjModel.from_xml_string(xml);
    this.data = new mj.MjData(this.model);
    const m = this.model;
    const id = (type, name) => {
      const index = mj.mj_name2id(m, OBJ[type], name);
      if (index < 0) throw new Error(`no ${type} named ${name}`);
      return index;
    };
    this.joints = {};
    for (const name of Object.keys(DEFAULT_JOINT_POS)) {
      const joint = id('joint', PREFIX + name);
      this.joints[name] = {
        q: m.jnt_qposadr[joint], v: m.jnt_dofadr[joint], actuator: id('actuator', PREFIX + name),
        low: m.jnt_range[2 * joint], high: m.jnt_range[2 * joint + 1],
      };
    }
    const sensor = (name) => m.sensor_adr[id('sensor', name)];
    this.sensors = {
      gyro: sensor(`${PREFIX}imu_gyro`),
      headGyro: sensor(`${PREFIX}head_imu_gyro`),
      body: BODY_CONTACT_BODIES.map((body) => sensor(`body_ground_contact_${body}_force`)),
      wheels: WHEEL_BODIES.map((body) => sensor(`wheel_ground_contact_${body}_force`)),
    };
    this.base = id('body', `${PREFIX}base_link`);
    this.wheelGeoms = ['LW_collision', 'RW_collision'].map((name) => id('geom', PREFIX + name));
    this.camera = id('camera', `${PREFIX}depth_front`);
    let mass = 0;
    for (const value of m.body_mass) mass += value;
    this.weight = mass * 9.81;

    this.lastAction = new Float32Array(ACTION_DIM);
    // Per delayed group: the commands of the last MAX_LAG + 1 physics steps, newest last.
    this.history = Robot.emptyHistory();
    // Largest norm of each body contact force over the physics steps of the current control step, and the wheel
    // forces after the last one.
    this.bodyForcePeak = new Float64Array(BODY_CONTACT_BODIES.length);
    this.bodyForce = new Float64Array(BODY_CONTACT_BODIES.length);
    this.wheelForce = [new Float64Array(3), new Float64Array(3)];
    this.lags = new Int32Array(GROUPS.length);
  }

  /** Put the robot at (x, y) with heading `yaw`, standing, at rest; clears the motor and contact memory. */
  reset({ x = 0, y = 0, z = SPAWN_HEIGHT, yaw = 0 } = {}) {
    const { mj, model, data } = this;
    mj.mj_resetData(model, data);
    const qpos = data.qpos;
    qpos.set([x, y, z, Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)], 0);
    for (const [name, joint] of Object.entries(this.joints)) qpos[joint.q] = DEFAULT_JOINT_POS[name];
    this.clearMemory();
    mj.mj_forward(model, data);
  }

  /** Overwrite the simulation state (arrays of nq and nv numbers) and recompute everything derived from it. */
  setState(qpos, qvel) {
    this.data.qpos.set(qpos);
    this.data.qvel.set(qvel);
    this.mj.mj_forward(this.model, this.data);
  }

  /**
   * The command history right after a reset: targets of 0 for every joint. mjlab writes the motor commands once
   * when it resets an env, before any action exists, and that all-zero command fills the delay buffer; a delayed
   * motor therefore pulls towards angle 0 (not the standing pose) for the first 2-5 physics steps of an episode.
   */
  static emptyHistory() {
    return GROUPS.map((group) => Array.from({ length: MAX_LAG + 1 }, () => group.joints.map(() => 0)));
  }

  clearMemory() {
    this.lastAction.fill(0);
    this.history = Robot.emptyHistory();
    this.bodyForcePeak.fill(0);
    this.bodyForce.fill(0);
    for (const force of this.wheelForce) force.fill(0);
  }

  forward() {
    this.mj.mj_forward(this.model, this.data);
  }

  /**
   * One control step with `action` (7 numbers): DECIMATION physics steps.
   * @param lags optional array of DECIMATION arrays with the delay of each delayed group, in place of random draws.
   */
  step(action, lags = null) {
    this.lastAction.set(action);
    this.bodyForcePeak.fill(0);
    for (let k = 0; k < DECIMATION; k++) this.substep(lags ? lags[k] : null);
  }

  /** One physics step: motor torques from the delayed targets of `lastAction`, mj_step, contact sensors. */
  substep(lags = null) {
    this.applyMotors(lags);
    this.mj.mj_step(this.model, this.data);
    this.readContacts();
  }

  /** Write the motor torques of this physics step into data.ctrl. */
  applyMotors(lags = null) {
    const { data, lastAction: action } = this;
    const qpos = data.qpos;
    const qvel = data.qvel;
    const ctrl = data.ctrl;
    GROUPS.forEach((group, g) => {
      // The command of the group: one target per joint, a position or, for the wheels, a velocity.
      const command = group.joints.map((name) => {
        const index = ACTION_JOINTS.indexOf(name);
        if (index < 0) return DEFAULT_JOINT_POS[name]; // Head roll: held at 0.
        const offset = group.velocity ? 0 : DEFAULT_JOINT_POS[name];
        return offset + ACTION_SCALE[name] * action[index];
      });
      let target = command;
      if (group.delayed) {
        const history = this.history[g];
        history.push(command);
        history.shift();
        const lag = lags ? lags[g] : MIN_LAG + Math.floor(this.random() * (MAX_LAG - MIN_LAG + 1));
        this.lags[g] = lag;
        target = history[history.length - 1 - lag];
      }
      group.joints.forEach((name, i) => {
        const joint = this.joints[name];
        const q = qpos[joint.q];
        const v = qvel[joint.v];
        let torque = group.velocity ? group.kd * (target[i] - v) : group.kp * (target[i] - q) - group.kd * v;
        torque = clamp(torque, -group.limit, group.limit);
        if (group.spring) torque = clamp(torque + kneeSpring(q), -group.simLimit, group.simLimit);
        ctrl[joint.actuator] = torque;
      });
    });
  }

  /** Read the contact sensors of the physics step that just ran. */
  readContacts() {
    const s = this.data.sensordata;
    this.sensors.body.forEach((adr, i) => {
      this.bodyForce[i] = Math.hypot(s[adr], s[adr + 1], s[adr + 2]);
      this.bodyForcePeak[i] = Math.max(this.bodyForcePeak[i], this.bodyForce[i]);
    });
    this.sensors.wheels.forEach((adr, i) => this.wheelForce[i].set([s[adr], s[adr + 1], s[adr + 2]]));
  }

  /** Pose of the base: position [x, y, z] and rotation quaternion [w, x, y, z]. */
  basePose() {
    const p = this.data.xpos.subarray(3 * this.base, 3 * this.base + 3);
    const q = this.data.xquat.subarray(4 * this.base, 4 * this.base + 4);
    return { position: [p[0], p[1], p[2]], quat: [q[0], q[1], q[2], q[3]] };
  }

  /** Velocity of the base link in the world frame: linear [3] at the link origin and angular [3]. */
  baseVelocity() {
    const { data, base } = this;
    const cvel = data.cvel.subarray(6 * base, 6 * base + 6); // Angular, then linear at the subtree centre of mass.
    const com = data.subtree_com.subarray(3 * base, 3 * base + 3);
    const p = data.xpos.subarray(3 * base, 3 * base + 3);
    const o = [com[0] - p[0], com[1] - p[1], com[2] - p[2]];
    const w = [cvel[0], cvel[1], cvel[2]];
    return {
      linear: [cvel[3] - (w[1] * o[2] - w[2] * o[1]), cvel[4] - (w[2] * o[0] - w[0] * o[2]),
        cvel[5] - (w[0] * o[1] - w[1] * o[0])],
      angular: w,
    };
  }

  /** Gravity direction in the base frame (unit vector; [0, 0, -1] when upright). */
  static projectedGravity([w, x, y, z]) {
    return [-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))];
  }

  /** Yaw of the base x axis in the world (rad). */
  static heading([w, x, y, z]) {
    return Math.atan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z));
  }

  /**
   * The low-level observation (40 numbers) of both frozen policies.
   * @param twist [vx, vy, yaw rate] command.
   * @param height base height command (m).
   * @param jumpCommand [flag, phase, target height, distance].
   */
  observation(twist, height, jumpCommand) {
    const { data, sensors } = this;
    const s = data.sensordata;
    const qpos = data.qpos;
    const qvel = data.qvel;
    const o = new Float32Array(OBSERVATION_DIM);
    o.set([s[sensors.gyro], s[sensors.gyro + 1], s[sensors.gyro + 2]], 0);
    o.set(Robot.projectedGravity(this.basePose().quat), 3);
    o.set(twist, 6);
    POSITION_OBS_JOINTS.forEach((name, i) => { o[9 + i] = qpos[this.joints[name].q] - DEFAULT_JOINT_POS[name]; });
    VELOCITY_OBS_JOINTS.forEach((name, i) => { o[14 + i] = qvel[this.joints[name].v]; });
    o.set(this.lastAction, 21);
    o.set([0, 0, height], 28); // Base pitch, roll and height command.
    o.set([0, 1], 31); // Gimbal: yaw target 0, enabled.
    o.set([s[sensors.headGyro], s[sensors.headGyro + 1], s[sensors.headGyro + 2]], 33);
    o.set(jumpCommand, 36);
    return o;
  }

  /**
   * What the jump command and the terminations measure on a control step (python/beni_rl/jump.py, JumpMeasurement).
   * Call after `step` and before `forward`. `origin` is the lane origin [x, y, z].
   */
  measure(origin = [0, 0, 0]) {
    const { data } = this;
    const qpos = data.qpos;
    const { position, quat } = this.basePose();
    const velocity = this.baseVelocity();
    const gravity = Robot.projectedGravity(quat);
    const contacts = this.wheelForce.map((force) => Math.hypot(force[0], force[1], force[2]) > CONTACT_FORCE);
    const body = this.bodyForcePeak.some((force) => force > CONTACT_FORCE);
    let margin = Infinity;
    let error = 0;
    for (const [name, reference] of Object.entries(Q_REF)) {
      const joint = this.joints[name];
      const q = qpos[joint.q];
      margin = Math.min(margin, q - joint.low, joint.high - q);
      error = Math.max(error, Math.abs(q - reference));
    }
    const height = position[2] - origin[2];
    const tilt = (Math.acos(clamp(-gravity[2], -1, 1)) * 180) / Math.PI;
    const load = this.wheelForce.reduce((sum, force) => sum + Math.max(-force[2], 0), 0) / this.weight;
    const angular = Math.hypot(...velocity.angular);
    const clearance = this.wheelGeoms.map((geom) => {
      const vertical = Math.abs(data.geom_xmat[9 * geom + 8]);
      const extent = WHEEL_RADIUS * Math.sqrt(Math.max(0, 1 - vertical * vertical)) + WHEEL_HALF_WIDTH * vertical;
      return data.geom_xpos[3 * geom + 2] - origin[2] - extent;
    });
    return {
      position, quat, velocity, gravity, height, tilt, load, error, margin, angular, contacts, body, clearance,
      wz: velocity.angular[2],
      jumpReady: tilt < 20 && Math.abs(height - H_REF) <= 0.025 && error <= 0.3 && margin > 0.02
        && contacts.every(Boolean) && load >= 0.7 && !body && angular < 2,
      air: !contacts.some(Boolean) && !body && clearance.every((c) => c >= 0.01),
    };
  }
}

/** A small seeded random generator (mulberry32): () => number in [0, 1). */
export function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
