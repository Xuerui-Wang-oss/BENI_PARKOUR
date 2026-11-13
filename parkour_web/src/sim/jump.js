// The jump command: a port of the parts of python/beni_rl/jump.py (BeniJumpCommand) that decide behaviour, as the
// parkour env configures it (python/parkour_rl/env.py: training mode, no automatic jumps, cool-down of 20 control
// steps, 0.2 s of readiness after touchdown before the next request).
//
// A request is accepted when the robot has been ready for 0.2 s, no jump is being tracked and the cool-down is over.
// The jump then waits ("preparing") until the robot has also driven straight for 0.2 s, plays the 41-frame reference
// (0.4 s) that the jump policy follows, and is tracked through flight and landing until the robot has been ready
// again for 0.2 s after touchdown.
//
// Times are kept as 32-bit floats and compared like PyTorch compares a float32 tensor with a number, so that a
// threshold is crossed on the same control step as in Python.

const f = Math.fround;
const DT = f(0.01);
const READINESS_STEPS = 20; // 0.2 s of readiness before a request is accepted.
const YAW_SETTLE_STEPS = 20; // 0.2 s of straight driving before the reference starts.
const REFERENCE_FRAMES = 41;
const REFERENCE_HEIGHT = 0.18;
const TAKEOFF_YAW_RATE = 0.3; // rad/s
const PREPARE_TIMEOUT_S = f(2);
const AIR_HOLD_STEPS = 5; // 0.05 s without contact count as lifted.
const STABLE_HOLD_STEPS = 100;
const TAKEOFF_DEADLINE_S = f(0.8);
const PROTECTION_S = f(2);
const LANDING_STABLE_DEADLINE_S = f(1.5);
const FOLLOWUP_S = f(5);

export class JumpCommand {
  constructor({ coolDownSteps = 20, retriggerHoldSteps = 20 } = {}) {
    this.coolDownSteps = coolDownSteps;
    this.retriggerHoldSteps = retriggerHoldSteps;
    this.reset();
  }

  /** Start of an episode. */
  reset() {
    this.flag = false; // The reference is playing.
    this.count = 0; // Frame of the reference.
    this.coolDown = 0;
    this.readySteps = 0;
    this.yawReadySteps = 0;
    this.preparing = false;
    this.prepareAge = f(0);
    this.headingActive = false;
    this.requestTwist = [0, 0, 0];
    this.clearTracking();
  }

  clearTracking() {
    this.tracking = false;
    this.gate = false;
    this.lifted = false;
    this.failed = false;
    this.age = f(0);
    this.airRun = 0;
    this.readyHold = 0;
    this.landS = f(-1);
    this.stableS = f(-1);
  }

  /** [flag, phase, target height, distance]: entries 36-39 of the low-level observation. */
  get command() {
    return this.flag ? [1, f(f(this.count * 0.01) / f(0.4)), REFERENCE_HEIGHT, 0] : [0, 0, 0, 0];
  }

  /** The straight command is protected through propulsion and flight. */
  get directionActive() {
    return this.tracking && !this.preparing && this.landS < 0 && this.age < PROTECTION_S
      && (this.age <= TAKEOFF_DEADLINE_S || this.lifted);
  }

  /** The twist command is held at its value at the request: through preparation, flight and landing. */
  get commandHold() {
    const landing = this.tracking && this.landS >= 0 && this.stableS < 0
      && this.age < f(this.landS + LANDING_STABLE_DEADLINE_S);
    return this.headingActive && (this.preparing || this.directionActive || landing);
  }

  /**
   * Ask for a jump. `twist` is the [vx, vy, yaw rate] command array; an accepted request zeroes its yaw rate.
   * @returns 0 accepted, 1 not ready, 2 cool-down, 3 busy.
   */
  request(twist) {
    let code = 0;
    if (this.readySteps < READINESS_STEPS) code = 1;
    if (this.coolDown > 0) code = 2;
    if (this.tracking) code = 3;
    if (code) return code;
    this.clearTracking();
    this.tracking = true;
    this.preparing = true;
    this.prepareAge = f(0);
    if (Math.abs(twist[2]) >= 1e-6) this.yawReadySteps = 0;
    twist[2] = 0;
    this.headingActive = true;
    this.requestTwist = [twist[0], twist[1], twist[2]];
    if (this.yawReadySteps >= YAW_SETTLE_STEPS) this.beginJump();
    return 0;
  }

  beginJump() {
    this.preparing = false;
    this.gate = true;
    this.flag = true;
    this.count = 0;
  }

  clearCommand() {
    this.flag = false;
    this.count = 0;
  }

  finish() {
    this.tracking = false;
    this.gate = false;
    this.preparing = false;
  }

  /**
   * Update the measurements of the jump once per control step, after the physics and before the commands advance.
   * @param m `Robot.measure()`.
   * @param commandWz the yaw-rate command of this control step.
   * @param terminal the episode ends on this control step.
   */
  observe(m, commandWz, terminal = false) {
    const ready = m.jumpReady;
    this.readySteps = ready ? this.readySteps + 1 : 0;
    const straight = ready && Math.abs(m.wz) <= TAKEOFF_YAW_RATE && Math.abs(commandWz) < 1e-6;
    this.yawReadySteps = straight ? this.yawReadySteps + 1 : 0;
    const fallen = m.body && (m.tilt >= 60 || m.height < 0.065);
    if (fallen) this.headingActive = false;
    if (this.preparing) this.prepareAge = f(this.prepareAge + DT);
    if (this.preparing && (this.prepareAge >= PREPARE_TIMEOUT_S || m.body || fallen)) {
      this.failed = true;
      this.finish();
      this.headingActive = false;
      this.clearCommand();
      this.coolDown = this.coolDownSteps;
    }
    if (this.tracking && !this.preparing) this.age = f(this.age + DT);

    const activeAir = this.tracking && !this.preparing && m.air && this.landS < 0;
    this.airRun = activeAir ? this.airRun + 1 : 0;
    if (activeAir && this.airRun >= AIR_HOLD_STEPS) this.lifted = true;
    if (this.tracking && this.lifted && m.contacts.some(Boolean) && this.landS < 0) this.landS = this.age;
    const landed = this.landS >= 0;
    this.readyHold = this.tracking && landed && ready ? this.readyHold + 1 : 0;
    if (this.tracking && landed && this.readyHold >= STABLE_HOLD_STEPS && this.stableS < 0) {
      this.stableS = f(this.age - f((STABLE_HOLD_STEPS - 1) * 0.01));
    }
    const failure = this.tracking && (m.body || m.tilt > 60 || (this.age > TAKEOFF_DEADLINE_S && !this.lifted)
      || (this.age >= PROTECTION_S && (!landed || this.readySteps < READINESS_STEPS)));
    if (failure) this.failed = true;
    this.gate = this.gate && !failure && this.age < PROTECTION_S;

    if (terminal) return;
    // The tracking ends after a stable landing and its follow-up, after 9 s, or, as soon as the robot is ready
    // again, after a failure or `retriggerHoldSteps` of readiness since touchdown.
    const finished = this.tracking && ((this.stableS >= 0 && this.age >= f(f(this.stableS + 1) + FOLLOWUP_S))
      || this.age >= 9);
    if (finished) this.finish();
    const retry = this.tracking && !this.flag && this.coolDown === 0 && this.readySteps >= READINESS_STEPS
      && (this.failed || (landed && this.readyHold >= this.retriggerHoldSteps));
    if (retry) this.finish();
  }

  /** Advance the command once per control step, after `observe`: cool-down, reference frame, start of a jump. */
  update() {
    this.coolDown = Math.max(this.coolDown - 1, 0);
    if (this.flag) this.count += 1;
    if (this.count >= REFERENCE_FRAMES) {
      this.clearCommand();
      this.coolDown = this.coolDownSteps;
    }
    if (this.preparing && this.yawReadySteps >= YAW_SETTLE_STEPS) this.beginJump();
  }
}
