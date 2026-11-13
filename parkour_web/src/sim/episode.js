// One episode of the parkour hierarchy: the robot on a lane, driven by the high-level policy through the two frozen
// low-level policies. `step()` advances the simulation by one control step (10 ms); every SUB_STEPS-th call first
// renders the depth image and runs the depth encoder and the high-level policy (20 Hz).

import { Hierarchy, SUB_STEPS } from './hierarchy.js';

export class Episode {
  /**
   * @param robot a `Robot` that was just reset to the spawn pose.
   * @param camera a `DepthCamera` of the robot.
   * @param networks {encoder, policy, crawl, jump} from policies.js; the three policies are reset here.
   * @param lane {origin, spawnX, goalX}: see `Hierarchy`.
   */
  constructor(robot, camera, networks, lane) {
    this.robot = robot;
    this.camera = camera;
    this.networks = networks;
    this.hierarchy = new Hierarchy(robot, lane);
    for (const name of ['policy', 'crawl', 'jump']) networks[name].reset();
    this.controlSteps = 0;
    this.depth = null; // The latest depth image (Float32Array, 60 x 80), as the policy saw it.
    this.highLevelAction = [0, 0, 0]; // The latest [vx, height, trigger] action.
    this.done = false;
    this.outcome = '';
  }

  /** Advance one control step. @returns {done, outcome}; `outcome` names the termination terms that ended it. */
  async step() {
    if (this.done) return { done: true, outcome: this.outcome };
    const { robot, hierarchy, networks } = this;
    if (this.controlSteps % SUB_STEPS === 0) {
      this.depth = this.camera.render();
      const input = new Float32Array(464);
      input.set(hierarchy.observation(), 0);
      input.set(await networks.encoder.encode(this.depth), 24);
      this.highLevelAction = Array.from(await networks.policy.act(input));
      hierarchy.decodeAction(this.highLevelAction);
    }
    const { crawlObservation, jumpObservation } = hierarchy.beforePhysics();
    // Both policies run on every control step, so the one that takes over has a current memory.
    const crawlAction = await networks.crawl.act(crawlObservation);
    const jumpAction = await networks.jump.act(jumpObservation);
    robot.step(hierarchy.selectAction(crawlAction, jumpAction));
    const { done, outcome } = hierarchy.afterPhysics();
    this.controlSteps += 1;
    this.done = done;
    this.outcome = outcome;
    return { done, outcome };
  }
}
