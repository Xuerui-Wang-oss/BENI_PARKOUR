"""Preview of the frozen BENI policy lowering its body to pass under a floating box. No training.

One robot per lane, each with its own height command, all driving at one speed toward a box with
the same clearance. The policy was never trained on height commands below 0.10 m; this shows how
the shipped checkpoint already reacts to them.

--model exclude removes the wheel-thigh collision pairs, which otherwise stop the knee at 0.69 rad.

    python -m beni_rl.crawl_preview
    python -m beni_rl.crawl_preview --model exclude --clearance 0.218
    python -m beni_rl.crawl_preview --heights 0.11 0.05 0.04 0.03 --speed 1.0
    python -m beni_rl.crawl_preview --steps 1100    # No viewer; print the status lines only.
"""
from __future__ import annotations

import argparse

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.terrains import TerrainGeneratorCfg
from mjlab.utils.lab_api.math import matrix_from_quat

from . import ASSETS_DIR
from . import parkour_obstacle_test as obstacle_test
from .beni_robot import get_beni_spec

STAND_HEIGHT = .11
LOWER_S = 1.  # The height command switches this long after a reset; driving starts at SETTLE_S.
PAST_BOX = 1.  # Drive this far beyond the box, then stop and stand up.
STAND_S = 2.


def excluded_spec():
    spec = get_beni_spec()
    for side in 'LR':
        spec.add_exclude(bodyname1=f'{side}1_Link', bodyname2=f'{side}W_Link')
    return spec


def make_env_cfg(lanes, clearance, model):
    cfg = obstacle_test.make_test_env_cfg(lanes)
    # ObstacleLaneCfg reads the clearance from this table; with one row only the first value is used.
    obstacle_test.OBSTACLES['hollow'] = (clearance, clearance)
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        size=obstacle_test.LANE, num_rows=1, curriculum=True, seed=0,
        sub_terrains={f'lane_{i}': obstacle_test.ObstacleLaneCfg(kind='hollow') for i in range(lanes)})
    if model == 'exclude':
        cfg.scene.entities['robot'].spec_fn = excluded_spec
    return cfg


class CrouchDriver:
    """Stand, lower to the commanded height, drive under the box, stand up again, then repeat.

    Called like a policy, once per control step, so it can also run inside a viewer.
    """

    def __init__(self, env, policy, heights, speed):
        self.env, self.policy, self.speed = env, policy, speed
        device, n = env.device, env.num_envs
        self.heights = torch.tensor(heights, device=device)
        # Put each robot on its own lane instead of the generator's random assignment.
        terrain = env.scene.terrain
        lane = torch.arange(n, device=device)
        terrain.terrain_types[:] = lane
        terrain.terrain_levels[:] = 0
        terrain.env_origins[:] = terrain.terrain_origins[0, lane]
        env.reset()

        self.robot = env.scene['robot']
        self.jump = env.command_manager.get_term('jump')
        self.twist = env.command_manager.get_term('twist')
        self.hidden = torch.zeros(1, n, 512, device=device)
        self.past = torch.zeros(n, device=device, dtype=torch.bool)
        self.hit = self.past.clone()
        (self.head,) = self.robot.find_geoms('head_collision')[0]
        model = env.sim.mj_model
        (geom,) = (i for i in range(model.ngeom) if model.geom(i).name.endswith('head_collision'))
        self.head_half_size = torch.tensor(model.geom_size[geom], device=device, dtype=torch.float32)
        self.end_x = obstacle_test.RUN_UP + obstacle_test.HOLLOW_LENGTH + PAST_BOX
        self.steps = round((obstacle_test.SETTLE_S + self.end_x / speed + STAND_S) / env.step_dt)

    def _report(self):
        sample = self.jump.sample
        step = int(self.env.episode_length_buf[0])
        if sample is None or step == 0:  # No step of this run has finished yet.
            return
        self.hit |= sample['body']
        if step % 100:
            return
        data = self.robot.data
        rotation = matrix_from_quat(data.geom_quat_w[:, self.head])
        top = (data.geom_pos_w[:, self.head, 2] + (rotation[:, 2].abs() * self.head_half_size).sum(-1)
               - self.env.scene.env_origins[:, 2])
        cells = [f'cmd {c:.3f}: base {h:.3f} head top {z:.3f} knee {k:.2f}{" HIT" if b else ""}'
                 for c, h, z, k, b in zip(self.heights.tolist(), sample['height'].tolist(), top.tolist(),
                                          sample['knee'].mean(-1).tolist(), self.hit.tolist())]
        print(f't={step * self.env.step_dt:4.1f}s  ' + '  |  '.join(cells), flush=True)

    def __call__(self, obs=None):
        del obs  # Recomputed below, after the scripted commands change.
        env = self.env
        if env.episode_length_buf[0] >= self.steps:
            env.reset()
        # A reset from here or from the viewer starts the run again.
        fresh = env.episode_length_buf == 0
        self.hidden[:, fresh] = 0
        self.past[fresh] = False
        self.hit[fresh] = False

        t = env.episode_length_buf * env.step_dt
        x = self.robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
        self.past |= x > self.end_x
        self.twist.vel_command_b[:, 0] = torch.where(self.past, 0., self.speed) * (t >= obstacle_test.SETTLE_S)
        # The test config samples no yaw command; steer back to the lane direction.
        self.twist.vel_command_b[:, 2] = -2. * self.robot.data.heading_w
        self.twist.base_pose_command[:, 2] = torch.where((t >= LOWER_S) & ~self.past, self.heights, STAND_HEIGHT)
        self._report()
        action, self.hidden, _ = self.policy(env.get_observations()['actor'], self.hidden)
        return action


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=('stock', 'exclude'), default='stock',
                        help='Robot model: as shipped, or with the wheel-thigh collision pairs removed')
    parser.add_argument('--heights', type=float, nargs='+', default=[STAND_HEIGHT, .04, .03],
                        help='Height command of each lane (m)')
    parser.add_argument('--clearance', type=float, default=.225, help='Gap under the box (m)')
    parser.add_argument('--speed', type=float, default=.5, help='Forward speed command (m/s)')
    parser.add_argument('--checkpoint', default='policy_best.pt', help='File under assets/models, or a path')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--steps', type=int, help='Run this many control steps without the viewer')
    args = parser.parse_args()

    env = ManagerBasedRlEnv(make_env_cfg(len(args.heights), args.clearance, args.model), device=args.device)
    policy = obstacle_test.load_actor(ASSETS_DIR / 'models' / args.checkpoint).as_onnx().to(args.device).eval()
    driver = CrouchDriver(env, policy, args.heights, args.speed)
    print(f'{args.model} model, box clearance {args.clearance} m, {args.speed} m/s. '
          'Robots in the viewer (Scene > Environment > Select):')
    for i, height in enumerate(args.heights):
        print(f'  env {i}: height command {height} m')
    if args.steps:
        for _ in range(args.steps):
            env.step(driver())
    else:
        from .play import BeniViserPlayViewer
        BeniViserPlayViewer(env, driver).run()
    env.close()


if __name__ == '__main__':
    main()
