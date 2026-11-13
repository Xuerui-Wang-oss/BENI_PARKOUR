# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""Measure how well a checkpoint tracks height commands down to the crouch. No training.

Each robot stands for one second, then follows one height command at one speed for the rest of
the run; the settled base height is averaged over the last two seconds.

    python -m beni_rl_crawl.crouch_check logs/rsl_rl/beni_crawl_flat/<run>/model_13500.pt
    python -m beni_rl_crawl.crouch_check assets/models/policy_best.pt --heights 0.11 0.08 --speeds 0 1
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict

import torch
from mjlab.envs import ManagerBasedRlEnv
from rsl_rl.models import RNNModel
from tensordict import TensorDict

from . import mdp
from .beni_crawl_env_cfg import make_beni_crawl_env_cfg
from .recovery import H_LOW
from .rl_cfg import beni_crawl_ppo_runner_cfg

STAND_HEIGHT = .11
SWITCH_S = 1.  # The height and speed commands start this long after the reset.
SETTLE_S = 2.  # Averaging window at the end of the run.
TOLERANCE_M = .005


def load_actor(checkpoint, device):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    actor_cfg = {k: v for k, v in asdict(beni_crawl_ppo_runner_cfg().actor).items() if v is not None}
    actor_cfg.pop('class_name')
    actor = RNNModel(obs=TensorDict({'actor': torch.zeros(1, 40)}, batch_size=[1]),
                     obs_groups={'actor': ['actor']}, obs_set='actor', output_dim=7, **actor_cfg)
    actor.load_state_dict(saved['actor_state_dict'], strict=True)
    return actor.as_onnx().to(device).eval()


def make_env_cfg(num_envs):
    cfg = make_beni_crawl_env_cfg(play=True)
    cfg.scene.num_envs = num_envs
    # Keep every robot running to the end; the commands are written here, not sampled.
    cfg.terminations = {}
    cfg.auto_reset = False
    twist = cfg.commands['twist']
    twist.resampling_time_range = (1e9, 1e9)
    twist.rel_standing_envs = twist.rel_forward_envs = 0.
    twist.heading_command = False
    twist.ranges.heading = None
    twist.ranges.lin_vel_x = twist.ranges.lin_vel_y = twist.ranges.ang_vel_z = (0., 0.)
    twist.pitch_range = twist.roll_range = (0., 0.)
    cfg.commands['gimbal'].yaw_range = (0., 0.)
    cfg.commands['gimbal'].enable_probability = 1.
    return cfg


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', help='Path of a model_<N>.pt file')
    parser.add_argument('--heights', type=float, nargs='+', default=[.11, .10, .09, .08, .07, H_LOW],
                        help='Height commands in metres')
    parser.add_argument('--speeds', type=float, nargs='+', default=[0., .5, 1.], help='Forward speed commands in m/s')
    parser.add_argument('--seconds', type=float, default=6., help='Length of each run')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    heights = torch.tensor(args.heights).repeat_interleave(len(args.speeds))
    speeds = torch.tensor(args.speeds).repeat(len(args.heights))
    env = ManagerBasedRlEnv(make_env_cfg(len(heights)), device=args.device)
    policy = load_actor(args.checkpoint, args.device)
    heights, speeds = heights.to(args.device), speeds.to(args.device)
    robot = env.scene['robot']
    twist = env.command_manager.get_term('twist')
    knees = robot.find_joints(('L2', 'R2'), preserve_order=True)[0]
    hidden = torch.zeros(1, len(heights), 512, device=args.device)
    env.reset()

    steps = round(args.seconds / env.step_dt)
    settle = steps - round(SETTLE_S / env.step_dt)
    count = 0
    height_sum = torch.zeros_like(heights)
    knee_sum, speed_sum, max_tilt = torch.zeros_like(heights), torch.zeros_like(heights), torch.zeros_like(heights)
    body_hit = torch.zeros(len(heights), dtype=torch.bool, device=args.device)
    for step in range(steps):
        started = step * env.step_dt >= SWITCH_S
        twist.vel_command_b[:, 0] = speeds * started
        twist.vel_command_b[:, 2] = -2. * robot.data.heading_w  # Steer back to the initial direction.
        twist.base_pose_command[:, 2] = heights if started else STAND_HEIGHT
        # Recompute the observation after writing the commands.
        action, hidden, _ = policy(env.observation_manager.compute_group('actor'), hidden)
        env.step(action)
        if started:
            max_tilt = torch.maximum(max_tilt, mdp.base_tilt(env))
            body_hit |= mdp._contact_mask(env, 'body_ground_contact', .1).any(dim=1)
        if step >= settle:
            count += 1
            height_sum += mdp.base_height(env).squeeze(1)
            knee_sum += robot.data.joint_pos[:, knees].mean(dim=1)
            speed_sum += robot.data.root_link_lin_vel_b[:, 0]

    height, knee, speed = height_sum / count, knee_sum / count, speed_sum / count
    error = height - heights
    tilt = torch.rad2deg(max_tilt)
    passed = (error.abs() <= TOLERANCE_M) & ~body_hit & (tilt < 20.)
    print(f'checkpoint {args.checkpoint}: settled values over the last {SETTLE_S:g} s of {args.seconds:g} s')
    print(' cmd_h  speed |  height   error   knee   vx   max_tilt  body_hit | ok')
    for i in range(len(heights)):
        print(f' {heights[i]:.3f}  {speeds[i]:4.1f}  |  {height[i]:.4f}  {error[i]*1000:+5.1f}mm  {knee[i]:.2f}  {speed[i]:+.2f}   {tilt[i]:5.1f}     '
              f'{str(bool(body_hit[i])):5s} | {"ok" if passed[i] else "--"}')
    print(f'{int(passed.sum())}/{len(heights)} within {TOLERANCE_M*1000:g} mm without body contact or tilt above 20 degrees; '
          f'largest error {error.abs().max()*1000:.1f} mm at command {heights[error.abs().argmax()]:.3f} m')
    env.close()


if __name__ == '__main__':
    main()
