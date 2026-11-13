"""Scripted obstacle test for the frozen BENI policy. No training.

A privileged script drives each robot toward one obstacle and requests a jump at a set distance
before it: cubes to land on, pits to clear, hollow cubes to drive under. Outcomes are judged from
simulator state rather than the jump command's own success flag, which measures height from a flat
ground at z=0.

    python -m beni_rl.parkour_obstacle_test --checkpoint policy_best.pt --trials 4
    python -m beni_rl.parkour_obstacle_test --view
    python -m beni_rl.parkour_obstacle_test --view --kind cube --speed 0.5 --trigger 0.25
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.terrains import SubTerrainCfg, TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput
from rsl_rl.models import RNNModel
from tensordict import TensorDict

from . import ASSETS_DIR, REPO_ROOT
from .beni_jump_env_cfg import make_beni_jump_env_cfg
from .recovery import H_REF
from .rl_cfg import beni_jump_ppo_runner_cfg

LANE = (6., 1.2)  # Length along x, width along y.
SPAWN_X = 1.
RUN_UP = 2.  # Spawn to the obstacle's front edge.
GROUND_THICKNESS = .5
CUBE_LENGTH = 1.
PIT_DEPTH = .25
HOLLOW_LENGTH = .3
HOLLOW_THICKNESS = .1
ROWS = 6
# Value at the first and last row: cube height, pit length, hollow-cube clearance (m).
OBSTACLES = {'cube': (.15, .12), 'pit': (.30, .35), 'hollow': (.20, .21)}
SPEEDS = (3, 3, 3)
TRIGGERS = (None, .15, .25, .35, .41, .55)  # Base-to-front-edge distance at the jump request; None never jumps.
SETTLE_S = 2.
AFTER_S = 4.
# Speed and trigger shown by --view: the best-performing setting of each kind in the sweep.
VIEW_DEFAULTS = {'cube': (.5, .25), 'pit': (1., .25), 'hollow': (1., None)}


def obstacle_value(kind, difficulty):
    first, last = OBSTACLES[kind]
    return first + difficulty * (last - first)


@dataclass(kw_only=True)
class ObstacleLaneCfg(SubTerrainCfg):
    """One straight lane along +x with a single full-width obstacle."""

    kind: str

    def function(self, difficulty, spec, rng):
        del rng
        value = obstacle_value(self.kind, difficulty)
        body = spec.body('terrain')
        length, width = self.size
        front = SPAWN_X + RUN_UP

        def box(x_min, x_max, z_min, z_max, color):
            geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                                 size=((x_max-x_min)/2, width/2, (z_max-z_min)/2),
                                 pos=((x_min+x_max)/2, width/2, (z_min+z_max)/2))
            return TerrainGeometry(geom=geom, color=color)

        ground, obstacle = (.5, .5, .5, 1.), (.9, .5, .1, 1.)
        if self.kind == 'pit':
            geometries = [box(0., front, -GROUND_THICKNESS, 0., ground),
                          box(front, front+value, -GROUND_THICKNESS, -PIT_DEPTH, ground),
                          box(front+value, length, -GROUND_THICKNESS, 0., ground)]
        else:
            geometries = [box(0., length, -GROUND_THICKNESS, 0., ground)]
            if self.kind == 'cube':
                geometries.append(box(front, front+CUBE_LENGTH, 0., value, obstacle))
            else:
                geometries.append(box(front, front+HOLLOW_LENGTH, value, value+HOLLOW_THICKNESS, obstacle))
        return TerrainOutput(origin=np.array((SPAWN_X, width/2, 0.)), geometries=geometries)


def make_test_env_cfg(num_envs):
    cfg = make_beni_jump_env_cfg(play=True)
    cfg.scene.num_envs = num_envs
    cfg.scene.terrain.terrain_type = 'generator'
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        size=LANE, num_rows=ROWS, curriculum=True, seed=0,
        sub_terrains={kind: ObstacleLaneCfg(kind=kind) for kind in OBSTACLES})
    # Keep every trial running to the end; outcomes are measured here.
    cfg.terminations = {}
    cfg.auto_reset = False
    cfg.events['reset_base'].params['pose_range'] = {'y': (-.05, .05), 'yaw': (-.05, .05)}
    twist = cfg.commands['twist']
    twist.resampling_time_range = (1e9, 1e9)
    twist.rel_standing_envs = 0.
    twist.heading_command = False
    twist.ranges.heading = None
    twist.ranges.lin_vel_x = twist.ranges.lin_vel_y = twist.ranges.ang_vel_z = (0., 0.)
    twist.pitch_range = twist.roll_range = (0., 0.)
    twist.height_range = (.11, .11)
    cfg.commands['gimbal'].yaw_range = (0., 0.)
    cfg.commands['gimbal'].enable_probability = 1.
    return cfg


def load_actor(checkpoint):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    actor_cfg = {k: v for k, v in asdict(beni_jump_ppo_runner_cfg().actor).items() if v is not None}
    actor_cfg.pop('class_name')
    actor = RNNModel(obs=TensorDict({'actor': torch.zeros(1, 40)}, batch_size=[1]),
                     obs_groups={'actor': ['actor']}, obs_set='actor', output_dim=7, **actor_cfg)
    actor.load_state_dict(saved['actor_state_dict'], strict=True)
    return actor.eval()


class ScriptedDriver:
    """Drive each robot at its obstacle, request the jump and record what happens.

    Called like a policy, once per control step, so it can also run inside a viewer. Each
    condition is (obstacle column, lane row, speed, trigger distance or None).
    """

    def __init__(self, env, policy, conditions, loop=False):
        self.env, self.policy, self.conditions, self.loop = env, policy, conditions, loop
        device, n = env.device, env.num_envs
        self.kinds = list(OBSTACLES)
        col = torch.tensor([c[0] for c in conditions], device=device)
        row = torch.tensor([c[1] for c in conditions], device=device)
        self.speed = torch.tensor([c[2] for c in conditions], device=device)
        self.has_trigger = torch.tensor([c[3] is not None for c in conditions], device=device)
        self.trigger = torch.tensor([c[3] or 0. for c in conditions], device=device)
        self.value = torch.tensor([obstacle_value(self.kinds[c[0]], c[1]/(ROWS-1)) for c in conditions], device=device)
        self.is_cube, is_pit = col == self.kinds.index('cube'), col == self.kinds.index('pit')
        self.length = torch.where(self.is_cube, CUBE_LENGTH, torch.where(is_pit, self.value, HOLLOW_LENGTH))
        self.duration = ((SETTLE_S + RUN_UP/self.speed + AFTER_S) / env.step_dt).round().long()
        self.steps = int(self.duration.max())

        # Put each trial on its own lane instead of the generator's random assignment.
        terrain = env.scene.terrain
        terrain.terrain_types[:] = col
        terrain.terrain_levels[:] = row
        terrain.env_origins[:] = terrain.terrain_origins[row, col]
        env.reset()

        self.robot = env.scene['robot']
        self.jump = env.command_manager.get_term('jump')
        self.twist = env.command_manager.get_term('twist')
        self.hidden = torch.zeros(1, n, 512, device=device)

        def zeros(dtype=torch.float32):
            return torch.zeros(n, device=device, dtype=dtype)
        self.started, self.requested, self.lifted, self.landed, self.stop, self.body_hit = (
            zeros(torch.bool) for _ in range(6))
        self.air_run = zeros(torch.long)
        self.request_distance, self.request_speed, self.landing_x, self.max_tilt, self.peak_clearance = (
            zeros() for _ in range(5))

    def _distance(self):
        """Base to the obstacle's front edge; negative once the base is past it."""
        return RUN_UP - (self.robot.data.root_link_pos_w[:, 0] - self.env.scene.env_origins[:, 0])

    def _record(self):
        """Fold the control step that just ran into the per-trial records."""
        sample = self.jump.sample
        if sample is None:  # No step has run yet.
            return
        wheel_contact = sample['contacts'].any(-1)
        self.air_run = torch.where(self.requested & ~wheel_contact, self.air_run+1, 0)
        self.lifted |= self.air_run >= 5
        landing = self.lifted & ~self.landed & wheel_contact
        self.landing_x[landing] = -self._distance()[landing]
        self.landed |= landing
        self.body_hit |= sample['body'] & self.started
        self.max_tilt = torch.maximum(self.max_tilt, sample['tilt'] * self.started)
        self.peak_clearance = torch.maximum(
            self.peak_clearance, sample['clearance'].amin(-1) * (self.lifted & ~self.landed))

    def __call__(self, obs=None):
        del obs  # Recomputed below, after the scripted commands change.
        env = self.env
        if self.loop:
            done = (env.episode_length_buf >= self.duration).nonzero().flatten()
            if len(done):
                env.reset(env_ids=done)
        self._record()
        # A reset from here or from the viewer starts that trial again.
        fresh = env.episode_length_buf == 0
        for record in (self.requested, self.lifted, self.landed, self.stop, self.body_hit, self.air_run,
                       self.request_distance, self.request_speed, self.landing_x, self.max_tilt,
                       self.peak_clearance):
            record[fresh] = 0
        self.hidden[:, fresh] = 0

        self.started = env.episode_length_buf * env.step_dt >= SETTLE_S
        distance = self._distance()
        self.stop |= torch.where(self.is_cube, self.landed | (~self.has_trigger & (distance < -.1)),
                                 distance < -(self.length+1.))
        self.stop |= distance < -(self.length+1.)
        vx = torch.where(self.stop, 0., self.speed) * self.started
        self.twist.vel_command_b[:, 0] = vx
        # The jump command holds the request-time speed until it sees a stable landing, which it
        # cannot see on a cube. Keep that held speed equal to the scripted one.
        self.jump.state['request_twist'][:, 0] = vx
        due = self.has_trigger & ~self.requested & (distance <= self.trigger) & self.started
        if due.any():
            ids = due.nonzero().flatten()
            accepted = ids[self.jump.request(ids) == 0]
            self.requested[accepted] = True
            self.request_distance[accepted] = distance[accepted]
            self.request_speed[accepted] = self.robot.data.root_link_lin_vel_b[accepted, 0]
        action, self.hidden, _ = self.policy(env.get_observations()['actor'], self.hidden)
        return action

    def results(self):
        """Per-trial outcomes. Call once, after the last control step."""
        self._record()
        sample = self.jump.sample
        position = self.robot.data.root_link_pos_w
        end_x = -self._distance()  # Relative to the front edge.
        reached = torch.where(
            self.is_cube,
            (end_x > .05) & (end_x < CUBE_LENGTH-.05) & ((position[:, 2] - self.value - H_REF).abs() < .03),
            (end_x > self.length + .3) & ((position[:, 2] - H_REF).abs() < .03))
        standing = ((sample['tilt'] < 20) & sample['contacts'].all(-1) &
                    (self.robot.data.root_link_lin_vel_w[:, :2].norm(dim=-1) < .2))
        success = reached & standing & ~self.body_hit
        return [dict(kind=self.kinds[c[0]], value=round(float(self.value[i]), 4), speed=c[2], trigger=c[3],
                     success=bool(success[i]), reached=bool(reached[i]), standing=bool(standing[i]),
                     body_hit=bool(self.body_hit[i]), requested=bool(self.requested[i]),
                     lifted=bool(self.lifted[i]), request_distance=float(self.request_distance[i]),
                     request_speed=float(self.request_speed[i]), landing_x=float(self.landing_x[i]),
                     end_x=float(end_x[i]), end_z=float(position[i, 2]), max_tilt=float(self.max_tilt[i]),
                     peak_clearance=float(self.peak_clearance[i]))
                for i, c in enumerate(self.conditions)]


@torch.inference_mode()
def run(checkpoint, conditions, device, view=False):
    env = ManagerBasedRlEnv(make_test_env_cfg(len(conditions)), device=device)
    driver = ScriptedDriver(env, load_actor(checkpoint).as_onnx().to(device).eval(), conditions, loop=view)
    rows = []
    if view:
        from .play import BeniViserPlayViewer
        BeniViserPlayViewer(env, driver).run()
    else:
        for _ in range(driver.steps):
            env.step(driver())
        rows = driver.results()
    env.close()
    return rows


def report(rows):
    for kind, unit in (('cube', 'height'), ('pit', 'length'), ('hollow', 'clearance')):
        triggers = sorted({r['trigger'] for r in rows if r['kind'] == kind}, key=lambda t: -1. if t is None else t)
        print(f'\n{kind}: successes / trials, by {unit} (m), speed (m/s) and jump trigger distance (m)')
        print('  value  speed ' + ''.join(f'{"no jump" if t is None else t:>9}' for t in triggers))
        for value in sorted({r['value'] for r in rows if r['kind'] == kind}, reverse=kind == 'hollow'):
            for speed in SPEEDS:
                cells = []
                for t in triggers:
                    cell = [r for r in rows if (r['kind'], r['value'], r['speed'], r['trigger']) == (kind, value, speed, t)]
                    cells.append(f'{sum(r["success"] for r in cell)}/{len(cell)}')
                print(f'  {value:5.2f}  {speed:4.1f}  ' + ''.join(f'{c:>9}' for c in cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', default='policy_best.pt', help='File under assets/models, or a path')
    parser.add_argument('--trials', type=int, default=4, help='Trials per condition')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--view', action='store_true',
                        help='Watch in the browser viewer, one robot per obstacle size, on repeat')
    parser.add_argument('--kind', choices=tuple(OBSTACLES), help='Watch only this obstacle; default all three')
    parser.add_argument('--speed', type=float, help='Forward speed command (m/s) with --view')
    parser.add_argument('--trigger', type=float,
                        help='Jump request distance before the obstacle (m) with --view; 0 never jumps')
    args = parser.parse_args()
    checkpoint = ASSETS_DIR / 'models' / args.checkpoint
    kinds = list(OBSTACLES)
    if args.view:
        conditions = []
        print('Robots in the viewer (Scene > Environment > Select):')
        for kind in ((args.kind,) if args.kind else kinds):
            speed, trigger = VIEW_DEFAULTS[kind]
            if args.speed is not None:
                speed = args.speed
            if args.trigger is not None and kind != 'hollow':
                trigger = args.trigger or None
            values = ', '.join(f'{obstacle_value(kind, row/(ROWS-1)):.2f}' for row in range(ROWS))
            jump = 'no jump' if trigger is None else f'jump requested {trigger} m before'
            print(f'  env {len(conditions)}-{len(conditions)+ROWS-1}: {kind} {values} m, {speed} m/s, {jump}')
            conditions += [(kinds.index(kind), row, speed, trigger) for row in range(ROWS)]
        run(checkpoint, conditions, args.device, view=True)
        return
    conditions = [(col, row, speed, trigger)
                  for col, kind in enumerate(kinds) for row in range(ROWS) for speed in SPEEDS
                  for trigger in ((None,) if kind == 'hollow' else TRIGGERS) for _ in range(args.trials)]
    rows = run(checkpoint, conditions, args.device)
    report(rows)
    output = REPO_ROOT / 'logs' / 'parkour_obstacle_test' / f'{checkpoint.stem}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=1) + '\n')
    print(f'\nPer-trial results: {output}')


if __name__ == '__main__':
    main()
