# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""Environment configuration of the high-level parkour task.

A high-level policy drives two frozen low-level policies, crawl and jump, through the lanes of parkour_terrain.py.
The task of an episode: reach the goal of the lane, stop there upright, and do so within the lane's time budget.

The env is the low-level jump env in its play form (no noise, no randomisation) with the crawl robot, the parkour
lanes and a depth camera. Its command terms are handed to ``hierarchy.ParkourVecEnvWrapper``; its rewards,
terminations and curriculum are those of the high-level task. One env step is one control step of the low-level
policies (0.01 s). The wrapper runs five of them per high-level step, picks the low-level policy and encodes the
depth image; this module only configures the env that it wraps.

Observation groups (``N`` = number of envs):
    actor      (N, 40)  input of the frozen low-level policies, kept from the low-level env
    hl_actor   (N, 24)  proprioception, last action, jump status, distance to the goal
    hl_critic  (N, 42)  the hl_actor terms, then privileged ones: true velocity, height above the ground, place and
                        heading in the lane, the next obstacle and how many are left, time left, state of the jump
                        command
"""

import math

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import (CurriculumTermCfg, EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg,
                            SceneEntityCfg, TerminationTermCfg)
from mjlab.sensor import CameraSensorCfg

from beni_rl.beni_jump_env_cfg import make_beni_jump_env_cfg
from beni_rl.mdp import any_contact, delayed_bad_orientation, delayed_illegal_contact
from beni_rl_crawl.beni_robot import BENI_POSITION_OBS_JOINTS, BENI_VELOCITY_OBS_JOINTS, get_beni_robot_cfg
from beni_rl_crawl.recovery import H_REF

from .curriculum import LaneCurriculum
from .depth import DEPTH_CAMERA, DEPTH_SENSOR, DEPTH_SIZE
from .goal import MIN_AVERAGE_SPEED, TIME_BASE_S, GoalCommandCfg
from .hierarchy import PRESETTLE_STEPS
from .hl_state import HighLevelStateCfg
from .parkour_terrain import END_CLEARANCE, LANE, SPAWN_X, make_parkour_terrain_cfg
from .reward import (action_rate, at_goal, failure, goal_bonus, goal_distance, goal_reached, heading,
                     height_above_ground, in_pit, jump_started, jump_state, jump_status, last_action, lateral_offset,
                     next_obstacles, obstacles_left, out_of_lane, out_of_time, progress, stuck, time_left,
                     trigger_failed)

NUM_ENVS = 1024  # Training envs: a multiple of the 4 PPO mini-batches, 256 per lane column.
# Hard limit of an episode (s). Every lane time budget of goal.py is shorter: the farthest goal lies END_CLEARANCE
# before the lane end, which gives 63.2 s. The out_of_time termination therefore always comes before time_out, and
# the time left that the critic observes is the time the episode really has.
EPISODE_LENGTH_S = 64.0
assert TIME_BASE_S + (LANE[0] - END_CLEARANCE - SPAWN_X) / MIN_AVERAGE_SPEED <= EPISODE_LENGTH_S

# Initial Location
SPAWN_SPREAD_X = (-0.4, 0.8)
SPAWN_SPREAD_Y = 0.3

# Success: the robot has stopped upright at the goal, free of body contact (CONTACT_FORCE below). Crossing the goal at
# speed is not a success.
GOAL_TOLERANCE = 0.3  # Distance to the goal along the lane (m).
GOAL_MAX_SPEED = 0.2  # Planar speed (m/s).
GOAL_MAX_TILT_DEG = 20.  # Body tilt (deg).
# Control steps (0.5 s) for which the four conditions have to hold without a break. When a jump lands or the robot
# stumbles they hold for up to 0.23 s while the robot is being driven on at speed, which is not a stop.
GOAL_HOLD_STEPS = 50

# Failures.
FELL_TILT_DEG = 60.  # Body tilt that counts as fallen (deg) ...
FELL_STEPS = 30  # ... once it has lasted this many control steps (0.3 s): a landing tilts the body briefly.
# Root height relative to the lane ground below which the robot is in a pit (m); on a pit floor it is about -0.14 m.
PIT_HEIGHT = -0.05
# The robot has left its lane this far behind the spawn point or past the goal (m). Lanes have no end walls, and the
# vx command cannot be negative, so an overshoot cannot be undone.
LANE_BEHIND = 0.5
GOAL_OVERSHOOT = 1.0
STUCK_DISTANCE = 0.1  # A robot that advances less than this (m) ...
STUCK_WINDOW_S = 3.0  # within this time (s) is stuck: at a cube face, under a slab or at the foot of a ramp.
CONTACT_FORCE = 0.1  # Force on the base, a leg or the head that counts as body contact (N).
# Control steps (0.5 s) of body contact that end the episode. The contact sensor also fires on lane walls and obstacle
# faces, so a brief touch must not end it.
CONTACT_STEPS = 50

# Rewards given once, on the control step that ends the episode or, for TRIGGER_FAILED_PENALTY, on the control step on
# which a trigger latch runs out without its jump being accepted, and for JUMP_COST on the control step on which a
# jump begins.
SUCCESS_BONUS = 10.
FAILURE_PENALTY = 5.0
TRIGGER_FAILED_PENALTY = 1.0
# Small against SUCCESS_BONUS, so a jump the obstacle needs always pays; it makes jumping at a blind rhythm, without an
# obstacle ahead, cost something.
JUMP_COST = .1
# Most at_goal shaping reward an episode can collect, reached by a robot that ends at rest exactly on the goal,
# however it got there; a stop at the edge of the success window collects about 0.3.
AT_GOAL_REWARD = 2.0

# Range of the goal-distance observation (m). The episode ends GOAL_OVERSHOOT past the goal, and farther than 5 m
# away the actor only needs to know that the goal is not near.
GOAL_DISTANCE_CLIP = (-GOAL_OVERSHOOT, 5.0)
NEXT_OBSTACLES = 1  # Obstacles ahead of the robot that the critic observes in detail; obstacles_left counts all.
NEVER_S = (1e9, 1e9)  # Resampling time of a command that only a reset may draw (s); longer than any run.

JUMP_COOL_DOWN_STEPS = 20  # Control steps (0.2 s) after the jump reference ends; at least 1.
JUMP_RETRIGGER_HOLD_S = 0.2  # Unbroken readiness after touchdown before the next request is accepted (s).

def _set_up_scene(cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
    """Put the crawl robot, the parkour lanes and the depth camera into the scene of ``cfg``."""
    # The robot of the crawl package: on the beni_rl robot the wheel collides with the thigh, the crawl policy cannot
    # crouch below about 0.097 m, and the hollow column fails.
    cfg.scene.entities = {"robot": get_beni_robot_cfg()}
    cfg.scene.num_envs = NUM_ENVS
    terrain = cfg.scene.terrain
    terrain.terrain_type = "generator"
    terrain.terrain_generator = make_parkour_terrain_cfg()
    # Training starts every env on row 0 or row 1, half on each, and the curriculum moves it from there. Row 0 is flat
    # and its trigger is blocked, so the obstacle and jump observations never change there: had every env started on
    # it, their empirical normalisation would have a std of 0 and blow up the first obstacle to about -750 (critic
    # obstacle distance) for tens of iterations. None spreads the envs over all rows.
    terrain.max_init_terrain_level = None if play else 1
    # The camera renders geom group 0 only, which is the terrain with its lane walls. The robot's own geoms are in
    # no pixel of the image at the standing pose, so leaving them out of the render changes nothing there.
    cfg.scene.sensors += (CameraSensorCfg(
        name=DEPTH_SENSOR, camera_name=DEPTH_CAMERA, width=DEPTH_SIZE[1], height=DEPTH_SIZE[0],
        data_types=("depth",), use_textures=False, enabled_geom_groups=(0,)),)


def _interleave_columns(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
    """Startup event of the training env: env i runs lane column i % (number of columns), not a block of columns.

    mjlab gives each column a contiguous block of env ids, and rsl_rl cuts the rollout of a recurrent policy into
    mini-batches of contiguous env ids without shuffling, so every mini-batch would hold a single obstacle kind.
    Interleaved, every mini-batch holds all kinds in equal numbers.
    """
    del env_ids
    terrain = env.scene.terrain
    num_columns = len(terrain.cfg.terrain_generator.sub_terrains)
    terrain.terrain_types[:] = torch.arange(env.num_envs, device=env.device) % num_columns
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]


def _hand_commands_to_wrapper(cfg: ManagerBasedRlEnvCfg) -> None:
    """Stop the low-level command terms from choosing commands, so that the wrapper is their only writer."""
    # play=False: in play mode the jump command stays busy for 7-9 s after a jump. probability=0: it never starts a
    # jump by itself. The wrapper switches the requests on and makes them.
    jump = cfg.commands["jump"]
    jump.play, jump.probability = False, 0.
    jump.cool_down_steps, jump.retrigger_hold_s = JUMP_COOL_DOWN_STEPS, JUMP_RETRIGGER_HOLD_S

    # Twist (vx, vy, yaw rate, and the base pitch, roll and height): never resampled, no env in a special mode, every
    # range a single value. A reset then gives zero velocity at standing height and the term keeps what the wrapper
    # writes. With the stock settings a vx written from outside survived one control step in 6 of 16 envs.
    twist = cfg.commands["twist"]
    twist.resampling_time_range = NEVER_S
    twist.rel_standing_envs = twist.rel_heading_envs = twist.rel_forward_envs = 0.
    twist.heading_command = False
    twist.ranges.heading = None
    twist.ranges.lin_vel_x = twist.ranges.lin_vel_y = twist.ranges.ang_vel_z = (0., 0.)
    twist.pitch_range = twist.roll_range = (0., 0.)
    twist.height_range = (H_REF, H_REF)

    # Gimbal: the head is always commanded to look straight ahead.
    gimbal = cfg.commands["gimbal"]
    gimbal.resampling_time_range = NEVER_S
    gimbal.yaw_range = (0., 0.)
    gimbal.enable_probability = 1.


def _actor_terms() -> dict[str, ObservationTermCfg]:
    """The terms of ``hl_actor`` (24 numbers), in order; ``hl_critic`` starts with the same ones.

    The actor gets the wheel speeds (in ``joint_vel``) but not the true base velocity.
    """
    position_joints = SceneEntityCfg("robot", joint_names=BENI_POSITION_OBS_JOINTS, preserve_order=True)
    velocity_joints = SceneEntityCfg("robot", joint_names=BENI_VELOCITY_OBS_JOINTS, preserve_order=True)
    return {
        "base_ang_vel": ObservationTermCfg(func=envs_mdp.builtin_sensor, params={"sensor_name": "robot/imu_gyro"}),  # 3
        "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),  # 3
        "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel, params={"asset_cfg": position_joints}),  # 5
        "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel, params={"asset_cfg": velocity_joints}),  # 7
        "last_action": ObservationTermCfg(func=last_action),  # 2
        "jump_status": ObservationTermCfg(func=jump_status, params={"presettle_steps": PRESETTLE_STEPS}),  # 3
        "goal_distance": ObservationTermCfg(func=goal_distance, clip=GOAL_DISTANCE_CLIP),  # 1
    }


def _privileged_terms() -> dict[str, ObservationTermCfg]:
    """The terms that only ``hl_critic`` gets (18 numbers), in order: simulator knowledge the actor has to infer."""
    return {
        "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),  # 3
        "height_above_ground": ObservationTermCfg(func=height_above_ground),  # 1
        # The critic gets no depth feature (rl_cfg.py); these two give it the robot's place relative to the lane walls.
        "lateral_offset": ObservationTermCfg(func=lateral_offset),  # 1
        "heading": ObservationTermCfg(func=heading),  # 1

        # Privileged information about next obstacles
        "next_obstacles": ObservationTermCfg(func=next_obstacles, params={"count": NEXT_OBSTACLES}),  # 7
        # Number of obstacles left
        "obstacles_left": ObservationTermCfg(func=obstacles_left),  # 1
        
        "time_left": ObservationTermCfg(func=time_left),  # 1
        "jump_state": ObservationTermCfg(func=jump_state),  # 3
    }


def _rewards(step_dt: float) -> dict[str, RewardTermCfg]:
    """The reward terms of the high-level task.

    The reward manager multiplies every term by ``step_dt``, the control step (s), so a weight is a rate per second.
    The weights of the four one-shot rewards are divided by ``step_dt`` to give SUCCESS_BONUS, -FAILURE_PENALTY,
    -TRIGGER_FAILED_PENALTY and -JUMP_COST.
    """
    return {
        # +1 per metre driven towards the goal, -1 per metre driven away from it.
        "progress": RewardTermCfg(func=progress, weight=1.),
        # Potential-based: about AT_GOAL_REWARD in total for ending stopped at the goal, which makes slowing down there
        # worth discovering; about 0 for driving through it, and nothing for lingering in the goal window.
        "at_goal": RewardTermCfg(func=at_goal, weight=AT_GOAL_REWARD,
                                 params={"tolerance": GOAL_TOLERANCE, "max_speed": GOAL_MAX_SPEED}),
        "goal_reached": RewardTermCfg(func=goal_bonus, weight=SUCCESS_BONUS / step_dt),
        # For every episode end other than goal_reached, running out of time included.
        "failure": RewardTermCfg(func=failure, weight=-FAILURE_PENALTY / step_dt),
        # -1 per second while the base, a leg or the head touches the terrain.
        "body_contact": RewardTermCfg(func=any_contact, weight=-1.,
                                      params={"sensor_name": "body_ground_contact", "threshold": CONTACT_FORCE}),
        # Small: the low-level policies were trained with commands that change every 3-8 s.
        "action_rate": RewardTermCfg(func=action_rate, weight=-.02),
        # -0.1 per second until the episode ends.
        "time": RewardTermCfg(func=envs_mdp.is_alive, weight=-.1),
        # For every trigger whose 0.6 s latch runs out. A trigger accepted after a wait costs nothing, and the wrapper
        # ignores triggers on row 0, during a jump and during the jump cool-down, so those cost nothing either.
        "trigger_failed": RewardTermCfg(func=trigger_failed, weight=-TRIGGER_FAILED_PENALTY / step_dt),
        # For every jump, on the control step it begins.
        "jump": RewardTermCfg(func=jump_started, weight=-JUMP_COST / step_dt),
    }


def _terminations() -> dict[str, TerminationTermCfg]:
    """The termination terms of the high-level task.

    ``time_out`` and ``out_of_time`` are truncations, the other terms terminations. The horizon is finite, so PPO
    treats both alike, and both cost the failure penalty; the split only keeps "ran out of time" apart in the logs.
    """
    return {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "out_of_time": TerminationTermCfg(func=out_of_time, time_out=True),
        "goal_reached": TerminationTermCfg(func=goal_reached, params={
            "tolerance": GOAL_TOLERANCE, "max_speed": GOAL_MAX_SPEED, "max_tilt": math.radians(GOAL_MAX_TILT_DEG),
            "sensor_name": "body_ground_contact", "threshold": CONTACT_FORCE, "hold_steps": GOAL_HOLD_STEPS}),
        "fell": TerminationTermCfg(func=delayed_bad_orientation, params={
            "limit_angle": math.radians(FELL_TILT_DEG), "max_steps": FELL_STEPS}),
        "in_pit": TerminationTermCfg(func=in_pit, params={"height": PIT_HEIGHT}),
        "out_of_lane": TerminationTermCfg(func=out_of_lane, params={"behind": LANE_BEHIND, "beyond": GOAL_OVERSHOOT}),
        "stuck": TerminationTermCfg(func=stuck, params={"distance": STUCK_DISTANCE, "window_s": STUCK_WINDOW_S}),
        "body_contact": TerminationTermCfg(func=delayed_illegal_contact, params={
            "sensor_name": "body_ground_contact", "threshold": CONTACT_FORCE, "max_steps": CONTACT_STEPS}),
        # A world whose physics diverged to NaN (seen with the wheels pushing against a cube face): its episode ends
        # as a failure and the world is reset. Without this term the low-level "actor" observation group raises an
        # error on the NaN and stops the whole training run.
        "nan": TerminationTermCfg(func=envs_mdp.nan_detection),
    }


def make_parkour_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the env configuration of the high-level parkour task.

    ``play=False`` is the training env: NUM_ENVS envs, env i in lane column i % len(KINDS), that start on row 0 or 1
    and move between the rows with the lane curriculum. ``play=True`` has no curriculum and does not confine the envs
    to the first rows.
    """
    # The play form of the low-level jump env: no observation noise, no domain randomisation, no pushes, no fallen
    # resets and no low-level curriculum. play=False would bring all of them back.
    cfg = make_beni_jump_env_cfg(play=True)
    _set_up_scene(cfg, play)

    _hand_commands_to_wrapper(cfg)
    cfg.commands["goal"] = GoalCommandCfg()
    cfg.commands["hl_state"] = HighLevelStateCfg()

    # reset in a random position
    cfg.events["reset_base"].params["pose_range"] = {"x": SPAWN_SPREAD_X,
                                                     "y": (-SPAWN_SPREAD_Y, SPAWN_SPREAD_Y)}
    
    cfg.viewer.distance = 2.
    cfg.viewer.elevation = -45.
    cfg.viewer.max_extra_envs = 0  # Only the followed robot: envs of the same column share a lane.

    if not play:
        cfg.events["interleave_columns"] = EventTermCfg(func=_interleave_columns, mode="startup")

    # The low-level critic group is dropped: nothing reads it. The new groups have no noise, history or delay
    # (those would tick at the 100 Hz control rate); the recurrent actor and critic provide the memory.
    cfg.observations = {
        "actor": cfg.observations["actor"],
        "hl_actor": ObservationGroupCfg(terms=_actor_terms()),
        "hl_critic": ObservationGroupCfg(terms={**_actor_terms(), **_privileged_terms()}),
    }

    cfg.rewards = _rewards(step_dt=cfg.sim.mujoco.timestep * cfg.decimation)
    cfg.terminations = _terminations()
    cfg.curriculum = {} if play else {"lane_level": CurriculumTermCfg(func=LaneCurriculum)}

    # Finite horizon: running out of time ends the task, it is not an arbitrary cut of an endless one.
    cfg.episode_length_s = EPISODE_LENGTH_S
    cfg.is_finite_horizon = True
    return cfg
