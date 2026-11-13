# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.terrains import TerrainGeneratorCfg

from .env import GOAL_OVERSHOOT, make_parkour_env_cfg
from .goal import MIN_AVERAGE_SPEED, TIME_BASE_S, GoalCommandCfg
from .parkour_terrain import GOAL_MARGIN, RUN_UP, ParkourLaneCfg, obstacle_length
from .reward import crouch_off_hollow, goal_distance
from .rl_cfg import parkour_ppo_runner_cfg

ENV1_KINDS = ("pit", "cube", "hollow", "ramp_cube", "thin_cube")
ENV1_NUM_ROWS = 5
ENV1_LANE = (12.0, 2.0)  # Length along x, width along y (m).

ENV1_ROWS = {
    "pit": ((0.15,), (0.25,), (0.30,), (0.35,)),
    "cube": ((0.06,), (0.08,), (0.10,), (0.12,)),
    "hollow": ((0.26,), (0.23,), (0.21,), (0.20,)),
    "ramp_cube": ((8.0,), (12.0,), (16.0,), (20.0,)),
    "thin_cube": ((0.06,), (0.08,), (0.10,), (0.12,)),
}
ENV1_NUM_ENVS = 1280  # Training envs: 256 per lane column, a multiple of the 4 PPO mini-batches.
ENV1_EPISODE_LENGTH_S = 22
# The farthest goal (m): GOAL_MARGIN[1] past the longest obstacle (the 8 deg ramp_cube, 2.07 m) after the longest
# run-up.
FARTHEST_GOAL_X = RUN_UP[1] + max(obstacle_length(kind, value) for kind, rows in ENV1_ROWS.items() for row in rows
                                  for value in row) + GOAL_MARGIN[1]
# Every time budget is shorter than the episode: at most 4 + (9.07 + 0.5) / 0.6 = 19.9 s, from the nearest spawn.
assert TIME_BASE_S + (FARTHEST_GOAL_X - GoalCommandCfg.spawn_min_x) / MIN_AVERAGE_SPEED < ENV1_EPISODE_LENGTH_S
# Range of the actor's goal distance (m). GOAL_MARGIN[0] is the nearest goal past the obstacle, so the value stays
# constant until the robot has passed it. The critic also gets the distance clipped only at the lane length.
ENV1_GOAL_DISTANCE_CLIP = (-GOAL_OVERSHOOT, GOAL_MARGIN[0])
ENV1_GOAL_DISTANCE_FULL_CLIP = (-GOAL_OVERSHOOT, ENV1_LANE[0])
# Crouching costs CROUCH_COST per second at the lowest height command, except from CROUCH_AHEAD (m) before the slab of a
# hollow to CROUCH_BEHIND (m) past it.
CROUCH_COST = 0.3
CROUCH_AHEAD, CROUCH_BEHIND = 1.0, 0.3


def make_env1_terrain_cfg(seed: int = 0) -> TerrainGeneratorCfg:
    """The env1 lanes: ENV1_NUM_ROWS rows and one column per kind of ENV1_KINDS, laid out by ENV1_ROWS."""
    return TerrainGeneratorCfg(size=ENV1_LANE, num_rows=ENV1_NUM_ROWS, curriculum=True, seed=seed,
                               sub_terrains={kind: ParkourLaneCfg(kind=kind, num_rows=ENV1_NUM_ROWS, rows=ENV1_ROWS)
                                             for kind in ENV1_KINDS})


def _spawn_along_lane(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
    """Reset event, run right after reset_base: move every robot of ``env_ids`` along its lane to a new random spawn.

    The goal term draws the spawn x and keeps it for the goal, the time budget and the terminations. reset_base has
    just written the root pose into qpos, but the body positions (root_link_pos_w) hold the pose that ended the
    previous episode until the reset's sim.forward, so the pose is read back from qpos. Lanes run along world +x.
    """
    env_ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    robot = env.scene["robot"]
    pose = robot.data.data.qpos[env_ids[:, None], robot.data.indexing.free_joint_q_adr]  # (n, 7): position, quat.
    pose[:, 0] += env.command_manager.get_term("goal").sample_spawn_x(env_ids)
    robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)


def make_env1_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the env configuration of env1: ``make_parkour_env_cfg(play)`` with the changes of the module docstring.

    ``play=False`` is the training env: ENV1_NUM_ENVS envs, env i in lane column i % 5, that start on row 0 or 1 and
    move between the rows with the lane curriculum. ``play=True`` has no curriculum and does not confine the envs to
    the first rows.
    """
    cfg = make_parkour_env_cfg(play)
    cfg.scene.num_envs = ENV1_NUM_ENVS
    # max_init_terrain_level stays as env.py set it: rows 0-1 in training, all rows in play.
    cfg.scene.terrain.terrain_generator = make_env1_terrain_cfg()
    cfg.episode_length_s = ENV1_EPISODE_LENGTH_S

    cfg.observations["hl_actor"].terms["goal_distance"].clip = ENV1_GOAL_DISTANCE_CLIP
    critic = cfg.observations["hl_critic"].terms
    critic["goal_distance"].clip = ENV1_GOAL_DISTANCE_CLIP
    critic["goal_distance_full"] = ObservationTermCfg(func=goal_distance, clip=ENV1_GOAL_DISTANCE_FULL_CLIP)  # 1

    # reset_base keeps its y spread and zero yaw and puts x on the env origin; the spawn event then moves x along the
    # lane. Reset events run in the order of cfg.events, so the dict is rebuilt with the new one right after reset_base.
    cfg.events["reset_base"].params["pose_range"]["x"] = (0., 0.)
    events = {}
    for name, term in cfg.events.items():
        events[name] = term
        if name == "reset_base":
            events["spawn_along_lane"] = EventTermCfg(func=_spawn_along_lane, mode="reset")
    cfg.events = events

    cfg.rewards["crouch"] = RewardTermCfg(func=crouch_off_hollow, weight=-CROUCH_COST,
                                          params={"ahead": CROUCH_AHEAD, "behind": CROUCH_BEHIND})
    if not play:
        # One obstacle per lane: a failure is early (demotion) when the robot has not passed it (curriculum.py).
        cfg.curriculum["lane_level"].params["before_obstacle"] = True
    return cfg


def env1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """The PPO configuration of Beni-Parkour (rl_cfg.py) with its own log directory, beni_parkour_env1.

    The network sizes follow from the observations: actor 24 + 440 inputs, critic 44.
    """
    cfg = parkour_ppo_runner_cfg()
    cfg.experiment_name = "beni_parkour_env1"
    return cfg
