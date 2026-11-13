# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from beni_rl.jump import THRESHOLDS
from beni_rl.mdp import any_contact, base_height, base_tilt
from beni_rl_crawl.recovery import H_LOW, H_REF

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers import RewardTermCfg, TerminationTermCfg

    from beni_rl.jump import BeniJumpCommand

    from .goal import GoalCommand
    from .hl_state import HighLevelState


# The command terms that this module reads.


def _goal(env: ManagerBasedRlEnv) -> GoalCommand:
    return env.command_manager.get_term("goal")


def _hl_state(env: ManagerBasedRlEnv) -> HighLevelState:
    return env.command_manager.get_term("hl_state")


def _jump(env: ManagerBasedRlEnv) -> BeniJumpCommand:
    return env.command_manager.get_term("jump")


def _readiness_steps(env: ManagerBasedRlEnv) -> int:
    """Control steps of unbroken readiness that the jump command requires before it accepts a request (20)."""
    return round(THRESHOLDS["readiness_s"] / env.step_dt)


# Observation terms: float (N, d).


def last_action(env: ManagerBasedRlEnv) -> torch.Tensor:
    """(N, 2) the normalised (vx, height) action of the current high-level step."""
    return _hl_state(env).action


def jump_status(env: ManagerBasedRlEnv, presettle_steps: int) -> torch.Tensor:
    """
    (N, 3) flags as 0 or 1: the jump policy is driving, a trigger is pending, a trigger now starts a jump on time if
    the robot stays ready and straight through the pre-settle.
    """
    state, jump = _hl_state(env), _jump(env)
    # The conditions under which BeniJumpCommand.request accepts a request and begins the jump at once, counted
    # presettle_steps control steps ahead: both step counts keep growing while the robot stays ready and straight.
    yaw_settle_steps = round(jump.cfg.yaw_settle_s / env.step_dt)
    ready = ((jump.ready_steps > 0) & (jump.ready_steps + presettle_steps >= _readiness_steps(env))
             & (jump.yaw_ready_steps + presettle_steps >= yaw_settle_steps)
             & (jump.jump_cool_down == 0) & ~jump.state["tracking"])
    if presettle_steps > 0:
        # jump.py restarts the yaw settle on every step the robot turns faster than this, also during the pre-settle.
        wz = env.scene["robot"].data.root_link_ang_vel_w[:, 2]
        ready &= wz.abs() <= jump.cfg.takeoff_yaw_rate
    return torch.stack((state.in_jump, state.pending > 0, ready), dim=1).float()


def goal_distance(env: ManagerBasedRlEnv) -> torch.Tensor:
    """(N, 1) distance left to the goal along the lane (m); negative past the goal."""
    return _goal(env).distance().unsqueeze(1)


def height_above_ground(env: ManagerBasedRlEnv) -> torch.Tensor:
    """(N, 1) height of the robot root above the terrain under it (m): the ground, a pit floor, a cube or a ramp."""
    return base_height(env) - _goal(env).ground_height().unsqueeze(1)


def lateral_offset(env: ManagerBasedRlEnv) -> torch.Tensor:
    """(N, 1) distance of the robot root from the centre line of its lane (m), positive to the left.

    The env origin lies on the centre line (parkour_terrain.py), and lanes run along world +x.
    """
    robot = env.scene["robot"]
    return (robot.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]).unsqueeze(1)


def heading(env: ManagerBasedRlEnv) -> torch.Tensor:
    """(N, 1) yaw of the robot relative to the direction of its lane, world +x (rad), in [-pi, pi]."""
    return env.scene["robot"].data.heading_w.unsqueeze(1)


def next_obstacles(env: ManagerBasedRlEnv, count: int) -> torch.Tensor:
    """(N, (K + 3) * count) the next ``count`` obstacles of the lane that the robot has not passed, K + 3 numbers each.

    Per obstacle: one-hot kind over the goal term's kinds (K: 4 for Beni-Parkour, 5 for env1), distance to its start
    and to its end (m), value / limit of its kind.
    """
    return _goal(env).next_obstacles(count).flatten(1)


def obstacles_left(env: ManagerBasedRlEnv) -> torch.Tensor:
    """(N, 1) number of obstacles of the lane that the robot has not passed, the next one included."""
    return _goal(env).obstacles_left().unsqueeze(1)


def time_left(env: ManagerBasedRlEnv) -> torch.Tensor:
    """(N, 1) fraction of the episode's time budget that is left, in [0, 1]."""
    return _goal(env).time_fraction_left().unsqueeze(1)


def jump_state(env: ManagerBasedRlEnv) -> torch.Tensor:
    """(N, 3) state of the jump command, each entry in [0, 1].

    The entries: a jump is being tracked (from the accepted request until the command lets go of it), the fraction
    of the cool-down that is left, and the fraction of the required readiness that has built up.
    """
    jump = _jump(env)
    need = _readiness_steps(env)
    return torch.stack((jump.state["tracking"].float(),
                        jump.jump_cool_down / max(jump.cfg.cool_down_steps, 1),
                        jump.ready_steps.clamp(max=need) / need), dim=1)


# Reward terms: float (N,).


def progress(env: ManagerBasedRlEnv) -> torch.Tensor:
    """(N,) speed towards the goal along the lane (m/s): positive while approaching it, negative while moving away."""
    velocity_x = env.scene["robot"].data.root_link_lin_vel_w[:, 0]
    return torch.sign(_goal(env).distance()) * velocity_x


class at_goal:
    """Reward: potential-based shaping towards a stop at the goal.

    The potential Phi = exp(-(d / tolerance)^2) * exp(-(v / max_speed)^2), d the distance to the goal along the lane
    (m), v the planar root speed (m/s), is near 1 for a robot stopped at the goal and near 0 elsewhere. The term
    returns its change per second, (Phi_t - Phi_{t-1}) / step_dt, which the reward manager multiplies by step_dt and
    the weight. Over an episode the term therefore adds up to weight * (Phi at the end - Phi at the start), whatever
    the path: about the weight for a robot that ends stopped at the goal, about 0 for one that passes it at speed,
    and nothing extra for dawdling inside the goal window.

    The discount factor is left out of the difference (gamma is 0.995 per high-level step, about 0.999 per control
    step), and Phi is not zeroed at termination, so a success earns about the weight on top of the success bonus.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        del cfg
        self._previous = torch.zeros(env.num_envs, device=env.device)  # (N,) Phi of the previous control step.
        # (N,) bool: the next call is the first of an episode; there is no previous Phi to compare with.
        self._first = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(self, env: ManagerBasedRlEnv, tolerance: float, max_speed: float) -> torch.Tensor:
        """(N,) change of Phi since the previous control step, per second; 0 on the first step of an episode."""
        speed = env.scene["robot"].data.root_link_lin_vel_w[:, :2].norm(dim=1)
        potential = torch.exp(-(_goal(env).distance() / tolerance).square() - (speed / max_speed).square())
        previous = torch.where(self._first, potential, self._previous)
        self._previous[:] = potential
        self._first[:] = False
        return (potential - previous) / env.step_dt

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Start a new episode for ``env_ids``."""
        # Phi is not computed here: when mjlab calls this, the robot data still holds the pose that ended the previous
        # episode. The first call of the episode takes its own Phi as the previous one instead.
        self._first[env_ids] = True


def goal_bonus(env: ManagerBasedRlEnv) -> torch.Tensor:
    """(N,) 1 on the control step on which the episode ends with the termination ``goal_reached``."""
    return env.termination_manager.get_term("goal_reached").float()


def failure(env: ManagerBasedRlEnv) -> torch.Tensor:
    """(N,) 1 on the control step on which the episode ends any other way than with ``goal_reached``.

    Running out of time counts as a failure too: otherwise creeping towards an obstacle until the time budget is used
    up pays better than attempting it while jumps still fail often.
    """
    terminations = env.termination_manager
    return (terminations.dones & ~terminations.get_term("goal_reached")).float()


def action_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
    """(N,) squared change of the normalised (vx, height) action from the previous high-level step to this one."""
    state = _hl_state(env)
    return (state.action - state.prev_action).square().sum(1)


def trigger_failed(env: ManagerBasedRlEnv) -> torch.Tensor:
    """(N,) 1 on the control step on which a trigger latch runs out because the jump command refused every request.

    A trigger accepted after a wait and a latch cleared by an episode reset give nothing.
    """
    return _hl_state(env).trigger_failed.float()


def jump_started(env: ManagerBasedRlEnv) -> torch.Tensor:
    """(N,) 1 on the control step on which a jump begins: its reference is on the first frame.

    The reward manager runs before the command manager advances the reference, so this is 1 exactly once per jump.
    """
    jump = _jump(env)
    return (jump.jump_flag & (jump.jump_count == 0)).float()


def crouch_off_hollow(env: ManagerBasedRlEnv, ahead: float, behind: float) -> torch.Tensor:
    """(N,) how far the height command is below the standing height, 0 at H_REF to 1 at H_LOW, away from a hollow.

    It is 0 while the root is from ``ahead`` (m) before the slab of a hollow to ``behind`` (m) past it. The wrapper
    writes the height command into the twist command before every control step, and the rewards are computed before
    the command terms run, so this is the height the robot is being told.
    """
    command = env.command_manager.get_term("twist").base_pose_command[:, 2]
    crouch = ((H_REF - command) / (H_REF - H_LOW)).clamp(0., 1.)
    return torch.where(_goal(env).near("hollow", ahead, behind), 0., crouch)


# Termination terms: bool (N,). env.step computes them before the rewards, so goal_bonus and failure above read the
# flags of the same control step.


class goal_reached:
    """Termination: success. The robot has been at the goal, slow, upright and free for ``hold_steps`` control steps.

    At the goal: within ``tolerance`` (m) of it along the lane, or, with ``tolerance_past``, from ``tolerance`` before
    it to ``tolerance_past`` (m) past it. Slow: planar speed below ``max_speed`` (m/s).
    Upright: body tilt below ``max_tilt`` (rad). Free: the sensor ``sensor_name`` reports no body contact above
    ``threshold`` (N); a robot jammed against a lane wall inside the goal window made 7 of 8 random-policy successes.
    Driving through the goal at speed is not a success, and the four conditions have to hold without a break: the
    speed also drops below ``max_speed`` for a moment when a jump lands or the robot stumbles. Under a random policy,
    3 of the 4 episodes that met the first three conditions on a single control step were being driven on at
    0.5-1.4 m/s.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
        del cfg
        # (N,) control steps for which the four conditions have held without a break.
        self._steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def __call__(self, env: ManagerBasedRlEnv, tolerance: float, max_speed: float, max_tilt: float,
                 sensor_name: str, threshold: float, hold_steps: int,
                 tolerance_past: float | None = None) -> torch.Tensor:
        """(N,) bool: the conditions have held for ``hold_steps`` control steps in a row."""
        speed = env.scene["robot"].data.root_link_lin_vel_w[:, :2].norm(dim=1)
        distance = _goal(env).distance()  # Negative past the goal.
        at_goal = (distance < tolerance) & (distance > -(tolerance if tolerance_past is None else tolerance_past))
        stopped = (at_goal & (speed < max_speed) & (base_tilt(env) < max_tilt)
                   & (any_contact(env, sensor_name, threshold) == 0.))
        self._steps[stopped] += 1
        self._steps[~stopped] = 0
        return self._steps >= hold_steps

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Start a new episode for ``env_ids``."""
        self._steps[env_ids] = 0


def out_of_time(env: ManagerBasedRlEnv) -> torch.Tensor:
    """(N,) the time budget of the episode is used up."""
    return _goal(env).time_fraction_left() <= 0.


def in_pit(env: ManagerBasedRlEnv, height: float) -> torch.Tensor:
    """(N,) the robot root is below ``height`` (m, relative to the lane ground; negative), so it fell into a pit."""
    return base_height(env).squeeze(1) < height


def out_of_lane(env: ManagerBasedRlEnv, behind: float, beyond: float) -> torch.Tensor:
    """(N,) the robot is more than ``behind`` (m) behind the spawn point or more than ``beyond`` (m) past the goal."""
    goal = _goal(env)
    x = goal.position()
    return (x < goal.spawn_x - behind) | (x > goal.goal_x + beyond)


class stuck:
    """Termination: the robot has not advanced ``distance`` (m) along the lane within ``window_s`` (s).

    A blocked robot stays upright (at a cube face, under a slab, at the foot of a ramp), so no other termination
    ends its episode. The term keeps one mark per env: the position and the control step of its last advance. When
    the robot is more than ``distance`` past the mark, the mark moves to the robot; the term fires once the mark is
    older than ``window_s``.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
        del cfg
        self._env = env
        self._mark_x = torch.zeros(env.num_envs, device=env.device)  # Lane frame (m).
        self._mark_step = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)  # Step of the episode.

    def __call__(self, env: ManagerBasedRlEnv, distance: float, window_s: float) -> torch.Tensor:
        """(N,) bool: the mark of the env is older than ``window_s``."""
        x = _goal(env).position()
        advanced = x > self._mark_x + distance
        self._mark_x[advanced] = x[advanced]
        self._mark_step[advanced] = env.episode_length_buf[advanced]
        return (env.episode_length_buf - self._mark_step) * env.step_dt > window_s

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Start a new episode for ``env_ids``: the mark goes back to the spawn point and step 0."""
        # The spawn is read from the goal term, not from the robot: when mjlab calls this, the robot data still holds
        # the pose that ended the previous episode. Without a spawn event it is 0, the env origin.
        self._mark_x[env_ids] = _goal(self._env).spawn_x[env_ids]
        self._mark_step[env_ids] = 0
