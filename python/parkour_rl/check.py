# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""Check the parkour task before training: a scripted driver runs every cell through the real wrapper.

    python -m parkour_rl.check
    python -m parkour_rl.check env1
    python -m parkour_rl.check env2

Run it once on every machine before ``parkour_rl.train``; its only option is ``env1`` or ``env2``, which checks
Beni-Parkour-Env1 or Beni-Parkour-Env2 instead of Beni-Parkour. Beni-Parkour takes about 6 minutes on a laptop CPU.
It builds the play form of the task, one robot per cell of the lane grid (36 cells, 4 x 9, for Beni-Parkour; 25,
5 x 5, for env1; 80, 16 lane columns x 5 rows, for env2),
wraps it with ``hierarchy.ParkourVecEnvWrapper`` and drives it with ``PrivilegedRule``, which reads the lane from
the simulator, until every robot has finished one episode.

It prints the termination that ended the first episode of every cell, the observation sizes, the request counts of
the jump command and the wall time per high-level step. Those numbers are a report of what the frozen low-level
policies do under a simple rule, not a pass mark. The script fails only when one of these invariants is broken:
    1. the observation has the keys and sizes of the env's observation groups, the reward is finite, the dones
       are 0 or 1, and ``sim.sense`` is back in place after every step;
    2. a commanded speed is still in the low-level observation after the step, so the env has not overwritten it;
    3. the jump policy is given the first frame of every jump;
    4. no jump request is refused as "disabled";
    5. every episode ends within the episode limit.

``N`` is the number of envs (36, 25 for env1, 80 for env2).
"""

import math
import time

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from tensordict import TensorDict

from beni_rl.beni_velocity_flat_env_cfg import POLICY_OBSERVATION_LAYOUT
from beni_rl_crawl.recovery import H_LOW, H_REF

from . import task_from_argv  # Importing the package registers the tasks.
from .depth import FEATURE_DIM
from .hierarchy import (HEIGHT_SCALE, PRESETTLE_STEPS, SUB_STEPS, TRIGGER_THRESHOLD, VX_CENTER, VX_SCALE,
                        ParkourVecEnvWrapper)

# The scripted rule. Distances are along the lane, from the robot root (m). The values are those at which the probes
# of the frozen policies worked at CRUISE_SPEED.
CRUISE_SPEED = 1.  # Speed command away from the goal (m/s).
# Speed command while a thin_cube is the next obstacle (m/s): its bars cleared best at 1.2 m/s (1.0 and 1.4 were worse).
THIN_CUBE_SPEED = 1.2
# Near the goal the speed command is BRAKE_GAIN * (distance to the goal - BRAKE_MARGIN). With it the crawl policy
# stopped inside the goal window in 8 of 8 runs.
BRAKE_GAIN = 2.  # 1/s
BRAKE_MARGIN = .1  # m
# A jump is triggered when the front edge of the obstacle is closer than this (m). The wrapper makes its first request
# PRESETTLE_STEPS control steps after the trigger, and the jump starts then. For LEAD_KINDS the value is the distance
# at which the jump should start, and the trigger leads it by the distance the commanded speed covers in
# PRESETTLE_STEPS. With these values (CPU, 2026-11-04, 32 runs each) pits of 0.30 and 0.35 m were crossed 32 of 32
# times at 1.0 m/s and thin cubes of 0.06-0.12 m cleared 31-32 of 32 times at 1.2 m/s. For a cube the value is the
# trigger itself: at 1.0 m/s the jump then starts about 0.37 m before the face (0.10 m cube passed 28 of 32, 0.15 m
# 14 of 32, as many as before the pre-settle).
TRIGGER_DISTANCE = {"pit": .25, "cube": .6, "thin_cube": .45}
LEAD_KINDS = ("pit", "thin_cube")
CROUCH_DISTANCE = .8  # The base is lowered when a hollow is closer than this (m): crouching takes up to 0.5 s.
# High-level steps (0.5 s) for which the crouch is kept after the root has left a slab: the rear of the robot is
# still under it.
CROUCH_HOLD_STEPS = 10

# Entries of the 40-number low-level observation that the invariants read.
VX_COMMAND = POLICY_OBSERVATION_LAYOUT["velocity_commands"].start  # 6
JUMP_FLAG = POLICY_OBSERVATION_LAYOUT["jump_command"].start  # 36: 1 while the jump reference plays.
JUMP_PHASE = JUMP_FLAG + 1  # 37: progress through the jump reference, 0 on its first frame.
VX_TOLERANCE = 1e-5  # Round-off of the action scaling (m/s).


class PrivilegedRule:
    """The scripted driver: cruise, crouch under a hollow, jump a pit, a cube or a thin_cube, brake at the goal.

    It is privileged: it reads the next obstacle and the distance to the goal from the env's ``"goal"`` command
    term, which the high-level policy has to infer from the depth image.
    """

    def __init__(self, env: ManagerBasedRlEnv) -> None:
        self._goal = env.command_manager.get_term("goal")
        kinds = self._goal.kinds
        self._num_kinds = len(kinds)
        # (len(kinds),) trigger distance of each obstacle kind (m); 0 for the kinds that are not jumped.
        self._trigger_distance = torch.tensor([TRIGGER_DISTANCE.get(kind, 0.) for kind in kinds], device=env.device)
        # (len(kinds),) 1 for the kinds whose trigger leads the start of the jump (LEAD_KINDS), 0 for the others.
        self._leads = torch.tensor([float(kind in LEAD_KINDS) for kind in kinds], device=env.device)
        self._presettle_s = PRESETTLE_STEPS * env.step_dt  # Time from a trigger to the start of its jump (s).
        # Positions of the hollow and the thin_cube in the one-hot kind of an obstacle; None if no lane has a thin_cube.
        self._hollow = kinds.index("hollow")
        self._thin_cube = kinds.index("thin_cube") if "thin_cube" in kinds else None
        # (N,) high-level steps for which a crouch is still kept. Not cleared when an episode ends: only the first
        # episode of an env is reported.
        self._crouch_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def command(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Commands of the next high-level step: vx (N,) in m/s, base height (N,) in m, jump trigger (N,) bool."""
        # The first obstacle that the root has not passed: its one-hot kind (all zeros if there is none) and the
        # distance to its front edge, which is negative while the root is inside or under the obstacle.
        obstacle = self._goal.next_obstacles(1)[:, 0]  # (N, len(kinds) + 3)
        kind, front = obstacle[:, :self._num_kinds], obstacle[:, self._num_kinds]

        cruise = torch.full_like(front, CRUISE_SPEED)
        if self._thin_cube is not None:
            cruise = torch.where(kind[:, self._thin_cube].bool(), THIN_CUBE_SPEED, cruise)
        vx = torch.minimum((BRAKE_GAIN * (self._goal.distance() - BRAKE_MARGIN)).clamp(min=0.), cruise)

        hollow_near = kind[:, self._hollow].bool() & (front < CROUCH_DISTANCE)
        crouch = hollow_near | (self._crouch_steps > 0)
        self._crouch_steps = torch.where(hollow_near, CROUCH_HOLD_STEPS, self._crouch_steps - 1).clamp(min=0)
        height = torch.where(crouch, H_LOW, H_REF)

        # Only while the obstacle is still ahead. The trigger stays set over several steps; the wrapper starts one
        # latch on its rising edge and ignores it while the jump command tracks a jump, but a trigger still set when
        # the jump command lets go would ask for another jump. For LEAD_KINDS it comes earlier by the distance that vx
        # covers in the pre-settle, so that the jump starts at the trigger distance.
        lead = vx * self._presettle_s * (kind @ self._leads)
        trigger = (front > 0.) & (front - lead < kind @ self._trigger_distance)
        return vx, height, trigger


def to_action(vx: torch.Tensor, height: torch.Tensor, trigger: torch.Tensor) -> torch.Tensor:
    """The high-level action (N, 3) with which the wrapper commands ``vx`` (m/s), ``height`` (m) and ``trigger``.

    The inverse of the action scaling in ``ParkourVecEnvWrapper``.
    """
    return torch.stack(((vx - VX_CENTER) / VX_SCALE, (height - H_REF) / HEIGHT_SCALE,
                        torch.where(trigger, TRIGGER_THRESHOLD + 1., -1.)), dim=1)


def record_first_outcomes(env: ManagerBasedRlEnv) -> list[str | None]:
    """Record which termination ends the first episode of every env; returns the list that is filled in.

    Entry i is None while the first episode of env i runs, then the name of the termination term that ended it
    (names joined by "+" if several fired on that control step). The wrapper only reports that an episode ended
    within a high-level step, so the termination manager is wrapped here, the way
    ``beni_rl.jump.install_observer`` wraps the reward manager.
    """
    manager = env.termination_manager
    compute = manager.compute
    outcomes: list[str | None] = [None] * env.num_envs

    def recording_compute() -> torch.Tensor:
        finished = compute()
        for env_id in finished.nonzero().flatten().tolist():
            if outcomes[env_id] is None:
                outcomes[env_id] = "+".join(name for name in manager.active_terms if manager.get_term(name)[env_id])
        return finished

    manager.compute = recording_compute
    return outcomes


class FirstJumpFrames:
    """Forward pre-hook of the frozen jump policy: asserts invariant 3 on every control step and counts its frames.

    A jump starts when the jump command sets its flag and puts its reference on frame 0. The jump policy has to be
    given that frame, flag 1 and phase 0, on the same control step. That only happens if the wrapper computes the
    low-level observation after it has requested the jump: the observation that the env returned from its last
    step is older. The check is made on the policy's input because a freshly computed observation always shows the
    frame; the question is whether the policy gets it.
    """

    def __init__(self, wrapper: ParkourVecEnvWrapper) -> None:
        self._jump = wrapper.unwrapped.command_manager.get_term("jump")
        self.seen = 0  # First frames of a jump that the policy was given.
        # The policy is a private member of the wrapper; its input cannot be seen from anywhere else.
        wrapper._jump_policy.register_forward_pre_hook(self)

    def __call__(self, policy: torch.nn.Module, inputs: tuple) -> None:
        starting = self._jump.jump_flag & (self._jump.jump_count == 0)  # (N,) bool
        observation = inputs[0]["actor"][starting]  # (k, 40), the rows of the envs whose jump starts.
        assert (observation[:, JUMP_FLAG] == 1).all() and (observation[:, JUMP_PHASE] == 0).all(), (
            "invariant 3: the jump policy was not given the first frame of a jump")
        self.seen += int(starting.sum())


def check_observation(observation: TensorDict, num_envs: int, dims: dict[str, int]) -> None:
    """Assert that ``observation`` has exactly the keys of ``dims`` and the shape (num_envs, dims[key]) under each."""
    shapes = {key: tuple(value.shape) for key, value in observation.items()}
    assert shapes == {key: (num_envs, dim) for key, dim in dims.items()}, (
        f"invariant 1: observation shapes {shapes}")


@torch.inference_mode()
def drive(wrapper: ParkourVecEnvWrapper, outcomes: list[str | None], dims: dict[str, int]) -> float:
    """Drive every env with the scripted rule until ``outcomes`` is complete.

    Invariants 1 and 2 are asserted after every high-level step; ``dims`` holds the size of every observation group.
    Returns the wall time per high-level step (s). The loop runs in inference mode, as rsl_rl's rollouts do.
    """
    env = wrapper.unwrapped
    jump = env.command_manager.get_term("jump")
    rule = PrivilegedRule(env)
    sense = env.sim.sense
    # No episode outlasts max_episode_length control steps, so this many high-level steps end every first episode.
    max_steps = math.ceil(env.max_episode_length / SUB_STEPS)
    print(f"Driving {env.num_envs} envs with the scripted rule for at most {max_steps} high-level steps ...",
          flush=True)

    check_observation(wrapper.get_observations(), env.num_envs, dims)
    start = time.perf_counter()
    for step in range(1, max_steps + 1):
        vx, height, trigger = rule.command()
        tracked_before = jump.state["tracking"].clone()
        observation, reward, dones, _ = wrapper.step(to_action(vx, height, trigger))

        check_observation(observation, env.num_envs, dims)
        assert reward.shape == (env.num_envs,) and reward.isfinite().all(), f"invariant 1, step {step}: {reward}"
        assert ((dones == 0) | (dones == 1)).all(), f"invariant 1, step {step}: dones {dones}"
        # Compared with ==, not with is: reading a method from its object makes a new bound-method object each time.
        assert env.sim.sense == sense, f"invariant 1, step {step}: sim.sense was not put back"

        # Invariant 2. Two kinds of envs are left out. One that started a new episode: the reset zeroed its speed
        # command. One whose jump the jump command tracked at the start or at the end of the step: from an accepted
        # request until the landing has settled, that command holds the speed at its value from the request.
        free = (dones == 0) & ~tracked_before & ~jump.state["tracking"]
        low_level_vx = env.observation_manager.compute_group("actor")[:, VX_COMMAND]
        assert torch.allclose(low_level_vx[free], vx[free], atol=VX_TOLERANCE), (
            f"invariant 2, step {step}: commanded {vx[free]}, in the low-level observation {low_level_vx[free]}")

        if None not in outcomes:
            break
    return (time.perf_counter() - start) / step


def print_report(outcomes: list[str], kinds: tuple[str, ...], num_rows: int, dims: dict[str, int],
                 counts: dict[str, torch.Tensor], first_frames: int, seconds_per_step: float) -> None:
    """Print the outcome of every cell (rows down, obstacle kinds across), the jump statistics and the step time.

    The observation sizes ``dims`` follow the outcomes.
    """
    # Env i sits on column i // num_rows, the obstacle kind, and on row i % num_rows.
    table = [["row", *kinds]] + [[str(row)] + [outcomes[column * num_rows + row] for column in range(len(kinds))]
                                 for row in range(num_rows)]
    width = max(map(len, (*kinds, *outcomes)))
    print("\nTermination that ended the first episode of each cell:")
    for label, *cells in table:
        print(label.ljust(5) + "  ".join(cell.ljust(width) for cell in cells).rstrip())
    reached = sum("goal_reached" in outcome for outcome in outcomes)
    print(f"\ngoal_reached in {reached} of {len(outcomes)} cells")
    print("observation sizes:", ", ".join(f"{key} {dim}" for key, dim in dims.items()))
    # An env that finishes early keeps driving new episodes until the last one is done, and these count as well.
    print("jump requests over the whole run:", {key: int(value) for key, value in counts.items()})
    print(f"first frames of a jump given to the jump policy: {first_frames}")
    print(f"wall time per high-level step: {seconds_per_step:.3f} s")


def main() -> None:
    """Build and wrap the play env, drive it, print the report and assert the invariants."""
    task_id = task_from_argv()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ManagerBasedRlEnv(load_env_cfg(task_id, play=True), device=device)
    wrapper = ParkourVecEnvWrapper(env)
    jump = env.command_manager.get_term("jump")
    outcomes = record_first_outcomes(env)
    first_frames = FirstJumpFrames(wrapper)
    # Sizes of the observation that the wrapper returns, from the env's observation groups; rl_cfg.py feeds them to
    # the actor and the critic.
    groups = env.observation_manager.group_obs_dim
    dims = {"hl_actor": groups["hl_actor"][0], "hl_critic": groups["hl_critic"][0], "depth": FEATURE_DIM}

    seconds_per_step = drive(wrapper, outcomes, dims)

    assert None not in outcomes, "invariant 5: an episode outlasted the episode limit"
    # The report comes before the last two invariants: its request counts show why one of them fails.
    goal = env.command_manager.get_term("goal")
    generator = env.scene.terrain.cfg.terrain_generator
    # The columns of the report: one obstacle kind each, or, where the lanes mix kinds (env2), the lane columns.
    columns = goal.kinds if goal.column_kind is not None else tuple(generator.sub_terrains)
    print_report(outcomes, columns, generator.num_rows, dims, jump.counts, first_frames.seen, seconds_per_step)
    assert jump.counts["rejected_disabled"] == 0, "invariant 4: a jump request was refused as disabled"
    assert first_frames.seen > 0, "invariant 3 was not exercised: no jump started"
    print("All invariants hold.")
    wrapper.close()


if __name__ == "__main__":
    main()
