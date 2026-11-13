# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""Glue between the 20 Hz high-level parkour policy and the two frozen 100 Hz low-level policies.

``ParkourVecEnvWrapper`` is the env that rsl_rl trains on. One step of it is one high-level step: it turns the
high-level action into commands, runs SUB_STEPS control steps of the wrapped env while the frozen crawl policy or
the frozen jump policy produces the joint actions, and returns the observation of the high-level policy, which
includes the encoded depth image. mjlab's train and play scripts use it once it is assigned to
``mjlab.scripts.train.RslRlVecEnvWrapper`` and ``mjlab.scripts.play.RslRlVecEnvWrapper``; no runner subclass is
needed.

High-level action (N, 3) in the units the policy outputs; ``N`` is the number of envs:
    0  forward speed   vx = VX_CENTER + VX_SCALE * action, inside VX_RANGE
    1  base height     height = H_REF + HEIGHT_SCALE * action, inside (H_LOW, H_REF)
    2  jump trigger    a value above TRIGGER_THRESHOLD asks for a jump

The trigger is a "stand up and jump" latch. The jump command of the env only accepts a request from a settled
standing stance, and it starts the jump only once the yaw-rate command has been 0 for 0.2 s, so a trigger stays
pending for up to LATCH_STEPS control steps: meanwhile the wrapper commands the standing height and a zero yaw rate.
It makes the first request PRESETTLE_STEPS control steps after the trigger, when that 0.2 s has passed, so that a
jump starts the same time after every trigger, and repeats the request on every control step until it is accepted.
A trigger is ignored on row 0 of the terrain, which is flat, while the jump command is still tracking the previous
jump, and during its cool-down.

The crawl policy drives by default, the jump policy from an accepted request until shortly after touchdown. The
robot is kept along its lane by a scripted heading hold, so the high-level policy does not steer.
"""

from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from rsl_rl.models import RNNModel
from tensordict import TensorDict

from beni_rl.beni_velocity_flat_env_cfg import ACTION_DIM, POLICY_OBSERVATION_DIM, POLICY_OBSERVATION_LAYOUT
from beni_rl_crawl.recovery import H_LOW, H_REF

from .depth import DEPTH_SENSOR, DepthEncoder

SUB_STEPS = 5  # Control steps (0.01 s) per high-level step (0.05 s).
NUM_ACTIONS = 3  # vx, base height, jump trigger.

# Scaling of the (vx, height) action. rsl_rl explores every action dim with the same std, 1 at the start, so one
# unit of action is made a sensible step of each command here: 0.4 m/s of speed, 0.02 m of height.
VX_CENTER = .8  # Speed commanded by action 0 (m/s).
VX_SCALE = .4  # m/s per unit of action.
# Speed commands (m/s). Upper bound: jumps were only probed up to 1.5 m/s. Lower bound 0: the task is to stop at the
# goal.
VX_RANGE = (0., 1.5)
# m per unit of action. Action 0 commands the standing height H_REF, because a jump is only accepted near it (when
# action 0 commanded 0.09 m, 1 of 175 random triggers was accepted). -1 commands 0.09 m, which clears the 0.21 m
# hollow; -2.16 commands H_LOW, the lowest height the crawl policy was trained on.
HEIGHT_SCALE = .02
# Bounds of the (vx, height) action: they map to VX_RANGE and to (H_LOW, H_REF).
ACTION_LOW = ((VX_RANGE[0] - VX_CENTER) / VX_SCALE, (H_LOW - H_REF) / HEIGHT_SCALE)  # (-2., -2.16)
ACTION_HIGH = ((VX_RANGE[1] - VX_CENTER) / VX_SCALE, 0.)  # (1.75, 0.)

TRIGGER_THRESHOLD = 1.  # A trigger action above this asks for a jump.
# Control steps (0.6 s) for which a trigger stays pending, the PRESETTLE_STEPS without a request included. A crouching
# or turning robot needs that long to reach the stance from which a request is accepted: under a random policy 40 of
# 107 triggers were accepted with this latch, 16 of 173 with a 0.15 s one.
LATCH_STEPS = 60
# Control steps (0.2 s, the jump command's yaw_settle_s) at the start of a latch in which no jump is requested. The
# jump command starts a jump only after 0.2 s of zero yaw-rate command, and the heading hold below commands a yaw
# rate whenever the heading is more than about 1.4 deg off, so a jump requested at once started 0 to 0.27 s after its
# trigger (late in 60% of 3,374 probe jumps) and its take-off point moved by up to 0.24 m at 1.2 m/s: a 0.30 m pit
# was crossed in 7-12 of 32 runs. With the wait 319 of 320 jumps started exactly PRESETTLE_STEPS after their trigger,
# and pits of 0.30 and 0.35 m were crossed in 64 of 64 runs (CPU, 2026-11-04). 0 requests at once, as before.
PRESETTLE_STEPS = 20

# Heading hold: the yaw-rate command that turns the robot back to the direction of its lane, world +x. Without it
# robots reached a lane wall in 13 of 52 trials.
HEADING_GAIN = 2.  # Yaw rate per heading error (1/s).
# Lane keeping: the hold steers to a heading that points back to the centre line of the lane, LATERAL_GAIN (rad) per
# metre of lateral offset and at most LATERAL_HEADING_LIMIT (rad). The heading hold alone restores the heading after
# a landing but leaves the robot where the landing put it: 0.14-0.43 m further off the centre line per jump (mean per
# env1 cell, model_2999, 2026-11-06), up to 0.8 m of the 0.95 m to the wall. With HEADING_GAIN 2, a gain of 0.5 makes
# the offset decay without overshoot at 1 m/s. 0 switches the lane keeping off.
LATERAL_GAIN = .5
LATERAL_HEADING_LIMIT = .2
YAW_LIMIT = 1.  # Largest yaw-rate command (rad/s).
# A smaller yaw-rate command is written as exactly 0 (rad/s): a jump only starts after 0.2 s of zero yaw command.
YAW_DEADBAND = .05

SETTLE_S = .4

# Device of the depth encoder; None: the device of the env. The simulation of a Mac runs on the CPU (Warp has no MPS
# device), where the encoder takes 1.9 s of the 2.5 s of a high-level step of 80 envs; on "mps" it takes 0.18 s, and
# its features differ from the CPU's by 1e-4 at most (M5, 2026-11-10). A script sets this before it builds the wrapper.
DEPTH_ENCODER_DEVICE: str | None = None

# Entries of the 40-number low-level observation that are edited so that each policy sees what it was trained on.
JUMP_COMMAND = POLICY_OBSERVATION_LAYOUT["jump_command"]  # slice(36, 40): the crawl policy was trained with zeros.
HEIGHT_COMMAND = POLICY_OBSERVATION_LAYOUT["base_pose"].stop - 1  # 30: the height of (pitch, roll, height).
# Height commands the jump policy was trained with (m). Its input normaliser has a std of 0.0058 m there, so a crouch
# command would be far outside its input range.
JUMP_HEIGHT_RANGE = (.10, .12)

CRAWL_POLICY = Path(__file__).parent / "low_level_policy" / "model_crawl.pt"
JUMP_POLICY = Path(__file__).parent / "low_level_policy" / "model_jump.pt"
# Architecture of the actor stored in both checkpoints, as arguments of rsl_rl's RNNModel.
LOW_LEVEL_MODEL = dict(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
                       rnn_type="gru", rnn_hidden_dim=512, rnn_num_layers=1)


def load_low_level_policy(path: Path, device: torch.device) -> RNNModel:
    """Load the frozen actor of the low-level checkpoint ``path``.

    ``policy(TensorDict({"actor": observation}, batch_size=[N]))`` maps the low-level observation (N, 40) to the
    deterministic joint action (N, 7) and advances the memory that the policy keeps for the N envs.
    ``policy.reset(dones)`` zeroes the memory of the envs whose entry of ``dones`` (N,) is set; ``policy.reset()``
    clears all of it.
    """
    policy = RNNModel(
        obs=TensorDict({"actor": torch.zeros(1, POLICY_OBSERVATION_DIM)}, batch_size=[1]),
        obs_groups={"actor": ["actor"]}, obs_set="actor", output_dim=ACTION_DIM,
        # The distribution only gives the std stored in the checkpoint a place to load into; the action used is the
        # mean. The dict is built on every call because rsl_rl pops "class_name" from it.
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1., "std_type": "scalar"},
        **LOW_LEVEL_MODEL)
    
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    policy.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    return policy.to(device).eval().requires_grad_(False)


def _skip_sense() -> None:
    """Stands in for ``sim.sense`` on the control steps whose depth image nobody reads."""


def _merge_logs(logs: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    """Merge the log dicts of the control steps of one high-level step.

    The env starts a new log dict on every control step and only fills it on the steps in which an env was reset.
    The result maps every key to the values that were logged for it, a float tensor (k,) with k <= SUB_STEPS;
    rsl_rl's logger concatenates these over a rollout and averages them.
    """
    values_by_key: dict[str, list[torch.Tensor]] = {}
    for log in logs:
        for key, value in log.items():
            values_by_key.setdefault(key, []).append(torch.as_tensor(value, dtype=torch.float, device=device))
    return {key: torch.stack(values) for key, values in values_by_key.items()}


class ParkourVecEnvWrapper(RslRlVecEnvWrapper):
    """The parkour env as rsl_rl sees it: 3 actions, one step = SUB_STEPS control steps of the wrapped env.

    Every control step of a high-level step runs in this order (the order is measured, see the methods):
        1. write the vx, yaw-rate and height commands into the env's twist command;
        2. request a jump for every env whose trigger has been pending for PRESETTLE_STEPS control steps;
        3. compute the low-level observation, which now holds these commands, and step both frozen policies;
        4. step the env with the action of the jump policy where a jump is in progress, of the crawl policy elsewhere.

    The state of the latch and the latest action live in the env's ``"hl_state"`` command term, because the env's
    observation and reward terms read them and the env zeroes them for an env that starts a new episode. This
    wrapper is the only writer.

    ``step``, ``reset`` and the observation builder run under ``torch.inference_mode``, also when the caller is
    not in it (the constructor, mjlab's viewer): the memory of the low-level policies and several env buffers are
    created inside rsl_rl's inference-mode rollout and cannot be edited in place outside of it.
    """

    def __init__(self, env: ManagerBasedRlEnv, clip_actions: float | None = None) -> None:
        """Wrap ``env``, built from ``make_parkour_env_cfg`` of env.py; this resets it and loads the frozen networks.

        ``clip_actions`` is accepted because mjlab's scripts pass it, and ignored: the high-level action is clamped
        per dimension in ``step``, and the low-level policies were trained with unclipped actions.
        """
        del clip_actions
        super().__init__(env, clip_actions=None)
        self.num_actions = NUM_ACTIONS  # The base class set the 7 joint actions of the wrapped env.

        commands = self.unwrapped.command_manager
        self._robot = self.unwrapped.scene["robot"]
        self._twist = commands.get_term("twist")
        self._jump = commands.get_term("jump")
        self._goal = commands.get_term("goal")
        self._hl_state = commands.get_term("hl_state")
        # Switch the jump requests on from this control step. The env builds the jump command in training mode, in
        # which it refuses every request as "disabled" until it is told that the jump curriculum has started.
        self._jump.configure_training(0, 1, jump_start_iteration=0)

        self._crawl_policy = load_low_level_policy(CRAWL_POLICY, self.device)
        self._jump_policy = load_low_level_policy(JUMP_POLICY, self.device)

        torch.backends.cudnn.allow_tf32 = False
        self._depth_device = DEPTH_ENCODER_DEVICE or self.device
        self._depth_encoder = DepthEncoder().to(self._depth_device)
        self._action_low = torch.tensor(ACTION_LOW, device=self.device)
        self._action_high = torch.tensor(ACTION_HIGH, device=self.device)

        # (N,) long, the dones of the latest step: play.py resets the memory of the high-level policy with it.
        self.last_dones = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._observation = self._build_observation()

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """(N,) long, control steps since the start of each env's episode."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        """Ignore the assignment.

        mjlab's train script always lets rsl_rl overwrite the buffer with random episode lengths at the start of a
        run. Most of them lie beyond the time budget of a lane, so those first episodes would end at once as
        "out of time" and count as failures in the curriculum.
        """

    def get_observations(self) -> TensorDict:
        """The observation after the latest ``step`` or ``reset``: hl_actor (N, 24), hl_critic (N, 42), depth (N, 440).

        It is cached, not rebuilt: building it runs the depth encoder.
        """
        return self._observation

    @torch.inference_mode()
    def reset(self) -> tuple[TensorDict, dict]:
        """Reset every env and the memory of both low-level policies; returns the observation and the env's extras."""
        _, extras = self.env.reset()
        self._crawl_policy.reset()
        self._jump_policy.reset()
        self._observation = self._build_observation()
        return self._observation, extras

    @torch.inference_mode()
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Run one high-level step with ``actions`` (N, 3): SUB_STEPS control steps of the env.

        Returns the observation (see ``get_observations``), the reward (N,) float summed over the control steps,
        the dones (N,) long, 1 where an episode ended in one of the control steps and 0 elsewhere, and
        ``{"log": ...}`` (see ``_merge_logs``). There is no ``"time_outs"`` entry: the horizon is finite.
        """
        vx, height = self._decode_action(actions)
        reward = torch.zeros(self.num_envs, device=self.device)
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        logs = []
        sim = self.unwrapped.sim
        sense = sim.sense
        try:
            for sub_step in range(SUB_STEPS):
                self._write_commands(vx, height)
                self._request_jumps()
                low_level_action = self._low_level_action()
                # The env renders the depth camera in sim.sense on every control step, but the image is only read
                # after the last one.
                sim.sense = sense if sub_step == SUB_STEPS - 1 else _skip_sense
                _, step_reward, terminated, truncated, extras = self.env.step(low_level_action)
                finished = terminated | truncated
                reward = torch.where(done, reward, reward + step_reward)
                done = done | finished

                self._crawl_policy.reset(finished)
                self._jump_policy.reset(finished)
                logs.append(extras["log"])
        finally:
            # Also after an error: the viewer's reset renders through sim.sense on the wrapped env.
            sim.sense = sense
        self.last_dones = done.long()
        self._observation = self._build_observation()
        return self._observation, reward, self.last_dones, {"log": _merge_logs(logs, self.device)}

    def _decode_action(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Record ``actions`` (N, 3) in ``hl_state``, start the latch of every new trigger, and return the commands.

        Returns the vx (m/s) and the height (m) command, (N,) each.
        """
        state = self._hl_state
        action = actions[:, :2].clamp(self._action_low, self._action_high)
        state.prev_action[:] = state.action
        state.action[:] = action

        trigger = actions[:, 2] > TRIGGER_THRESHOLD

        # Wrapper
        jump = self._jump
        # if on a lane without obstacles (row 0 of Beni-Parkour and env1); in the last jump tracking; in the cool down
        blocked = self._goal.on_flat_lane() | jump.state["tracking"] | (jump.jump_cool_down > 0)
        pressed = trigger & ~blocked

        # Press trigger to jump
        started = pressed & ~state.trigger_high & (state.pending == 0)
        state.pending[started] = LATCH_STEPS
        # Record if press
        state.trigger_high[:] = trigger

        state.metrics["triggers_started"] += started.float()
        state.metrics["triggers_ignored"] += (trigger & blocked).float()

        return VX_CENTER + VX_SCALE * action[:, 0], H_REF + HEIGHT_SCALE * action[:, 1]

    def _write_commands(self, vx: torch.Tensor, height: torch.Tensor) -> None:
        """Write the commands of this control step into the twist command: ``vx`` (m/s) and ``height`` (m), (N,) each.

        They are written on every control step because the twist command changes them inside ``env.step``: during
        a jump it pins the velocity to its value at the request, and after any jump it zeroes the yaw rate.
        """
        jump = self._jump
        pending = self._hl_state.pending > 0
        # The jump command holds the twist from an accepted request until the landing has settled. The jump policy
        # only ever saw a constant twist during a jump, so vx stays as the command pinned it.
        locked = jump.preparing | jump.command_hold
        velocity = self._twist.vel_command_b  # (N, 3): vx, vy, yaw rate.
        velocity[:, 0] = torch.where(locked, velocity[:, 0], vx)
        # Exactly 0: the low-level policies never saw another vy, so their normalisers amplify any offset 100 times.
        velocity[:, 1] = 0.
        # Exactly 0 before and during a jump: a yaw command restarts the 0.2 s the jump command waits with a
        # straight command before take-off (jumps onto a cube: 21 of 24 without steering, 0 of 24 with it).
        # After touchdown the hold steers again as soon as the crawl policy drives the robot, not only when the jump
        # command releases the twist: on a cube top the robot never counts as ready, so that release comes up to
        # 1.5 s after touchdown and the yaw that the landing left carries the robot sideways along the whole cube.
        # Steering from the hand-over halved the drift after a cube (0.21 to 0.11 m) and, on lanes of 4-6 obstacles,
        # took the falls at a cube from 8 to 1 and the episodes that reach a wall from 13 to 3 of 120 (model_2999,
        # 2026-11-10).
        landed_on_crawl = (jump.state["land_s"] >= 0) & ~self._jump_in_progress()
        steering = ~pending & (~locked | landed_on_crawl)
        velocity[:, 2] = torch.where(steering, self._heading_hold(), 0.)
        self._twist.base_pose_command[:, 2] = torch.where(pending, H_REF, height)

    def _heading_hold(self) -> torch.Tensor:
        """(N,) yaw-rate command (rad/s) that turns the robot back to the centre line of its lane and along it."""
        # Distance from the centre line (m), positive to the left: the env origin lies on the line, lanes run along +x.
        offset = self._robot.data.root_link_pos_w[:, 1] - self.unwrapped.scene.env_origins[:, 1]
        heading = (-LATERAL_GAIN * offset).clamp(-LATERAL_HEADING_LIMIT, LATERAL_HEADING_LIMIT)
        yaw_rate = (HEADING_GAIN * (heading - self._robot.data.heading_w)).clamp(-YAW_LIMIT, YAW_LIMIT)
        return torch.where(yaw_rate.abs() < YAW_DEADBAND, 0., yaw_rate)

    def _request_jumps(self) -> None:
        """Request a jump for every env whose trigger is past its pre-settle, and count the latches down.

        A latch counts down on every control step of its first PRESETTLE_STEPS, which make no request, and after that
        on every refused request. A latch whose last request is refused has failed: ``hl_state.trigger_failed`` is
        set for this control step.
        """
        state = self._hl_state
        state.trigger_failed[:] = False
        # While a trigger is pending the yaw-rate command is 0 (_write_commands), so by the first request the jump
        # command's 0.2 s yaw settle has passed and the jump starts in that control step.
        settling = state.pending > LATCH_STEPS - PRESETTLE_STEPS
        state.pending[settling] -= 1
        env_ids = ((state.pending > 0) & ~settling).nonzero().flatten()
        # The jump command returns a code per request: 0 for accepted, otherwise the reason of the refusal.
        refused = self._jump.request(env_ids) != 0
        state.pending[env_ids] = torch.where(refused, state.pending[env_ids] - 1, 0)
        failed = refused & (state.pending[env_ids] == 0)
        state.trigger_failed[env_ids] = failed
        state.metrics["jumps_accepted"][env_ids] += (~refused).float()
        state.metrics["triggers_failed"][env_ids] += failed.float()

    def _low_level_action(self) -> torch.Tensor:
        """Step both frozen policies and return, for every env, the joint action of the one that drives it; (N, 7)."""
        # Computed here, after the commands and the requests. The observation that the env returned from its last
        # step is older than both: the policies would see the commands one control step late, and the jump policy
        # would never see the first frame of a jump.
        observation = self.unwrapped.observation_manager.compute_group("actor")  # (N, 40)
        crawl_observation = observation.clone()
        crawl_observation[:, JUMP_COMMAND] = 0.
        jump_observation = observation.clone()
        jump_observation[:, HEIGHT_COMMAND].clamp_(*JUMP_HEIGHT_RANGE)
        # Both policies are stepped on every control step, so the one that takes over has a current memory. Starting
        # it from zero at the hand-over delayed the take-off and lowered the jump (peak 0.199 m against 0.300 m).
        crawl_action = self._crawl_policy(TensorDict({"actor": crawl_observation}, batch_size=[self.num_envs]))
        jump_action = self._jump_policy(TensorDict({"actor": jump_observation}, batch_size=[self.num_envs]))

        in_jump = self._jump_in_progress()
        self._hl_state.in_jump[:] = in_jump
        return torch.where(in_jump.unsqueeze(1), jump_action, crawl_action)

    def _jump_in_progress(self) -> torch.Tensor:
        """(N,) bool: the envs that the jump policy drives in this control step."""
        jump = self._jump
        tracked = jump.state  # The jump command's record of the jump it is tracking; times in s since its start.
        # The jump has started and not failed, and the robot has not touched down yet or did so under SETTLE_S ago.
        not_settled = tracked["tracking"] & tracked["gate"] & (
            (tracked["land_s"] < 0) | (tracked["age"] < tracked["land_s"] + SETTLE_S))
        # preparing: the request is accepted and the yaw is settling. jump_flag: the jump reference is playing.
        return jump.preparing | jump.jump_flag | not_settled

    @torch.inference_mode()
    def _build_observation(self) -> TensorDict:
        """The observation of the high-level policy for the current state of the env.

        Keys: hl_actor (N, 24) and hl_critic (N, 42) as the env computed them in its latest step or reset, and
        depth (N, 440), the encoded image of the front camera. Nothing else is returned: rsl_rl stores every key
        for the whole rollout.
        """
        env = self.unwrapped
        observation = env.observation_manager.compute()
        # (N, rows, columns) in metres. A view of the render buffer, so only the encoder's output is kept.
        depth = env.scene[DEPTH_SENSOR].data.depth[..., 0]
        return TensorDict({"hl_actor": observation["hl_actor"], "hl_critic": observation["hl_critic"],
                           "depth": self._depth_encoder(depth.to(self._depth_device)).to(self.device)},
                          batch_size=[self.num_envs])
