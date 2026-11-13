# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""Per-env state of the high-level parkour controller, kept in the env as the ``"hl_state"`` command term.

The state belongs to the wrapper that runs the high-level policy, but observation and reward functions read it
inside ``env.step``, and once when the observation manager is built, before any wrapper exists. A command term is
built before the observation manager and mjlab resets it for every env that starts a new episode, so the state is
stored here. This module only holds the buffers: scaling, clamping and the trigger latch are in the wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg


class HighLevelState(CommandTerm):
    """Buffers written by ``hierarchy.ParkourVecEnvWrapper`` and read by the functions of ``reward.py``.

    ``N`` is the number of envs. One high-level step (20 Hz) is five control steps (100 Hz).

    Written once per high-level step:
        action        (N, 2) float  clamped, normalised (vx, height) action of the current high-level step
        prev_action   (N, 2) float  the same for the previous high-level step
        trigger_high  (N,) bool     the trigger was pressed in the previous high-level step: above the threshold
                                    and not blocked (see ``ParkourVecEnvWrapper._decode_action``)

    Written every control step:
        pending        (N,) long    control steps left in the trigger latch; 0 = no trigger pending
        trigger_failed (N,) bool    a latch ran out in this control step: its last request was refused too
        in_jump        (N,) bool    the frozen jump policy drives this env in the current control step

    All six are zero / False after construction and after an episode reset of that env.

    Counted per episode in ``metrics``, (N,) float each: ``triggers_started`` (latches started), ``jumps_accepted``,
    ``triggers_failed`` (latches that ran out) and ``triggers_ignored`` (high-level steps with the trigger above the
    threshold but blocked). mjlab's ``CommandTerm.reset`` averages each one over the envs that end an episode, logs
    the mean as ``Metrics/hl_state/<name>`` and zeroes it for those envs.
    """

    def __init__(self, cfg: HighLevelStateCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.action = torch.zeros(self.num_envs, 2, device=self.device)
        self.prev_action = torch.zeros(self.num_envs, 2, device=self.device)
        self.trigger_high = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.pending = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.trigger_failed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.in_jump = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.metrics = {name: torch.zeros(self.num_envs, device=self.device)
                        for name in ("triggers_started", "jumps_accepted", "triggers_failed", "triggers_ignored")}

    @property
    def command(self) -> torch.Tensor:
        """The tensor mjlab requires every command term to expose: the current action, (N, 2)."""
        return self.action

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Zero the state of ``env_ids``; mjlab calls this for the envs that start a new episode."""
        for buffer in (self.action, self.prev_action, self.trigger_high, self.pending, self.trigger_failed,
                       self.in_jump):
            buffer[env_ids] = 0

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        """Nothing to advance: the wrapper is the only writer."""

    def _update_metrics(self) -> None:
        """Nothing to update: the wrapper counts the metrics as they happen."""


@dataclass(kw_only=True)
class HighLevelStateCfg(CommandTermCfg):
    """Configuration of the ``"hl_state"`` command term."""

    # s; longer than any run, so the state is never zeroed on a timer, only when mjlab resets an env.
    resampling_time_range: tuple[float, float] = (1e9, 1e9)

    def build(self, env: ManagerBasedRlEnv) -> HighLevelState:
        return HighLevelState(self, env)
