# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import CurriculumTermCfg

DEMOTE_BELOW = 0.4  
SUCCESS_EMA = 0.01  
SUCCESS_COUNT = 5
DEMOTE_COUNT = 2
# Graduation, logged only: every kind's top-row success rate at GRADUATION_RATE or more for GRADUATION_ITERATIONS PPO
# iterations in a row. A PPO iteration is 24 high-level steps (rl_cfg.py) of 5 control steps (hierarchy.SUB_STEPS).
GRADUATION_RATE = .85
GRADUATION_ITERATIONS = 200
CONTROL_STEPS_PER_ITERATION = 24 * 5


def _blend(rate: torch.Tensor, index: torch.Tensor, success: torch.Tensor) -> None:
    """Blend the outcomes ``success`` (n,) into ``rate`` in place: the outcome of episode i into entry ``index[i]``."""
    finished = torch.bincount(index, minlength=len(rate))  # Episodes that ended, per entry.
    succeeded = torch.bincount(index, weights=success.float(), minlength=len(rate))  # Those that succeeded.
    mean_success = succeeded / finished.clamp(min=1)  # 0 instead of 0 / 0 for an entry without an episode.
    # Episodes that end together add their weights, up to 1. An entry without one has weight 0 and keeps its rate.
    weight = (finished * SUCCESS_EMA).clamp(max=1.)
    rate += weight * (mean_success - rate)


class LaneCurriculum:

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv) -> None:
        del cfg
        self._terrain = env.scene.terrain
        # The obstacle kinds of the goal term, and the index in them of the kind of each lane column. The curriculum
        # manager is built after the command manager.
        goal = env.command_manager.get_term("goal")
        self._kinds, self._column_kind = goal.kinds, goal.column_kind
        self._num_rows = self._terrain.cfg.terrain_generator.num_rows
        # (len(kinds),) running fraction of the episodes that reached the goal, per kind of the goal term's kinds.
        self._success_rate = torch.zeros(len(self._kinds), device=env.device)
        # (len(kinds), rows) the same per kind and per row that the episode ran on.
        self._cell_rate = torch.zeros(len(self._kinds), self._num_rows, device=env.device)
        # (N,) successes in a row of each env on its current row.
        self._streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        # (N,) early failures in a row of each env on its current row.
        self._early_failures = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        # Graduation, logged only: the PPO iteration counted last, the iterations in a row in which every kind's
        # top-row rate was at GRADUATION_RATE or more, and whether that count has reached GRADUATION_ITERATIONS.
        self._iteration = 0
        self._graduation_iterations = 0
        self._graduated = False

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor,
                 before_obstacle: bool = False) -> dict[str, float]:
        """Judge the episodes that just ended in the envs ``env_ids`` (n,) and move those envs.

        A failure is early when the robot covered less than DEMOTE_BELOW of the way from its spawn to the goal, or,
        with ``before_obstacle`` (env1), when it ended before the robot root passed the last obstacle of its lane.
        With env1's random spawn and goal, a failure at the obstacle covers 0.2-0.75 of the way, so DEMOTE_BELOW
        would count it as early only part of the time.

        Returns the values to log: ``mean_level`` (mean row of all envs); for every kind of the goal term
        ``level_<kind>`` (mean row of the envs of that kind), ``success_<kind>`` (its running success rate) and, for
        every row r, ``success_<kind>_row<r>`` (the running success rate of its episodes on row r); and
        ``graduation_iterations`` and ``graduated`` (1 once graduated; see GRADUATION_RATE).
        """
        # mjlab also resets every env once before the first step. No episode has ended then, so nothing is judged.
        if env.common_step_counter > 0:
            success = env.termination_manager.get_term("goal_reached")[env_ids]
            # Read before the envs move and before the reset: the robot is still where the episode ended, and the
            # spawn, goal and lane are those of the episode.
            goal = env.command_manager.get_term("goal")
            if before_obstacle:
                early = goal.obstacles_left()[env_ids] > 0
            else:
                early = goal.covered_fraction()[env_ids] < DEMOTE_BELOW
            # Also before the move, so that every episode counts for the row it ran on.
            self._track_success(env_ids, success)
            self._move(env_ids, success, early)
            self._track_graduation(env.common_step_counter // CONTROL_STEPS_PER_ITERATION)
        return self._log()

    def _move(self, env_ids: torch.Tensor, success: torch.Tensor, early: torch.Tensor) -> None:
        terrain = self._terrain
        early_failure = ~success & early
        self._streak[env_ids] = torch.where(success, self._streak[env_ids] + 1, 0)
        self._early_failures[env_ids] = torch.where(early_failure, self._early_failures[env_ids] + 1, 0)
        promote = self._streak[env_ids] >= SUCCESS_COUNT
        demote = self._early_failures[env_ids] >= DEMOTE_COUNT

        # Count afresh on the new row.
        moved = env_ids[promote | demote]
        self._streak[moved] = 0
        self._early_failures[moved] = 0
        rows = (terrain.terrain_levels[env_ids] + promote.long() - demote.long()).clamp(0, self._num_rows - 1)

        # Written directly: terrain.update_env_origins sends an env that is promoted on the last row to a random row.
        terrain.terrain_levels[env_ids] = rows
        terrain.env_origins[env_ids] = terrain.terrain_origins[rows, terrain.terrain_types[env_ids]]

    def _track_success(self, env_ids: torch.Tensor, success: torch.Tensor) -> None:
        """Blend the outcomes ``success`` (n,) of the episodes of ``env_ids`` into the rates of their kind and cell."""
        kinds = self._column_kind[self._terrain.terrain_types[env_ids]]
        rows = self._terrain.terrain_levels[env_ids]
        _blend(self._success_rate, kinds, success)
        # A view of the (len(kinds), rows) rates: the rate of kind k on row r is entry k * rows + r.
        _blend(self._cell_rate.view(-1), kinds * self._num_rows + rows, success)

    def _track_graduation(self, iteration: int) -> None:
        """Count the PPO iterations in a row in which every kind's top-row success rate is at GRADUATION_RATE or more.

        ``iteration`` is the current PPO iteration; the count changes once per iteration. Logging only: graduating
        changes nothing in training. A kind without a finished episode on the top row has its initial rate, 0.
        """
        if iteration == self._iteration:
            return
        self._iteration = iteration
        passed = bool((self._cell_rate[:, -1] >= GRADUATION_RATE).all())
        self._graduation_iterations = self._graduation_iterations + 1 if passed else 0
        if self._graduation_iterations >= GRADUATION_ITERATIONS and not self._graduated:
            self._graduated = True
            print(f"Lane curriculum graduated at iteration {iteration}: every kind's top-row success rate stayed at "
                  f"{GRADUATION_RATE} or more for {GRADUATION_ITERATIONS} iterations", flush=True)

    def _log(self) -> dict[str, float]:
        """The values ``__call__`` returns, computed over all N envs."""
        rows, kinds = self._terrain.terrain_levels.float(), self._column_kind[self._terrain.terrain_types]
        envs_per_kind = torch.bincount(kinds, minlength=len(self._kinds))
        mean_rows = torch.bincount(kinds, weights=rows, minlength=len(self._kinds)) / envs_per_kind  # (len(kinds),)
        log = {"mean_level": rows.mean().item()}
        log.update({f"level_{kind}": row for kind, row in zip(self._kinds, mean_rows.tolist())})
        log.update({f"success_{kind}": rate for kind, rate in zip(self._kinds, self._success_rate.tolist())})
        log.update({f"success_{kind}_row{row}": rate for kind, rates in zip(self._kinds, self._cell_rate.tolist())
                    for row, rate in enumerate(rates)})
        log.update({"graduation_iterations": float(self._graduation_iterations), "graduated": float(self._graduated)})

        return log
