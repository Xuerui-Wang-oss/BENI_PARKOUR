"""Start and goal positions taken from a table, instead of the env's random draw.

The table is an episode file of the trained policy's evaluation: one row per episode with ``lane`` (1 for lane
column 0), ``episode`` (1 for the first), ``spawn_x_m``, ``spawn_y_m`` and ``goal_x_m``, all in the lane frame. With it
a run starts every episode where the trained policy started it and drives to the same goal, so the two are compared
on the same episodes. Only the positions are taken: no outcome of the table reaches the env or the model.

``install`` is called on the env cfg before the env is built and ``attach`` on the env after. The table is held in
the module (``ACTIVE``), not in the cfg, so that it is the same object wherever the cfg was copied to.
"""

import csv
from dataclasses import dataclass, fields
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers import EventTermCfg

from parkour_rl.goal import MIN_AVERAGE_SPEED, TIME_BASE_S, GoalCommand, GoalCommandCfg


class Starts:
    """The table: per lane column, its episodes in order as (spawn x, spawn y, goal x) in metres."""

    def __init__(self, path: Path) -> None:
        self.path = path
        rows: dict[int, dict[int, tuple[float, float, float]]] = {}
        with path.open() as file:
            for row in csv.DictReader(file):
                rows.setdefault(int(row["lane"]) - 1, {})[int(row["episode"])] = (
                    float(row["spawn_x_m"]), float(row["spawn_y_m"]), float(row["goal_x_m"]))
        self.table = {column: [episodes[n] for n in sorted(episodes)] for column, episodes in rows.items()}
        self.episodes = min(len(episodes) for episodes in self.table.values())  # Episodes every lane has.
        self.episode: list[int] = []  # Per env, the episodes it has finished; ``attach`` keeps it.

    def of(self, columns: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor | None:
        """(n, 3) spawn x, spawn y, goal x of the episode each env of ``env_ids`` is about to start.

        ``columns`` (n,) are their lane columns. None when a column is not in the table. An env that has run all
        its episodes is parked on its last one.
        """
        values = []
        for column, env_id in zip(columns.tolist(), env_ids.tolist()):
            episodes = self.table.get(column)
            if episodes is None:
                return None
            done = self.episode[env_id] if env_id < len(self.episode) else 0
            values.append(episodes[min(done, len(episodes) - 1)])
        return torch.tensor(values, device=columns.device, dtype=torch.float32).reshape(-1, 3)


ACTIVE: Starts | None = None


def _place(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
    """Reset event, run after the spawn event: move the robots of ``env_ids`` to the spawn of the table.

    The pose is read back from qpos, as in ``parkour_rl.env1._spawn_along_lane``: x holds the random spawn that
    event just added, which is replaced, and y is set from the lane's centre line.
    """
    env_ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    goal = env.command_manager.get_term("goal")
    values = ACTIVE.of(env.scene.terrain.terrain_types[env_ids], env_ids)
    if values is None:
        return
    robot = env.scene["robot"]
    pose = robot.data.data.qpos[env_ids[:, None], robot.data.indexing.free_joint_q_adr]  # (n, 7): position, quat.
    pose[:, 0] += values[:, 0] - goal.spawn_x[env_ids]
    pose[:, 1] = env.scene.env_origins[env_ids, 1] + values[:, 1]
    goal.spawn_x[env_ids] = values[:, 0]
    robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)


class FixedGoalCommand(GoalCommand):
    """The goal term with the goal of the table. The time budget follows from it as it always does."""

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        values = ACTIVE.of(self._terrain.terrain_types[env_ids], env_ids)
        if values is None:
            return
        self.spawn_x[env_ids], self.goal_x[env_ids] = values[:, 0], values[:, 2]
        self.time_budget[env_ids] = TIME_BASE_S + (values[:, 2] - values[:, 0]) / MIN_AVERAGE_SPEED


@dataclass(kw_only=True)
class FixedGoalCommandCfg(GoalCommandCfg):
    def build(self, env: ManagerBasedRlEnv) -> FixedGoalCommand:
        return FixedGoalCommand(self, env)


def install(cfg: ManagerBasedRlEnvCfg, starts: Starts) -> None:
    """Make the env of ``cfg`` start its episodes from ``starts``."""
    global ACTIVE
    ACTIVE = starts
    starts.episode = [0] * cfg.scene.num_envs
    goal = cfg.commands["goal"]
    cfg.commands["goal"] = FixedGoalCommandCfg(**{field.name: getattr(goal, field.name) for field in fields(goal)})
    cfg.events["llm_starts"] = EventTermCfg(func=_place, mode="reset")  # Last, so after the spawn event.


def attach(env: ManagerBasedRlEnv, starts: Starts) -> None:
    """Count the episodes each env finishes, so that its next reset takes the next row of the table.

    The termination manager is wrapped, as in ``parkour_rl.evaluate.record_episodes``: the reset of an env follows
    its termination on the same control step.
    """
    manager = env.termination_manager
    compute = manager.compute

    def counting_compute() -> torch.Tensor:
        finished = compute()
        for env_id in finished.nonzero().flatten().tolist():
            starts.episode[env_id] += 1
        return finished

    manager.compute = counting_compute


def placed(env: ManagerBasedRlEnv) -> list[tuple[float, float, float, float]]:
    """Per env, where it stands now: lane x, lane y, goal x and time budget (m, m, m, s). For checking a start."""
    goal = env.command_manager.get_term("goal")
    y = env.scene["robot"].data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
    return [tuple(round(float(v), 4) for v in row)
            for row in zip(goal.position(), y, goal.goal_x, goal.time_budget)]
