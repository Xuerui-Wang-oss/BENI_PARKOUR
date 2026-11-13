# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""The ``"goal"`` command term of the parkour task: the goal and time budget of an episode, and lane queries.

On every episode reset the goal is drawn at a random distance past the last obstacle of the env's lane (from the
spawn on a lane without obstacles), so that the policy cannot stop by dead reckoning, and the time budget of the
episode follows from the distance between the spawn and the goal. The term also answers the questions about the lane
that observations, rewards, terminations and the curriculum ask: how far the robot is along the lane and from the
goal, how high the terrain under it is, and which obstacles come next. The lanes, their obstacle kinds and the number
of rows are read from the terrain configuration, so the term serves every parkour layout (Beni-Parkour, env1).

Positions are in the lane frame: x in metres along the lane, 0 at the env origin. The robot spawns there unless a
spawn event moves it along the lane (env1.py); ``spawn_x`` holds the spawn of the episode. Lanes run along world +x.
``N`` is the number of envs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from .parkour_terrain import FLAT_GOAL_RUN, GOAL_MARGIN, LIMIT, PIT_DEPTH, RAMP_CUBE_HEIGHT, SPAWN_X

# Time budget of an episode: TIME_BASE_S plus the time to cover the distance from the spawn to the goal at
# MIN_AVERAGE_SPEED. It grows with the distance because goals lie between 3 m (flat lane) and 35.5 m (the longest
# layout the terrain accepts) from the spawn. The largest budget, 4 + 35.5 / 0.6 = 63.2 s, has to stay below the
# episode length of the env.
TIME_BASE_S = 4.  # s
MIN_AVERAGE_SPEED = .6  # m/s
OBSTACLE_FAR = 10.  # Obstacle distances are clipped to this; it also stands for "no obstacle" (m).

# Obstacle table entry: (kind id, x_start, x_end, value, value / LIMIT[kind]). The kind id is the index in
# GoalCommand.kinds.
X_END = 2  # Index of x_end in an entry.
BEYOND_LANE = 1e3  # x of a padding entry, which the robot never reaches (m).
PADDING = (-1., BEYOND_LANE, BEYOND_LANE, 0., 0.)
NO_KIND = -2.  # Kind id of a kind no lane has; it matches no entry, padding (-1) included.


def obstacle_entries(layout: list[dict], kinds: tuple[str, ...], size: int) -> list[tuple]:
    """Table entries of one lane: its obstacles in lane order, padded to ``size`` entries.

    ``layout`` is one ``ParkourLaneCfg.layouts[row]``; an entry's kind id is the index of its kind in ``kinds``.
    ``size`` exceeds the number of obstacles of every lane, so that every lane ends with padding.
    """
    entries = [(kinds.index(o["kind"]), o["x_start"] - SPAWN_X, o["x_end"] - SPAWN_X, o["value"],
                o["value"] / LIMIT[o["kind"]]) for o in layout]
    return entries + [PADDING] * (size - len(entries))


def goal_range(layout: list[dict]) -> tuple[float, float]:
    """Nearest and farthest goal of one lane (m): GOAL_MARGIN past its last obstacle, FLAT_GOAL_RUN past the spawn if
    it has none."""
    if not layout:
        return FLAT_GOAL_RUN
    last_end = layout[-1]["x_end"] - SPAWN_X
    return last_end + GOAL_MARGIN[0], last_end + GOAL_MARGIN[1]


class GoalCommand(CommandTerm):
    """Goal and time budget of every env, drawn on each episode reset, and queries about the env's lane.

    Buffers, written only when an env is reset:
        spawn_x      (N,) float  spawn position along the lane (m); 0 unless a reset event calls ``sample_spawn_x``
        goal_x       (N,) float  goal position along the lane (m)
        time_budget  (N,) float  time allowed to reach the goal (s)

    Read-only, from the terrain configuration:
        kinds        tuple of str     obstacle kinds of the lane columns, each once, in column order; ``cfg.kinds``
                                      where the lanes mix kinds (env2.py)
        column_kind  (columns,) long  index in ``kinds`` of the kind of each lane column; None with ``cfg.kinds``

    The query methods compute their answer from the current simulation state on every call. They are not cached in
    ``_update_command``: inside ``env.step`` terminations and rewards run before the command manager, so a cache
    would be one control step old.
    """

    cfg: GoalCommandCfg

    def __init__(self, cfg: GoalCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._robot = env.scene["robot"]
        self._terrain = env.scene.terrain
        generator = self._terrain.cfg.terrain_generator
        # One lane cfg per column, in column order; their layouts were filled when the scene compiled the terrain.
        lanes = list(generator.sub_terrains.values())
        if cfg.kinds is None:
            self.kinds = tuple(dict.fromkeys(lane.kind for lane in lanes))
            self.column_kind = torch.tensor([self.kinds.index(lane.kind) for lane in lanes], device=self.device)
        else:
            self.kinds, self.column_kind = cfg.kinds, None
        layouts = [[lane.layouts[row] for lane in lanes] for row in range(generator.num_rows)]
        size = max(len(layout) for row in layouts for layout in row) + 1
        # (rows, columns, size, 5) and (rows, columns, 2), indexed by an env's row and column.
        self._obstacle_table = torch.tensor([[obstacle_entries(layout, self.kinds, size) for layout in row]
                                             for row in layouts], device=self.device)
        self._goal_ranges = torch.tensor([[goal_range(layout) for layout in row] for row in layouts],
                                         device=self.device)
        # Kind ids that ground_height tests for; NO_KIND for a kind that no lane has.
        self._kind_id = {kind: float(self.kinds.index(kind)) if kind in self.kinds else NO_KIND
                         for kind in ("pit", "cube", "ramp_cube", "thin_cube")}
        # (rows, columns): the cells without obstacles, and the farthest spawn of every cell (m).
        self._flat = self._obstacle_table[:, :, 0, 0] < 0
        self._spawn_max = torch.where(self._flat, cfg.flat_spawn_max_x,
                                      self._obstacle_table[:, :, 0, 1] - cfg.min_run_up)
        self.spawn_x = torch.zeros(self.num_envs, device=self.device)
        self.goal_x = torch.zeros(self.num_envs, device=self.device)
        self.time_budget = torch.zeros(self.num_envs, device=self.device)
        # Draw now: the observation manager calls the observation functions once while the env is being built.
        self._resample_command(torch.arange(self.num_envs, device=self.device))

    @property
    def command(self) -> torch.Tensor:
        """The tensor mjlab requires every command term to expose: the goal position, (N, 1)."""
        return self.goal_x.unsqueeze(1)

    def position(self) -> torch.Tensor:
        """(N,) x of the robot root in the lane frame (m)."""
        return self._robot.data.root_link_pos_w[:, 0] - self._env.scene.env_origins[:, 0]

    def distance(self) -> torch.Tensor:
        """(N,) distance left to the goal along the lane (m); negative past the goal."""
        return self.goal_x - self.position()

    def covered_fraction(self) -> torch.Tensor:
        """(N,) fraction of the way from the spawn to the goal that the robot has covered, in [0, 1]."""
        return ((self.position() - self.spawn_x) / (self.goal_x - self.spawn_x)).clamp(0., 1.)

    def time_fraction_left(self) -> torch.Tensor:
        """(N,) fraction of the time budget that is left, in [0, 1]; 0 once the budget is used up."""
        elapsed = self._env.episode_length_buf * self._env.step_dt
        return (1. - elapsed / self.time_budget).clamp(min=0.)

    def ground_height(self) -> torch.Tensor:
        """(N,) height of the terrain top under the robot root, relative to the lane ground (m).

        Negative on a pit floor, positive on a cube, a thin_cube, a ramp or its plateau, 0 on the ground and under a
        hollow.
        """
        x = self.position().unsqueeze(1)
        kind, x_start, x_end, value, _ = self._lane_obstacles().unbind(2)  # Each (N, entries).
        # The value of a cube is its height (m), that of a ramp_cube the angle of its ramp (deg).
        ramp = ((x - x_start) * torch.tan(torch.deg2rad(value))).clamp(max=RAMP_CUBE_HEIGHT)
        top = torch.zeros_like(value)  # Stays 0 for a hollow: the robot passes under the slab.
        top = torch.where(kind == self._kind_id["pit"], -PIT_DEPTH, top)
        top = torch.where((kind == self._kind_id["cube"]) | (kind == self._kind_id["thin_cube"]), value, top)
        top = torch.where(kind == self._kind_id["ramp_cube"], ramp, top)
        inside = (x_start <= x) & (x < x_end)  # True for at most one obstacle per env: obstacles do not overlap.
        return (top * inside).sum(1)

    def next_obstacles(self, count: int) -> torch.Tensor:
        """(N, count, len(kinds) + 3) the next ``count`` obstacles of each lane that the robot root has not passed,
        in lane order.

        Per obstacle: one-hot kind over ``kinds``, distance to its start and to its end (m, at most OBSTACLE_FAR;
        the start distance is negative while the robot is inside the obstacle), value / LIMIT[kind]. Where the lane
        has no further obstacle: zeros for the kind and the value, OBSTACLE_FAR for both distances.
        """
        x = self.position().unsqueeze(1)
        obstacles = self._lane_obstacles()
        passed = (obstacles[:, :, X_END] <= x).sum(1, keepdim=True)  # (N, 1) obstacles that lie behind the root.
        # (N, count) entries to report. Past the lane's last obstacle they are clamped to the last entry, which is
        # padding.
        entries = (passed + torch.arange(count, device=self.device)).clamp(max=self._obstacle_table.shape[2] - 1)
        envs = torch.arange(self.num_envs, device=self.device).unsqueeze(1)  # (N, 1), pairs each env with its entries.
        kind, x_start, x_end, _, relative_value = obstacles[envs, entries].unbind(2)  # Each (N, count).
        # The kind id of a padding entry (-1) matches no kind, so its one-hot is all zeros.
        one_hot = kind.unsqueeze(2) == torch.arange(len(self.kinds), device=self.device)
        distances = torch.stack((x_start - x, x_end - x), dim=2).clamp(max=OBSTACLE_FAR)
        return torch.cat((one_hot.float(), distances, relative_value.unsqueeze(2)), dim=2)

    def obstacles_left(self) -> torch.Tensor:
        """(N,) obstacles of each lane that the robot root has not passed, the next one included."""
        kind, _, x_end, _, _ = self._lane_obstacles().unbind(2)  # Each (N, entries).
        # A padding entry has kind -1 and is not counted.
        return ((kind >= 0) & (x_end > self.position().unsqueeze(1))).sum(1).float()

    def obstacle_outcomes(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Kind id (N, entries) of every obstacle of each lane, in lane order, -1 for padding; and (N, entries) bool:
        the robot root has passed it."""
        kind, _, x_end, _, _ = self._lane_obstacles().unbind(2)
        return kind, (kind >= 0) & (x_end <= self.position().unsqueeze(1))

    def on_flat_lane(self) -> torch.Tensor:
        """(N,) bool: the lane of the env has no obstacle."""
        return self._flat[self._terrain.terrain_levels, self._terrain.terrain_types]

    def near(self, kind: str, ahead: float, behind: float) -> torch.Tensor:
        """(N,) bool: the robot root is between ``ahead`` (m) before the start and ``behind`` (m) past the end of an
        obstacle of ``kind`` in its lane. False everywhere if no lane has that kind."""
        if kind not in self.kinds:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        x = self.position().unsqueeze(1)
        kind_id, x_start, x_end, _, _ = self._lane_obstacles().unbind(2)  # Each (N, entries).
        return ((kind_id == self.kinds.index(kind)) & (x >= x_start - ahead) & (x <= x_end + behind)).any(1)

    def _lane_obstacles(self) -> torch.Tensor:
        """(N, entries, 5) table entries of the lane each env is on.

        Looked up on every call: the curriculum, and the startup event of the 36-cell play layout, move envs to
        other lanes after this term is built.
        """
        return self._obstacle_table[self._terrain.terrain_levels, self._terrain.terrain_types]

    def sample_spawn_x(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Draw a spawn x (m) for ``env_ids`` on their current cells, store it in ``spawn_x`` and return it, (n,).

        Uniform in [spawn_min_x, start of the first obstacle - min_run_up], or in [spawn_min_x, flat_spawn_max_x] on a
        cell without obstacles. The spawn event of env1.py calls this after the curriculum has moved the env and before
        ``_resample_command``, which then measures the goal and the time budget from the new spawn.
        """
        rows, columns = self._terrain.terrain_levels[env_ids], self._terrain.terrain_types[env_ids]
        low, high = self.cfg.spawn_min_x, self._spawn_max[rows, columns]
        self.spawn_x[env_ids] = low + (high - low) * torch.rand(len(env_ids), device=self.device)
        return self.spawn_x[env_ids]

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Draw the goal and the time budget of ``env_ids``; mjlab calls this for the envs that start an episode.

        It runs after the curriculum has moved the env to its new lane, but before the reset pose reaches the
        simulation, so it reads the lane and not the robot. The goal of a cell without obstacles is measured from
        ``spawn_x``, which a spawn event has already drawn: mjlab runs the reset events before the command terms.
        """
        rows, columns = self._terrain.terrain_levels[env_ids], self._terrain.terrain_types[env_ids]
        nearest, farthest = self._goal_ranges[rows, columns].unbind(1)
        goal_x = nearest + (farthest - nearest) * torch.rand(len(env_ids), device=self.device)
        spawn_x = self.spawn_x[env_ids]
        self.goal_x[env_ids] = torch.where(self._flat[rows, columns], spawn_x + goal_x, goal_x)
        self.time_budget[env_ids] = TIME_BASE_S + (self.goal_x[env_ids] - spawn_x) / MIN_AVERAGE_SPEED

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        """Nothing to advance: the goal is fixed for the episode and the queries read the simulation directly."""

    def _update_metrics(self) -> None:
        """Nothing to log."""


@dataclass(kw_only=True)
class GoalCommandCfg(CommandTermCfg):
    """Configuration of the ``"goal"`` command term."""

    # s; longer than any run, so the goal is never re-drawn on a timer, only when mjlab resets an env.
    resampling_time_range: tuple[float, float] = (1e9, 1e9)
    # Spawn range along the lane (m) of env1.py's spawn event; without that event ``spawn_x`` stays 0.
    spawn_min_x: float = -.5
    min_run_up: float = 1.5  # Least distance from the spawn to the first obstacle (m).
    flat_spawn_max_x: float = 1.5  # Farthest spawn on a cell without obstacles (m).
    # Obstacle kinds of the task, in the order of the one-hot that the critic observes, for lanes that mix kinds
    # (env2.py). None: every lane column has one kind, and the kinds are read from the columns.
    kinds: tuple[str, ...] | None = None

    def build(self, env: ManagerBasedRlEnv) -> GoalCommand:
        return GoalCommand(self, env)
