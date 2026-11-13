# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""Parkour obstacle lanes for mjlab's terrain generator in curriculum mode: 4 columns x 9 rows.

Each column is one obstacle kind and each row one difficulty step. A lane is a straight corridor
along +x with a wall on each side; the robot spawns at SPAWN_X. Rows 0-5 hold the column's own kind
(flat, four growing sizes, two at the limit), row 6 mixes the column's kind with the other kinds,
rows 7 and 8 are fully random with 3-4 and 5-6 obstacles. Counts, kinds, run-up and spacing come from
the generator's rng, so every cell differs. Each cell's obstacles are recorded in
``ParkourLaneCfg.layouts[row]``. The goal is not stored here: goal.py draws it on every reset,
GOAL_MARGIN past the last obstacle (FLAT_GOAL_RUN from the spawn on a flat cell).
``ParkourLaneCfg.rows`` replaces the row table for other layouts: env1.py builds 5 columns x 5 rows
from it, one obstacle per row, with a fifth kind, thin_cube, a bar the robot jumps over.
``ParkourLaneCfg.count`` makes every row, row 0 included, a lane of random obstacles instead: env2.py builds its lanes
of 4-6 mixed obstacles from it.

The ground of a lane is laid as boxes of at most MAX_GROUND_BOX metres placed end to end, because the
wheel contact flickers on one long box and jump requests are then rejected. ``ground_edges`` picks the
box edges so that the box centres, where the remaining flicker sits, stay out of the approach to every
obstacle the robot jumps (JUMP_KINDS).
"""

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np
from mjlab.terrains import SubTerrainCfg, TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput

KINDS = ("pit", "cube", "hollow", "ramp_cube")
NUM_ROWS = 9
LANE = (37., 2.)  # Length along x, width along y.
SPAWN_X = 1.
GOAL_MARGIN = (2., 4.)  # Goal distance past the last obstacle (m); goal.py draws it on every reset.
FLAT_GOAL_RUN = (3., 6.)  # Goal distance from the spawn on a flat cell (m).
END_CLEARANCE = .5  # Lane left past the farthest possible goal (m).
GROUND_THICKNESS = .5
WALL_HEIGHT = .5  # Above the ground; the walls also close the pit sides below it.
WALL_THICKNESS = .05
CUBE_LENGTH = 1.
THIN_CUBE_LENGTH = .10  # A thin_cube is a bar this long along x, across the lane.
PIT_DEPTH = .25
HOLLOW_LENGTH = .3
HOLLOW_THICKNESS = .1
RAMP_CUBE_HEIGHT = .15
RAMP_THICKNESS = .02  # The ramp box hangs this far below its top face, inside the ground block.
# Parameter of each kind: pit width, cube and thin_cube height, hollow clearance (m), ramp angle (deg). The pit width
# is its span along x, the distance to jump; a pit covers the full lane across y.
LIMIT = {"pit": .35, "cube": .15, "hollow": .20, "ramp_cube": 20., "thin_cube": .12}
# Rows 1-5 of each column: the parameter of every obstacle in the row, in lane order.
ROWS = {
    "pit": ((.15,), (.25,), (.30,), (.35,), (.35, .35)),
    "cube": ((.06,), (.10,), (.12,), (.15,), (.15, .15)),
    "hollow": ((.26,), (.23,), (.21,), (.20,), (.20, .20)),
    "ramp_cube": ((10.,), (15.,), (20.,), (20., 20.), (20., 20.)),
}
MIXED_COUNT = {6: (2, 3), 7: (3, 4), 8: (5, 6)}  # Row 6: the column's kind plus random others; 7, 8: all random.
RUN_UP = (2., 3.)  # SPAWN_X to the first obstacle (m).
# Gap between obstacles (m), by row. From row 5 on two obstacles at the limit can follow each other, so the gap leaves
# room for a second jump. With the low-level jump timing (1 s hold after touchdown) two pits 1.5-2.5 m apart were both
# crossed in only 0.02-0.34 of trials at 1 m/s, with the parkour timing of env.py (JUMP_RETRIGGER_HOLD_S) in about 0.7
# at 1.5-3 m (CPU, 2026-11-04).
SPACING = (None, (3., 4.), (3., 4.), (2.5, 3.5), (2.5, 3.5), (3., 4.), (3., 4.), (3., 4.), (3., 4.))
# A layout that ends later than GOAL_MARGIN[1] + END_CLEARANCE before the end of its lane is rejected when the terrain
# compiles, so the largest goal margin always fits. The 37 m LANE is long enough for the longest possible layout, six
# ramp_cubes at 20 deg in row 8, which ends at 1 + 3 + 6 * (0.15 / tan 20 + 1) + 5 * 4 = 32.47 m, below
# 37 - 4 - 0.5 = 32.5 m. Seed 0 ends at 28.74 m at most.

# Ground boxes. On one long box the wheel contact flickers and jump requests are rejected: a request was acceptable
# on 50-66% of driving steps on a 26 m box, on about 90% on boxes of 4 m or less and on 100% on a plane (measured on
# CPU). The remaining losses sit within about 0.3 m of a box centre and never at a seam, so box centres are kept out
# of the last APPROACH metres before the obstacles the robot jumps (JUMP_KINDS) and CENTRE_RADIUS past their front edge.
MAX_GROUND_BOX = 4.  # Longest ground box (m).
MIN_GROUND_BOX = .5  # Shortest ground box (m).
GROUND_GRID = .1  # Spacing of the candidate box edges (m).
JUMP_KINDS = ("pit", "cube", "thin_cube")  # Obstacles the robot jumps.
APPROACH = 1.3  # Run-up before a jump obstacle that box centres are kept out of (m).
CENTRE_RADIUS = .3  # Reach of the contact losses around a box centre (m).

GREY, DARK = (.5, .5, .5, 1.), (.3, .3, .3, 1.)
ORANGE, BLUE, GREEN = (.9, .5, .1, 1.), (.2, .4, .9, 1.), (.2, .7, .3, 1.)  # Cube, hollow slab, ramp_cube.
RED = (.85, .15, .15, 1.)  # Thin cube.


def obstacle_length(kind, value):
    if kind == "pit":
        return value
    if kind == "hollow":
        return HOLLOW_LENGTH
    if kind == "ramp_cube":
        return RAMP_CUBE_HEIGHT / math.tan(math.radians(value)) + CUBE_LENGTH
    if kind == "thin_cube":
        return THIN_CUBE_LENGTH
    return CUBE_LENGTH


def ground_edges(x_min, x_max, zones):
    """Edges (list of floats, first x_min, last x_max) of the ground boxes that cover [x_min, x_max].

    Each box is MIN_GROUND_BOX to MAX_GROUND_BOX long. Among such splits the one with the fewest box centres
    inside ``zones`` (open intervals (low, high) in lane x) is chosen, then the one with the fewest boxes.
    Candidate edges lie GROUND_GRID apart, counted from x_min. The stretch must be at least MIN_GROUND_BOX long;
    the shortest one in a lane, between two pits of rows 5-8, is 3 m.
    """
    slack = 1e-9  # Rounding allowance (m) when a length is compared with a limit.
    grid_steps = int((x_max - x_min - slack) / GROUND_GRID)
    xs = [x_min + k * GROUND_GRID for k in range(grid_steps + 1)] + [x_max]  # Candidate edges.
    # Dynamic programme over the candidate edges from left to right. best[i] is the best way to cover
    # [x_min, xs[i]]: (box centres inside a zone, boxes, index of the previous edge), compared in that order.
    best = {0: (0, 0, None)}
    for i in range(1, len(xs)):
        options = []
        for j, (inside, boxes, _) in best.items():
            if MIN_GROUND_BOX - slack <= xs[i] - xs[j] <= MAX_GROUND_BOX + slack:
                centre = (xs[i] + xs[j]) / 2
                options.append((inside + any(low < centre < high for low, high in zones), boxes + 1, j))
        if options:  # There is none while xs[i] is closer than MIN_GROUND_BOX to x_min.
            best[i] = min(options)
    edges, i = [], len(xs) - 1
    while i is not None:
        edges.append(xs[i])
        i = best[i][2]
    return edges[::-1]


@dataclass(kw_only=True)
class ParkourLaneCfg(SubTerrainCfg):
    """One lane column; the row comes from the generator's difficulty."""

    kind: str
    layouts: dict = field(default_factory=dict)  # Row -> obstacles (kind, x_start, x_end, value), filled on compile.
    num_rows: int = NUM_ROWS  # Rows of the generator; a lane's row is round(difficulty * (num_rows - 1)).
    # Kind -> parameters of rows 1, 2, ... of a layout of its own (as in ROWS, row 0 flat); None: ROWS and MIXED_COUNT.
    rows: dict[str, tuple[tuple[float, ...], ...]] | None = None
    # Random lanes (env2.py): the number of obstacles of a lane (lowest, highest); None: the row tables above. Each
    # obstacle is of a kind drawn from the keys of ``values`` with its parameter uniform in the range given there
    # (lowest, highest), and lies ``gap`` metres (lowest, highest) behind the one before it.
    count: tuple[int, int] | None = None
    values: dict[str, tuple[float, float]] | None = None
    gap: tuple[float, float] | None = None

    def function(self, difficulty, spec, rng):
        row = round(difficulty * (self.num_rows - 1))
        if self.count is not None:
            kinds = [str(k) for k in rng.choice(list(self.values), int(rng.integers(self.count[0], self.count[1] + 1)))]
            items = [(kind, float(rng.uniform(*self.values[kind]))) for kind in kinds]
        elif row == 0:
            items = []
        elif self.rows is not None:
            items = [(self.kind, value) for value in self.rows[self.kind][row - 1]]
        elif row <= 5:
            items = [(self.kind, value) for value in ROWS[self.kind][row - 1]]
        else:
            count = int(rng.integers(MIXED_COUNT[row][0], MIXED_COUNT[row][1] + 1))
            own = [self.kind] if row == 6 else []
            pool = [k for k in KINDS if k != self.kind] if row == 6 else list(KINDS)
            kinds = own + [str(k) for k in rng.choice(pool, count - len(own))]
            rng.shuffle(kinds)
            items = [(kind, LIMIT[kind]) for kind in kinds]

        spacing = SPACING[row] if self.count is None else self.gap
        x = SPAWN_X + rng.uniform(*RUN_UP)
        obstacles = []
        for i, (kind, value) in enumerate(items):
            if i:
                x += rng.uniform(*spacing)
            obstacles.append(dict(kind=kind, x_start=x, x_end=x + obstacle_length(kind, value), value=value))
            x = obstacles[-1]["x_end"]
        max_end_x = self.size[0] - GOAL_MARGIN[1] - END_CLEARANCE
        if x > max_end_x:
            if self.count is not None:  # A random lane that came out too long: draw another one. Nothing is built yet.
                return self.function(difficulty, spec, rng)
            raise ValueError(f"{self.kind} row {row}: layout ends at {x:.2f} m, past {max_end_x:.2f} m, "
                             f"which leaves no room for the {GOAL_MARGIN[1]:g} m goal margin")
        self.layouts[row] = obstacles
        # Stretches of lane x that the centres of the ground boxes should stay out of.
        zones = [(o["x_start"] - APPROACH, o["x_start"] + CENTRE_RADIUS) for o in obstacles if o["kind"] in JUMP_KINDS]

        body = spec.body("terrain")
        length, width = self.size

        def box(x_min, x_max, z_min, z_max, color):
            geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                                 size=((x_max-x_min)/2, width/2, (z_max-z_min)/2),
                                 pos=((x_min+x_max)/2, width/2, (z_min+z_max)/2))
            return TerrainGeometry(geom=geom, color=color)

        def ground(x_min, x_max):
            """Ground from x_min to x_max, top face at z = 0, as short boxes laid end to end."""
            edges = ground_edges(x_min, x_max, zones)
            return [box(low, high, -GROUND_THICKNESS, 0., GREY) for low, high in zip(edges, edges[1:])]

        def ramp(x_min, angle):
            """Box rotated about y whose top face rises from (x_min, 0) to the plateau height."""
            theta = math.radians(angle)
            run, slope = RAMP_CUBE_HEIGHT / math.tan(theta), RAMP_CUBE_HEIGHT / math.sin(theta)
            normal = np.array((-math.sin(theta), 0., math.cos(theta)))
            center = np.array((x_min + run/2, width/2, RAMP_CUBE_HEIGHT/2)) - RAMP_THICKNESS/2 * normal
            geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(slope/2, width/2, RAMP_THICKNESS/2),
                                 pos=center, quat=(math.cos(theta/2), 0., -math.sin(theta/2), 0.))
            return TerrainGeometry(geom=geom, color=GREEN)

        def wall(y_min):
            geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                                 size=(length/2, WALL_THICKNESS/2, (WALL_HEIGHT+GROUND_THICKNESS)/2),
                                 pos=(length/2, y_min + WALL_THICKNESS/2, (WALL_HEIGHT-GROUND_THICKNESS)/2))
            return TerrainGeometry(geom=geom, color=GREY)

        geometries, ground_x = [], 0.
        for o in obstacles:
            x_start, x_end, value = o["x_start"], o["x_end"], o["value"]
            if o["kind"] == "pit":
                geometries += ground(ground_x, x_start) + [box(x_start, x_end, -GROUND_THICKNESS, -PIT_DEPTH, DARK)]
                ground_x = x_end
            elif o["kind"] == "cube":
                geometries.append(box(x_start, x_end, 0., value, ORANGE))
            elif o["kind"] == "hollow":
                geometries.append(box(x_start, x_end, value, value + HOLLOW_THICKNESS, BLUE))
            elif o["kind"] == "thin_cube":
                geometries.append(box(x_start, x_end, 0., value, RED))
            else:
                geometries += [ramp(x_start, value), box(x_end - CUBE_LENGTH, x_end, 0., RAMP_CUBE_HEIGHT, GREEN)]
        geometries += ground(ground_x, length) + [wall(0.), wall(width - WALL_THICKNESS)]
        return TerrainOutput(origin=np.array((SPAWN_X, width/2, 0.)), geometries=geometries)


def make_parkour_terrain_cfg(seed=0):
    return TerrainGeneratorCfg(size=LANE, num_rows=NUM_ROWS, curriculum=True, seed=seed,
                               sub_terrains={kind: ParkourLaneCfg(kind=kind) for kind in KINDS})
