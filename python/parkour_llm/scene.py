"""The play env of env1 or env2 on chosen lanes, with a second front camera that the LLM looks through.

The PPO depth sensor (``parkour_rl.depth.DEPTH_SENSOR``) is left as it is, because the wrapper still encodes it.
The LLM camera sits at the same pose on the same body and renders depth and RGB at its own resolution.
"""

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers import EventTermCfg
from mjlab.sensor import CameraSensorCfg

from parkour_rl.env1 import ENV1_KINDS, ENV1_NUM_ROWS, make_env1_cfg, make_env1_terrain_cfg
from parkour_rl.env2 import ENV2_LANES, ENV2_TEST_LAYOUTS, ENV2_TRAIN_LAYOUTS, make_env2_cfg, make_env2_terrain_cfg

LLM_SENSOR = "llm_view"
# Pose and field of view of the MJCF camera "depth_front" (assets/beni_base/mjcf/beni.xml), on the same body.
CAMERA_BODY = "robot/base_link"
CAMERA_POS = (0.07, 0.03, -0.016)
CAMERA_QUAT = (0.5, 0.5, -0.5, -0.5)
CAMERA_FOVY = 86.529
PPO_SIZE = (60, 80)  # Rows, columns of the image the PPO policy's encoder gets.

Cell = tuple[int, int]  # (lane column, row)


def all_cells(task: str) -> list[Cell]:
    """Every lane of ``task`` ("env1" or "env2") as (column, row)."""
    columns, rows = (len(ENV1_KINDS), ENV1_NUM_ROWS) if task == "env1" else (
        ENV2_LANES, ENV2_TRAIN_LAYOUTS + ENV2_TEST_LAYOUTS)
    return [(column, row) for column in range(columns) for row in range(rows)]


def parse_cells(task: str, spec: str) -> list[Cell]:
    """Lanes named by ``spec``.

    ``all``; ``obstacles`` (env1: every lane but the five flat ones); ``test`` (env2: the held-out layout, row 4);
    an env1 obstacle kind such as ``pit`` (its five rows); or
    a comma-separated list of ``column:row`` or ``kind:row`` entries such as ``pit:2,hollow:4`` or ``3:4``.
    """
    cells = all_cells(task)
    if spec == "all":
        return cells
    if spec == "obstacles":
        if task != "env1":
            raise ValueError("'obstacles' names the 20 lanes of env1 that have an obstacle (rows 1-4)")
        return [cell for cell in cells if cell[1] >= 1]
    if spec == "test":
        if task != "env2":
            raise ValueError("'test' names the held-out layout of env2")
        return [cell for cell in cells if cell[1] >= ENV2_TRAIN_LAYOUTS]
    chosen = []
    for entry in spec.split(","):
        name, _, row = entry.partition(":")
        if name.isdigit():
            column = int(name)
        elif task == "env1" and name in ENV1_KINDS:
            column = ENV1_KINDS.index(name)
        else:
            raise ValueError(f"unknown lane '{entry}' for {task}")
        chosen += [cell for cell in cells if cell[0] == column and (not row or cell[1] == int(row))]
    if not chosen:
        raise ValueError(f"'{spec}' names no lane of {task}")
    return chosen


def cell_name(task: str, cell: Cell) -> str:
    """A short label of a lane: ``pit r3`` (env1) or ``lane 7 r4`` (env2)."""
    column, row = cell
    return f"{ENV1_KINDS[column]} r{row}" if task == "env1" else f"lane {column} r{row}"


def _assign_cells(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, cells: list[Cell]) -> None:
    """Startup event: put env i on lane ``cells[i]`` instead of the generator's random row."""
    del env_ids
    terrain = env.scene.terrain
    if env.num_envs != len(cells):
        raise ValueError(f"{len(cells)} lanes need {len(cells)} envs; got {env.num_envs}")
    terrain.terrain_types[:] = torch.tensor([column for column, _ in cells], device=env.device)
    terrain.terrain_levels[:] = torch.tensor([row for _, row in cells], device=env.device)
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]


def make_llm_env_cfg(task: str, cells: list[Cell], size: tuple[int, int] = PPO_SIZE) -> ManagerBasedRlEnvCfg:
    """Play env of ``task`` with one robot on each lane of ``cells`` and the LLM camera of ``size`` (rows, columns)."""
    if task == "env1":
        cfg, terrain = make_env1_cfg(play=True), make_env1_terrain_cfg()
    elif task == "env2":
        cfg, terrain = make_env2_cfg(play=True), make_env2_terrain_cfg()
    else:
        raise ValueError(f"task must be env1 or env2, not {task}")
    cfg.scene.num_envs = len(cells)
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = terrain
    cfg.events["llm_cells"] = EventTermCfg(func=_assign_cells, mode="startup", params={"cells": cells})

    # All camera sensors of a scene must agree on textures, shadows and geom groups (mujoco_warp), so these three
    # follow the PPO depth sensor: only geom group 0, the terrain with its lane walls, is rendered.
    ppo = cfg.scene.sensors[-1]
    cfg.scene.sensors += (CameraSensorCfg(
        name=LLM_SENSOR, parent_body=CAMERA_BODY, pos=CAMERA_POS, quat=CAMERA_QUAT, fovy=CAMERA_FOVY,
        width=size[1], height=size[0], data_types=("rgb", "depth"),
        use_textures=ppo.use_textures, use_shadows=ppo.use_shadows, enabled_geom_groups=ppo.enabled_geom_groups),)
    return cfg
