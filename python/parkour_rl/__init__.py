# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""Register the BENI parkour tasks for mjlab.

Beni-Parkour is the high-level task: a policy that sees a depth image drives the frozen crawl and jump policies
through the parkour lanes (env.py, rl_cfg.py). Start it only through ``parkour_rl.train``, ``parkour_rl.play`` or
``parkour_rl.check``: they put ``hierarchy.ParkourVecEnvWrapper`` between the env and the policy, and the task does
not run without it.

Beni-Parkour-Env1 is env1 of the high-level task (env1.py): five lane columns (pit, cube, hollow, ramp_cube,
thin_cube) of five rows. Its play form runs one robot per cell of the 5 x 5 grid: env i sits on column i // 5 and
row i % 5.

Beni-Parkour-Env2 is env2 (env2.py): 16 lane columns of 4-6 mixed obstacles in five rows, four training layouts and
a test layout. Its play form runs one robot per lane: env i sits on column i // 5 and row i % 5.

Beni-Parkour-Preview-Jump and Beni-Parkour-Preview-Crawl are the preview tasks: one low-level policy driven through
the parkour lanes with the joystick (preview.py).

The previews and the play form of Beni-Parkour run one robot per cell of the 4 x 9 lane grid. Env i sits on column
i // 9 (0 pit, 1 cube, 2 hollow, 3 ramp_cube) and row i % 9.
"""

import sys

import torch
from mjlab.managers import EventTermCfg
from mjlab.tasks.registry import register_mjlab_task

from beni_rl.beni_jump_env_cfg import make_beni_jump_env_cfg
from beni_rl.rl_cfg import BeniVelocityOnPolicyRunner as JumpRunner, beni_jump_ppo_runner_cfg
from beni_rl_crawl.beni_crawl_env_cfg import make_beni_crawl_env_cfg
from beni_rl_crawl.rl_cfg import BeniVelocityOnPolicyRunner as CrawlRunner, beni_crawl_ppo_runner_cfg

from .env import make_parkour_env_cfg
from .env1 import env1_ppo_runner_cfg, make_env1_cfg, make_env1_terrain_cfg
from .env2 import Env2Runner, env2_ppo_runner_cfg, make_env2_cfg, make_env2_terrain_cfg
from .parkour_terrain import KINDS, NUM_ROWS, make_parkour_terrain_cfg
from .rl_cfg import parkour_ppo_runner_cfg

PARKOUR_TASK_ID = "Beni-Parkour"
ENV1_TASK_ID = "Beni-Parkour-Env1"
ENV2_TASK_ID = "Beni-Parkour-Env2"
JUMP_TASK_ID = "Beni-Parkour-Preview-Jump"
CRAWL_TASK_ID = "Beni-Parkour-Preview-Crawl"
NUM_CELLS = len(KINDS) * NUM_ROWS


def task_from_argv() -> str:
    """The high-level task that the scripts run: Beni-Parkour, or the env that an optional first command-line
    argument names, ``env1`` or ``env2``. The argument is removed from ``sys.argv``, where tyro would not know it."""
    task_id = {"env1": ENV1_TASK_ID, "env2": ENV2_TASK_ID}.get(sys.argv[1] if len(sys.argv) > 1 else "")
    if task_id is None:
        return PARKOUR_TASK_ID
    del sys.argv[1]
    return task_id


def assign_cells(env, env_ids):
    """Startup event: put env i on its cell instead of the generator's random row, column i // rows and row i % rows."""
    del env_ids
    terrain = env.scene.terrain
    rows = terrain.cfg.terrain_generator.num_rows
    cells = rows * len(terrain.cfg.terrain_generator.sub_terrains)
    if env.num_envs != cells:
        raise ValueError(f"The parkour cell layout needs exactly {cells} envs, one per cell; got {env.num_envs}")
    index = torch.arange(env.num_envs, device=env.device)
    terrain.terrain_types[:] = index // rows
    terrain.terrain_levels[:] = index % rows
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]


def attach_parkour_terrain(cfg, terrain_cfg=None):
    """Give the env configuration ``cfg`` the lanes ``terrain_cfg`` with one env per cell; changes and returns ``cfg``.

    Without ``terrain_cfg`` the lanes are the 4 x 9 parkour lanes of parkour_terrain.py.
    """
    terrain_cfg = make_parkour_terrain_cfg() if terrain_cfg is None else terrain_cfg
    cfg.scene.num_envs = terrain_cfg.num_rows * len(terrain_cfg.sub_terrains)
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = terrain_cfg
    cfg.events["parkour_cells"] = EventTermCfg(func=assign_cells, mode="startup")
    return cfg


register_mjlab_task(
    task_id=JUMP_TASK_ID,
    env_cfg=attach_parkour_terrain(make_beni_jump_env_cfg(play=True)),
    play_env_cfg=attach_parkour_terrain(make_beni_jump_env_cfg(play=True)),
    rl_cfg=beni_jump_ppo_runner_cfg(),
    runner_cls=JumpRunner,
)

register_mjlab_task(
    task_id=CRAWL_TASK_ID,
    env_cfg=attach_parkour_terrain(make_beni_crawl_env_cfg(play=True)),
    play_env_cfg=attach_parkour_terrain(make_beni_crawl_env_cfg(play=True)),
    rl_cfg=beni_crawl_ppo_runner_cfg(),
    runner_cls=CrawlRunner,
)

# Training starts every env on row 0 or 1 and moves it between the rows with the lane curriculum; play puts one env on
# every cell. No runner_cls: mjlab's own runner trains the task, and what is specific to it sits in the env wrapper.
register_mjlab_task(
    task_id=PARKOUR_TASK_ID,
    env_cfg=make_parkour_env_cfg(),
    play_env_cfg=attach_parkour_terrain(make_parkour_env_cfg(play=True)),
    rl_cfg=parkour_ppo_runner_cfg(),
)

# The same for env1, on its own 5 x 5 lanes.
register_mjlab_task(
    task_id=ENV1_TASK_ID,
    env_cfg=make_env1_cfg(),
    play_env_cfg=attach_parkour_terrain(make_env1_cfg(play=True), make_env1_terrain_cfg()),
    rl_cfg=env1_ppo_runner_cfg(),
)

# env2: training moves every env between the four training layouts; play puts one env on every lane of all five.
# Env2Runner starts a new run from the env1 actor.
register_mjlab_task(
    task_id=ENV2_TASK_ID,
    env_cfg=make_env2_cfg(),
    play_env_cfg=attach_parkour_terrain(make_env2_cfg(play=True), make_env2_terrain_cfg()),
    rl_cfg=env2_ppo_runner_cfg(),
    runner_cls=Env2Runner,
)

__all__ = ["CRAWL_TASK_ID", "ENV1_TASK_ID", "ENV2_TASK_ID", "JUMP_TASK_ID", "NUM_CELLS", "PARKOUR_TASK_ID",
           "attach_parkour_terrain", "task_from_argv"]
