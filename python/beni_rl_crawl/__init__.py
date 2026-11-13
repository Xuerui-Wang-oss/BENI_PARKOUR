# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""Register the BENI crawl task for mjlab.

This package is beni_rl with the changes needed to fine-tune the drive+jump policy into a
drive+crawl policy: the wheel-thigh collision pair removed, the knee free up to its hard limit,
height commands down to a deep crouch, and rewards that follow the commanded height.
"""

import os
from pathlib import Path

# Variable-length GRU batches otherwise leave large cached CUDA allocations
# that Warp cannot reclaim. Set the default before loading torch via mjlab.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "assets"
BENI_MJCF = ASSETS_DIR / "beni_base" / "mjcf" / "beni.xml"

from mjlab.tasks.registry import register_mjlab_task  # noqa: E402

from .beni_robot import get_beni_robot_cfg  # noqa: E402
from .beni_crawl_env_cfg import make_beni_crawl_env_cfg  # noqa: E402
from .rl_cfg import BeniVelocityOnPolicyRunner, beni_crawl_ppo_runner_cfg  # noqa: E402

TASK_ID = "Beni-Crawl-Flat"

register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=make_beni_crawl_env_cfg(),
    play_env_cfg=make_beni_crawl_env_cfg(play=True),
    rl_cfg=beni_crawl_ppo_runner_cfg(),
    runner_cls=BeniVelocityOnPolicyRunner,
)

__all__ = [
    "ASSETS_DIR",
    "BENI_MJCF",
    "REPO_ROOT",
    "TASK_ID",
    "get_beni_robot_cfg",
    "make_beni_crawl_env_cfg",
    "beni_crawl_ppo_runner_cfg",
]
