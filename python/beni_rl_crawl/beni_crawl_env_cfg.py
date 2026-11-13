# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT
"""Crawl training: velocity tracking and recovery with base height commands down to a deep crouch."""
from mjlab.managers import RewardTermCfg

from .beni_velocity_flat_env_cfg import make_beni_velocity_flat_env_cfg
from .recovery import configure_recovery_rewards, height_error_l1, H_LOW, H_REF


def make_beni_crawl_env_cfg(play=False):
    cfg = make_beni_velocity_flat_env_cfg(play=play)
    configure_recovery_rewards(cfg)
    # Linear height term keeps a pull in the last centimetre, where the squared term fades.
    cfg.rewards["base_height_l1"] = RewardTermCfg(func=height_error_l1, weight=-50., params={})
    # Height commands reach the deepest level crouch. The standing height is the top of the
    # range because the posture reference does not move above it.
    cfg.commands['twist'].height_range = (H_LOW, H_REF)
    return cfg
