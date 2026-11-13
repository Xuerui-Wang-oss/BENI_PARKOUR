# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""Training videos (``train --video``) that show one robot per lane column, tiled in two rows.

mjlab's VideoRecorder films one env, env 0, which is always the pit column in training. This recorder films the first
env of every obstacle kind (one lane column each) with its own camera, cfg.viewer with that env and no neighbours,
and tiles the views in two rows of ceil(K / 2) tiles, K kinds in the order of the goal term's kinds, with a blank
tile for an odd K. Beni-Parkour: pit top left, cube top right, hollow bottom left, ramp_cube bottom right. env1: pit,
cube, hollow on top, ramp_cube, thin_cube and a blank tile below. Each tile is labelled with its kind, the row the env
is on, and the env id. env2, whose lanes mix kinds: envs 0-5, labelled with their lane column and the row (the layout)
they are on. ``train.py`` installs it in place of mjlab's recorder.
"""

import math
from dataclasses import replace

import numpy as np
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from PIL import Image, ImageDraw

MIXED_TILES = 6  # Tiles of a frame where the lanes mix kinds (env2): two rows of three.


class ParkourVideoRecorder(VideoRecorder):
    """mjlab's VideoRecorder, but every frame holds one tile per lane column, each following one robot."""

    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        sim, scene = env.unwrapped.sim, env.unwrapped.scene
        goal = env.unwrapped.command_manager.get_term("goal")
        if goal.column_kind is None:
            # Lanes of mixed kinds (env2): the first MIXED_TILES envs, which run MIXED_TILES different lane columns.
            self._env_ids = list(range(min(MIXED_TILES, env.unwrapped.num_envs)))
            self._kinds = tuple(f"lane {int(scene.terrain.terrain_types[i])}" for i in self._env_ids)
        else:
            self._kinds = goal.kinds
            kind_of_env = goal.column_kind[scene.terrain.terrain_types]
            # First env of each kind (env k in training, where columns are interleaved); None for a kind without env.
            self._env_ids = [int(ids[0]) if len(ids := (kind_of_env == k).nonzero().flatten()) else None
                             for k in range(len(self._kinds))]
        viewer = env.unwrapped.cfg.viewer
        self._renderers = []
        for env_id in self._env_ids:
            renderer = None
            if env_id is not None:
                renderer = OffscreenRenderer(sim.mj_model, replace(viewer, env_idx=env_id, max_extra_envs=0), scene,
                                             sim_model=sim.model, expanded_fields=sim.expanded_fields)
                renderer.initialize()
            self._renderers.append(renderer)
        self._blank = np.zeros((viewer.height, viewer.width, 3), np.uint8)

    def _record_frame(self) -> None:
        """Append one frame of two rows of tiles to the clip being recorded (replaces mjlab's single-env frame)."""
        env = self.unwrapped
        tiles = []
        for kind, env_id, renderer in zip(self._kinds, self._env_ids, self._renderers):
            if renderer is None:
                tiles.append(self._blank)
                continue
            renderer.update(env.sim.data, debug_vis_callback=env.update_visualizers)
            tile = Image.fromarray(renderer.render())
            row = int(env.scene.terrain.terrain_levels[env_id])
            ImageDraw.Draw(tile).text((4, 2), f"{kind}  row {row}  env {env_id}", fill=(255, 255, 255),
                                      stroke_width=2, stroke_fill=(0, 0, 0))
            tiles.append(np.asarray(tile))
        # The bottom row is filled up with blank tiles for an odd number of kinds.
        columns = math.ceil(len(tiles) / 2)
        tiles += [self._blank] * (2 * columns - len(tiles))
        self.current_video_frames.append(np.concatenate([np.concatenate(tiles[:columns], 1),
                                                         np.concatenate(tiles[columns:], 1)], 0))

    def close(self) -> None:
        super().close()
        for renderer in self._renderers:
            if renderer is not None:
                renderer.close()
