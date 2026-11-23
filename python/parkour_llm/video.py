"""A video of one robot of a run: the robot from outside, beside what the LLM was shown and what it answered.

One frame per high-level step (0.05 s), so the clip plays at the speed of the simulation although the simulation
itself waited for every answer.
"""

import io
import textwrap
from dataclasses import replace
from pathlib import Path

import imageio
import mujoco
import numpy as np
from mjlab.envs import ManagerBasedRlEnv
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig
from PIL import Image, ImageDraw, ImageFont

from .views import depth_png

VIEW = (640, 480)  # The outside view (px).
PANEL = 420  # Width of the panel beside it (px).
# The camera rides behind the robot and a little to its right, looking along the lane at a point above the floor.
CAMERA = dict(distance=1.7, azimuth=25., elevation=-22.)
LOOK_AT_HEIGHT = .15  # m
SHADOW_DISTANCE = 5.4  # The renderer draws shadows up to 1.5 times this from the world origin (m).
MARGIN = 12
TEXT, MUTED, ACCENT, BACKGROUND = (241, 243, 246), (154, 165, 180), (212, 255, 79), (26, 31, 39)


class RunVideo:
    """Writes ``path``; ``frame`` adds one frame, ``close`` finishes the file."""

    def __init__(self, env: ManagerBasedRlEnv, path: Path, env_id: int = 0, title: str = "") -> None:
        self._env, self._env_id, self._title = env, env_id, title
        viewer = replace(env.cfg.viewer, origin_type=ViewerConfig.OriginType.WORLD, width=VIEW[0], height=VIEW[1],
                         max_extra_envs=0, env_idx=env_id, distance=SHADOW_DISTANCE)
        self._renderer = OffscreenRenderer(env.sim.mj_model, viewer, env.scene, sim_model=env.sim.model,
                                           expanded_fields=env.sim.expanded_fields)
        self._renderer.initialize()
        self._camera = mujoco.MjvCamera()
        self._camera.distance, self._camera.azimuth, self._camera.elevation = (
            CAMERA["distance"], CAMERA["azimuth"], CAMERA["elevation"])
        self._font, self._small = ImageFont.load_default(size=15), ImageFont.load_default(size=13)
        self._writer = imageio.get_writer(path, fps=round(1 / (5 * env.step_dt)), quality=8, macro_block_size=1)
        self.path = path

    def frame(self, shown: dict | None, time_s: float) -> None:
        """Add a frame of the current state. ``shown`` is the robot's latest call (``LlmPolicy.last``) or None."""
        root = self._env.scene["robot"].data.root_link_pos_w[self._env_id].cpu().numpy()
        floor = float(self._env.scene.env_origins[self._env_id, 2])
        self._camera.lookat[:] = root[0], root[1], floor + LOOK_AT_HEIGHT
        self._renderer.update(self._env.sim.data, camera=self._camera)
        canvas = Image.new("RGB", (VIEW[0] + PANEL, VIEW[1]), BACKGROUND)
        canvas.paste(Image.fromarray(self._renderer.render()), (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((MARGIN, 8), f"{self._title}   t = {time_s:5.2f} s", font=self._font, fill=(255, 255, 255),
                  stroke_width=2, stroke_fill=(0, 0, 0))
        if shown is not None:
            self._panel(canvas, draw, shown)
        self._writer.append_data(np.asarray(canvas))

    def _panel(self, canvas: Image.Image, draw: ImageDraw.ImageDraw, shown: dict) -> None:
        x, y, width = VIEW[0] + MARGIN, MARGIN, PANEL - 2 * MARGIN
        decision, images = shown["decision"], shown["images"]
        heading = f"CALL {shown['call'] + 1}: WHAT THE MODEL WAS SHOWN"
        if not images:
            # A model that reads the frame as a table of numbers (Jev): show the frame it was made from.
            if "preview" not in shown:
                shown["preview"] = [("depth", depth_png(shown["depth"]))]
            images = shown["preview"]
            heading = f"CALL {shown['call'] + 1}: THE FRAME, SENT AS A TABLE OF NUMBERS"
        draw.text((x, y), heading, font=self._small, fill=MUTED)
        y += 20
        # The pictures side by side, in the order they were sent.
        each = (width - MARGIN * (len(images) - 1)) // max(len(images), 1)
        height = 0
        for i, (_, png) in enumerate(images):
            picture = Image.open(io.BytesIO(png)).convert("RGB")
            height = round(each * picture.height / picture.width)
            canvas.paste(picture.resize((each, height), Image.NEAREST), (x + i * (each + MARGIN), y))
        y += height + 14
        draw.text((x, y), "WHAT IT ANSWERED", font=self._small, fill=MUTED)
        y += 20
        if decision.error:
            lines, colour = textwrap.wrap(f"No usable answer: {decision.error}", 54), (255, 138, 128)
        else:
            lines, colour = textwrap.wrap(decision.scene, 54)[:6], TEXT
        for line in lines:
            draw.text((x, y), line, font=self._small, fill=colour)
            y += 17
        y += 8
        jump = "no" if decision.jump_at_step < 0 else f"on step {decision.jump_at_step}"
        for name, value in (("speed", f"{decision.speed:.2f} m/s"), ("height", f"{100 * decision.height:.1f} cm"),
                            ("jump", jump)):
            draw.text((x, y), name, font=self._font, fill=MUTED)
            draw.text((x + 70, y), value, font=self._font, fill=ACCENT)
            y += 21
        if decision.note:
            y += 6
            for line in textwrap.wrap(f"note: {decision.note}", 54)[:4]:
                if y > VIEW[1] - 20:
                    break
                draw.text((x, y), line, font=self._small, fill=MUTED)
                y += 17

    def close(self) -> None:
        self._writer.close()
        self._renderer.close()
