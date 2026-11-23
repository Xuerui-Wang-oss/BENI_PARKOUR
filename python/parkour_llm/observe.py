"""One decision request from one robot's observation: the same for the Python runs and for the web page."""

import numpy as np

from .backends import Request
from .prompt import DEFAULT_PROMPT, body_state_text, observation_text, system_prompt
from .views import depth_png, depth_table, pitch_from_gravity, rgb_png


def build_request(view: str, period: int, hl_actor, depth: np.ndarray | None, rgb: np.ndarray | None,
                  history: list[dict], note: str, new_episode: bool, goal_clip: tuple[float, float],
                  table_style: str = "camera", prompt: str = DEFAULT_PROMPT,
                  ) -> tuple[Request, list[tuple[str, bytes]]]:
    """The request for one robot and the pictures in it as (name, PNG) pairs, in the order they are sent.

    ``hl_actor`` is the 24-number state of the high-level policy; ``depth`` (rows, columns) in metres and ``rgb``
    (rows, columns, 3) uint8 are the camera frames, each needed only by the views that show it. ``history`` holds
    the robot's own latest decisions, oldest first, as ``Decision.commands()`` dicts. ``table_style`` picks how the
    depth table is written (``views.TABLE_STYLES``) and ``prompt`` the version of the instructions
    (``prompt.PROMPTS``).
    """
    frame = depth if depth is not None else rgb
    images, table = [], None
    if view in ("rgb", "rgbd"):
        images.append(("rgb", rgb_png(rgb)))
    if view in ("depth", "rgbd"):
        images.append(("depth", depth_png(depth)))
    if view in ("depth", "rgbd", "table"):
        table = depth_table(depth, pitch_from_gravity(hl_actor[3:6]), table_style)
    text = observation_text(view, body_state_text(hl_actor, goal_clip), table, history, note, new_episode)
    system = system_prompt(view, period, (frame.shape[0], frame.shape[1]), table=table_style, version=prompt)
    return Request(system, text, [png for _, png in images]), images
