"""Camera frames in the forms the LLM is given: PNG pictures and a table of distances.

The depth frame is planar depth in metres (distance along the camera's forward axis), 0 where a ray hit nothing.
The camera looks straight ahead from the front of the body, so with the body level a depth value is the forward
distance to the surface that pixel shows.
"""

import io
import math

import numpy as np
from matplotlib import colormaps
from PIL import Image

from .scene import CAMERA_FOVY

NEAR, FAR = 0.1, 6.0  # Range of the depth picture's colour scale (m), logarithmic.
PICTURE_HEIGHT = 480  # Pixel rows of a picture as sent: frames are enlarged, never smoothed.
TABLE_ROWS = 60  # Most image rows a table lists; taller frames are subsampled.
CENTRE_FRACTION = 1 / 8  # Width of the central strip the table reads, as a fraction of the image width.


def _png(pixels: np.ndarray) -> bytes:
    """``pixels`` (rows, columns, 3) uint8 as a PNG, enlarged to PICTURE_HEIGHT rows with hard pixel edges."""
    image = Image.fromarray(pixels)
    scale = max(1, PICTURE_HEIGHT // image.height)
    buffer = io.BytesIO()
    image.resize((image.width * scale, image.height * scale), Image.NEAREST).save(buffer, format="PNG")
    return buffer.getvalue()


def rgb_png(rgb: np.ndarray) -> bytes:
    """The colour frame (rows, columns, 3) uint8 as a PNG."""
    return _png(np.ascontiguousarray(rgb))


def depth_png(depth: np.ndarray) -> bytes:
    """The depth frame (rows, columns) in metres as a colour-coded PNG.

    Red is NEAR or closer, blue is FAR or farther, on a logarithmic scale; black is "no return".
    """
    hit = depth > 0
    level = 1.0 - np.log(np.clip(depth, NEAR, FAR) / NEAR) / math.log(FAR / NEAR)
    pixels = (colormaps["turbo"](level)[..., :3] * 255).astype(np.uint8)
    pixels[~hit] = 0
    return _png(pixels)


def pitch_from_gravity(gravity) -> float:
    """Nose-up pitch of the body (rad) from the direction of gravity in the body frame, (0, 0, -1) when level."""
    return math.atan2(-float(gravity[0]), -float(gravity[2]))


TABLE_STYLES = ("camera", "floor")
FLOOR_ROWS = 6  # Rows at the bottom of the image, the floor right in front of the robot, that set the floor level.


def depth_table(depth: np.ndarray, pitch: float = 0., style: str = "camera") -> str:
    """The central strip of the depth frame as text, one line per image row from the top.

    ``style`` "camera" gives the columns described below. "floor" gives the same rows with the height measured
    from the floor in front of the robot instead of from the camera (0.00 is floor level; the level is the median
    height of the FLOOR_ROWS lowest rows of this same frame) and a fourth column, how much farther each row reaches
    than the row beneath it. Both styles hold the same information, all of it from the one depth frame.

    Columns: the row's elevation angle (deg, negative below the camera's forward axis), the horizontal distance
    (m) to what that row shows, and that point's height relative to the camera (m). Both are the pinhole
    unprojection of the pixel, turned by ``pitch`` (rad, nose up) so that they are level with the floor whatever
    the body does: flat floor reads a constant height, minus the camera's height above the floor, at every
    distance. Runs of rows without a return are folded into one line.
    """
    rows, columns = depth.shape
    focal = (rows / 2) / math.tan(math.radians(CAMERA_FOVY) / 2)  # Pixels; 31.875 for 60 rows.
    half = max(1, round(columns * CENTRE_FRACTION / 2))
    strip = depth[:, columns // 2 - half: columns // 2 + half]
    entries = []  # Top to bottom: (elevation in deg, distance, height) of a row, or None for a row without a return.
    for row in range(0, rows, max(1, rows // TABLE_ROWS)):
        below = (row + 0.5 - rows / 2) / focal  # Tangent of the angle below the forward axis.
        values = strip[row][strip[row] > 0]
        if values.size < strip.shape[1] / 2:
            entries.append(None)
            continue
        along = float(np.median(values))  # Distance along the camera axis; the point lies along * below under it.
        entries.append((-math.degrees(math.atan(below)), along * (math.cos(pitch) + below * math.sin(pitch)),
                        along * (math.sin(pitch) - below * math.cos(pitch))))
    hits = [entry for entry in entries if entry is not None]
    floor = float(np.median([height for _, _, height in hits[-FLOOR_ROWS:]])) if hits else 0.

    header = "elev_deg  dist_m  height_m" if style == "camera" else "elev_deg  dist_m  above_floor_m  step_m"
    lines, empty = [header], 0
    for index, entry in enumerate(entries):
        if entry is None:
            empty += 1
            continue
        if empty:
            lines.append(f"({empty} rows: nothing in range)")
            empty = 0
        elevation, distance, height = entry
        if style == "camera":
            lines.append(f"{elevation:+6.1f}  {distance:6.2f}  {height:+6.2f}")
        else:
            beneath = entries[index + 1] if index + 1 < len(entries) else None
            step = f"{distance - beneath[1]:+6.2f}" if beneath is not None else "      "
            lines.append(f"{elevation:+6.1f}  {distance:6.2f}  {height - floor:+13.2f}  {step}")
    if empty:
        lines.append(f"({empty} rows: nothing in range)")
    return "\n".join(lines)
