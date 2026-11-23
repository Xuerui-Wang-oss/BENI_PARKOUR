"""What the LLM is told: the system prompt, the per-decision observation text, and the reply schema.

The system prompt carries what the PPO policy learned by trial over thousands of iterations and the LLM cannot
know: what each obstacle needs from the two low-level skills, and the timing of a jump. The numbers come from the
probes recorded in ``parkour_rl.check`` and ``parkour_rl.hierarchy``. It does not say which colour an obstacle has.
"""

from beni_rl_crawl.beni_robot import BENI_POSITION_OBS_JOINTS, BENI_VELOCITY_OBS_JOINTS
from beni_rl_crawl.recovery import H_LOW, H_REF
from parkour_rl.hierarchy import HEIGHT_SCALE, VX_CENTER, VX_RANGE, VX_SCALE

from .views import FAR, NEAR

STEP_S = 0.05  # One high-level step (s).
VIEWS = ("depth", "rgb", "rgbd", "table")  # "table": the depth table alone, no picture.
# Versions of the instructions. "v1" is the text of the runs up to 2026-11-22 (RESULTS.md). "v2" differs in the
# places its failures pointed at:
#   - the jump distances aim at the middle of the windows that scripted triggers cleared on the env1 lanes at
#     1.0 m/s (root to obstacle: a pit of every length 0.40-0.55 m, a box and a bar of every height 0.55-0.70 m),
#     given from the camera, 0.07 m ahead of the root, and stated as windows;
#   - a box and a bar get one rule, since from the front a tall box and a bar look alike;
#   - when to schedule a trigger and when to wait for the next, closer frame;
#   - where a pit's edge lies between two rows of the table;
#   - how the lane's end wall differs from an obstacle, and that the goal distance is capped.
PROMPTS = ("v1", "v2")
DEFAULT_PROMPT = "v2"

# The reply. Structured outputs take no numeric bounds, so the policy clamps the numbers; -1 stands for "no jump".
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "scene": {"type": "string"},
        "speed": {"type": "number"},
        "height": {"type": "number"},
        "jump_at_step": {"type": "integer"},
        "note": {"type": "string"},
    },
    "required": ["scene", "speed", "height", "jump_at_step", "note"],
    "additionalProperties": False,
}

_ROLE = """\
You are the high-level controller of BENI, a small wheel-legged robot driving along a straight, walled obstacle \
lane in a physics simulation. The robot balances on two wheels at the end of two legs; its body rides about \
{h_ref:.2f} m above the floor when it stands. Lower-level controllers keep it balanced, steer it along the centre \
of the lane and move the joints. You decide three things: how fast to drive, how low to carry the body, and when \
to jump.

You are called once every {period_s:.2f} s of simulated time, and the simulation is paused while you answer. Each \
call shows you only the current camera frame and body state, your own last few decisions, and the note you left \
yourself, so anything you need to remember has to go into that note.

## The task

Reach the goal at the end of the lane and stop there: within 0.3 m of it, slower than 0.2 m/s, upright, held for \
0.5 s. The episode fails if the robot falls over, drops into a pit, advances less than 0.1 m in 3 s (stuck against \
something), keeps its body or legs pressed against anything for 0.5 s, drives more than 1 m past the goal, or runs \
out of time. The time budget is met by averaging about 0.6 m/s, so keep moving; cruising at 1.0 m/s is normal.\
{goal_note}

## Your commands

- speed: forward speed, {vx_low:.1f} to {vx_high:.1f} m/s. The robot cannot reverse.
- height: body height, {h_low:.3f} m (lowest crouch) to {h_ref:.3f} m (standing). Lowering or raising takes up to \
0.5 s.
- jump_at_step: each call covers the next {period} steps of {step_s} s, numbered 0 to {last}. Give the step on \
which to trigger a jump, or -1 for no jump.

How a jump works: from the trigger the robot stands up to full height, keeps driving at the commanded speed, and \
leaves the floor 0.2 s later, so it travels speed x 0.2 m between trigger and take-off. The body rises about 0.2 m. \
A trigger does nothing while a jump is in progress or for roughly a second after landing; the body state says \
whether a trigger would be accepted now. Do not jump without a reason: every jump risks a bad landing.

## Obstacles

Each obstacle spans the full width of the lane, and obstacles are 2 to 4 m apart. The distances below are \
measured from the camera, like every distance you are shown.

{jumps}\
- Tunnel: a slab 0.3 m long hanging across the lane, its underside 0.20 to 0.26 m above the floor. The standing \
robot does not fit. Lower the body to {h_low:.3f} m by the time the slab is 0.7 m ahead and stay low until about \
0.5 s after you are under it. Once you are under the slab the camera no longer sees it.
- Ramp: a slope of 8 to 20 degrees up to a platform 0.15 m high and 1 m long, with a step back down to the floor. \
Drive up and over at standing height. No jump.
{timing}
Near the goal, slow down: a speed of about 2 x (distance to goal - 0.1) m/s, capped at your cruise speed, stops the \
robot inside the goal window.

## The camera

The camera sits at the front of the body, about 0.10 m above the floor when the robot stands, and looks straight \
ahead: {fov_v:.0f} degrees of vertical field of view, {columns} x {rows} pixels, enlarged for you without \
smoothing. It shows \
only the lane (floor, side walls, obstacles), never the robot. With the body level, the horizon is the middle row \
of the image, and a point on the floor d metres ahead appears 0.053 / d of the image height below the middle row: \
about 11% below at 0.5 m, 5% at 1 m, 2.7% at 2 m. Distant floor is therefore squeezed into very few rows, and a \
pit only becomes distinguishable when it is close. When the body pitches, during a jump or on a ramp, the whole \
view shifts up or down.
"""

_GOAL_NOTE = {
    "v1": "",
    "v2": " The goal distance you are shown is capped: it reads 2.0 while the goal is 2 m away or farther and is "
          "the true distance only below that, so 2.0 means keep driving.",
}
_JUMPS = {
    "v1": """\
- Pit: a gap in the floor, 0.15 to 0.35 m long and 0.25 m deep. Jump across. At 1.0 m/s it works when the trigger \
comes with the near edge about 0.38 m ahead.
- Box: a block 1 m long and 0.06 to 0.12 m tall. Jump onto it, drive along the top and off the far end. At 1.0 m/s, \
trigger when its front face is about 0.53 m ahead.
- Bar: a low wall only 0.1 m long and 0.06 to 0.12 m tall. Jump over it. It is cleared best at 1.2 m/s with the \
trigger when it is about 0.62 m ahead.
""",
    "v2": """\
- Pit: a gap in the floor, 0.15 to 0.35 m long and 0.25 m deep. Jump across. At 1.0 m/s the jump works when the \
trigger comes with the near edge between 0.33 and 0.48 m ahead. Aim for 0.40 m.
- Box: a block 1 m long and 0.06 to 0.12 m tall. Jump onto it, drive along the top and off the far end.
- Bar: a low wall only 0.1 m long and 0.06 to 0.12 m tall. Jump over it.
- A box and a bar take the same jump, at the same speed: at 1.0 m/s it works when the trigger comes with the front \
face between 0.48 and 0.63 m ahead. Aim for 0.55 m. From the front a tall box and a bar look alike, and you do not \
need to tell them apart before the jump.
""",
}
_TIMING = {
    "v1": "",
    "v2": """
Timing a trigger: at 1.0 m/s the robot covers 0.05 m per step. If the obstacle is at distance d and the aim point \
is a, the trigger belongs on step (d - a) / 0.05, rounded. If that step is greater than {last}, it lies beyond \
this call: give -1 and decide on the next call, which sees the obstacle closer and more exactly. Never place a \
trigger from a reading that the next call can still improve. If d is already at or inside the aim point and the obstacle is still ahead, trigger on \
step 0. Keep the speed steady from two calls before a jump until the robot has landed.
""",
}
# How the table shows a pit's edge and the lane's end wall, per version and table style.
_PIT_EDGE = {
    ("v1", "camera"): "The last normal floor row before the jump is just in front of the near edge.",
    ("v1", "floor"): "The near edge is just beyond the last row that still reads floor level.",
    "v2": "The near edge lies between the last row that still reads floor level and the place where the next row up "
          "would have met open floor: about 0.05 m beyond that row when it reads 0.4 m, and 0.10 m beyond when it "
          "reads 0.6 m. Take the middle of that span as the edge distance.",
}
_END_WALL = {
    "v1": "",
    "v2": {
        "camera": """\
- The end wall of the lane: a return that climbs from floor level up past the camera's height. It is many metres \
away for most of the run and comes close only after the last obstacle, when the goal distance reads below 2.0. \
While the goal distance reads 2.0, anything across the lane within 2 m is an obstacle, never the end wall, and the \
side walls do not appear in the table at that range: a face that rises from the floor and stops at about the \
camera's height (+0.03 or lower) is a box or a bar, and a return at +0.09 or higher with floor running on beneath \
it is the tunnel slab.
""",
        "floor": """\
- The end wall of the lane: a return that climbs from floor level up past the camera's height. It is many metres \
away for most of the run and comes close only after the last obstacle, when the goal distance reads below 2.0. \
While the goal distance reads 2.0, anything across the lane within 2 m is an obstacle, never the end wall, and the \
side walls do not appear in the table at that range: a face that rises from the floor and stops at +0.13 or lower \
is a box or a bar, and a return at +0.19 or higher with floor level running on beneath it is the tunnel slab.
""",
    },
}

_DEPTH_IMAGE = """
Depth image: each pixel is coloured by its forward distance on a logarithmic scale, red at {near} m or closer, \
through yellow and green, to blue at {far:.0f} m or farther; black means nothing was hit.
"""

_DEPTH_TABLE = """
Depth table: the central strip of the depth frame as numbers, one line per image row from the top. elev_deg is the \
row's angle above (+) or below (-) the camera's forward axis, dist_m the horizontal distance to what that row \
shows, and height_m that point's height relative to the camera. Both are already corrected for the body's pitch, \
so flat floor reads a constant height of about -0.10 at every distance (closer to zero when crouched). Read the \
table from the bottom (nearest floor) upwards:
- Something standing on the floor (box, bar, the foot of a ramp): several rows share one distance while their \
height climbs above floor level. The shared distance is the distance to its face.
- A pit: only visible within about 0.8 m. The distance jumps between two neighbouring rows by more than the \
floor's own spacing, and the rows after the jump read 0.02 m or more below floor level. {pit_edge} Rows that are \
far away and slightly low are not a pit.
- Something overhead (the tunnel slab): rows return a height of +0.09 or more, which is 0.19 m or more above the \
floor, at a distance under about 2.5 m, while the rows beneath them still show floor at floor level running on \
underneath to a greater distance. The smallest such distance is the distance to the slab. Open lane has no \
return up there at all, or only the far end wall many metres away. The top of a box or the platform of a ramp \
reads +0.05 at most and stands on the floor: it is not a slab and never a reason to crouch.
{end_wall}"""

_DEPTH_TABLE_FLOOR = """
Depth table: the central strip of the depth frame as numbers, one line per image row from the top. elev_deg is the \
row's angle above (+) or below (-) the camera's forward axis and dist_m the horizontal distance to what that row \
shows. above_floor_m is that point's height above the floor right in front of the robot: 0.00 is floor level, a \
negative number is lower than the floor, a positive one is higher. step_m is how much farther this row reaches \
than the row beneath it. Everything is already corrected for the body's pitch. Read the table from the bottom \
(nearest floor) upwards:
- Open floor: above_floor_m stays within about 0.02 of zero and step_m grows smoothly from row to row.
- Something standing on the floor (box, bar, the foot of a ramp): several rows share one distance, so step_m is \
about 0, while above_floor_m climbs to 0.05 or more. The shared distance is the distance to its face.
- A pit: only visible within about 0.8 m. One to three rows read -0.02 or lower, and the step_m into the first of \
them is much larger than the steps beneath it. {pit_edge} Rows several metres away that read a little low are not \
a pit.
- Something overhead (the tunnel slab): rows read +0.19 or higher at a distance under about 2.5 m while the rows \
beneath them still show floor level running on underneath to a greater distance. The smallest such distance is \
the distance to the slab. Open lane has no return up there at all, or only the far end wall many metres away. \
The top of a box or the platform of a ramp reads +0.15 at most and stands on the floor: it is not a slab and \
never a reason to crouch.
{end_wall}"""

_RGB = """
Colour image: the floor and side walls are shades of grey and the space beyond the lane is black. It carries no \
distance information of its own; judge distance from where things meet the floor, using the camera geometry above.
"""

_REPLY = """
## Your reply

- scene: one or two sentences on what is ahead and your estimate of how far, in metres.
- speed, height, jump_at_step: the commands for the next {period} steps.
- note: at most 30 words to your next call, for example that you are still under a slab and must stay low for \
one more call, or that an obstacle was just passed.
"""


def system_prompt(view: str, period: int, size: tuple[int, int] = (60, 80), reply: bool = True,
                  table: str = "camera", version: str = DEFAULT_PROMPT) -> str:
    """The system prompt for ``view`` (one of VIEWS) with one call every ``period`` high-level steps.

    ``size`` is the camera frame as (rows, columns). ``reply=False`` leaves out the section on the reply's fields,
    for a model whose answer is typed questions instead of one JSON object (``backends.JevBackend``). ``table`` is
    the style of the depth table (``views.TABLE_STYLES``) and ``version`` one of PROMPTS.
    """
    text = _ROLE.format(goal_note=_GOAL_NOTE[version], jumps=_JUMPS[version], timing=_TIMING[version].format(last=period - 1),
                        h_ref=H_REF, h_low=H_LOW, vx_low=VX_RANGE[0], vx_high=VX_RANGE[1], period=period,
                        last=period - 1, step_s=STEP_S, period_s=period * STEP_S, fov_v=86.5,
                        rows=size[0], columns=size[1])
    if view in ("depth", "rgbd"):
        text += _DEPTH_IMAGE.format(near=NEAR, far=FAR)
    if view in ("depth", "rgbd", "table"):
        pit_edge = _PIT_EDGE["v2"] if version == "v2" else _PIT_EDGE[version, table]
        end_wall = _END_WALL[version][table] if version == "v2" else ""
        template = _DEPTH_TABLE if table == "camera" else _DEPTH_TABLE_FLOOR
        text += template.format(pit_edge=pit_edge, end_wall=end_wall)
    if view in ("rgb", "rgbd"):
        text += _RGB
    return text + _REPLY.format(period=period) if reply else text


def _numbers(values, digits: int = 2) -> str:
    return ", ".join(f"{float(value):+.{digits}f}" for value in values)


def body_state_text(hl_actor, goal_clip: tuple[float, float]) -> str:
    """The 24 numbers of the ``hl_actor`` observation of one env, named, in the order the PPO policy gets them.

    ``hl_actor`` is a sequence of 24 floats (``parkour_rl.env._actor_terms``); ``goal_clip`` is the range that the
    env clips the goal distance to. The last command is converted back to m/s and m; nothing else is changed.
    """
    v = [float(x) for x in hl_actor]
    speed = VX_CENTER + VX_SCALE * v[18]
    height = H_REF + HEIGHT_SCALE * v[19]
    goal = v[23]
    goal_note = f" (reads {goal_clip[1]:.1f} while the goal is farther than that)" if goal >= goal_clip[1] else ""
    yes = lambda flag: "yes" if flag > 0.5 else "no"  # noqa: E731
    return "\n".join((
        f"angular velocity roll, pitch, yaw (rad/s): {_numbers(v[0:3])}",
        f"gravity direction in the body frame x, y, z (0, 0, -1 when level): {_numbers(v[3:6])}",
        f"joint positions {', '.join(BENI_POSITION_OBS_JOINTS)} (rad from default): {_numbers(v[6:11])}",
        f"joint speeds {', '.join(BENI_VELOCITY_OBS_JOINTS)} (rad/s; LW and RW are the wheels): {_numbers(v[11:18], 1)}",
        f"last command: speed {speed:.2f} m/s, height {height:.3f} m",
        f"jump in progress: {yes(v[20])}; trigger pending: {yes(v[21])}; a trigger now would be accepted: {yes(v[22])}",
        f"distance to goal (m): {goal:+.2f}{goal_note}",
    ))


def observation_text(view: str, body: str, table: str | None, history: list[dict], note: str,
                     new_episode: bool) -> str:
    """The text of one call: recent decisions, the note, the body state and, for a depth view, the table."""
    parts = []
    if new_episode:
        parts.append("A new episode has just started: the robot is at the start of a lane, standing still.")
    if history:
        lines = []
        for age, past in enumerate(reversed(history), start=1):
            jump = "no jump" if past["jump_at_step"] < 0 else f"jump at step {past['jump_at_step']}"
            lines.append(f"- {age} call{'s' if age > 1 else ''} ago: speed {past['speed']:.2f}, height "
                         f"{past['height']:.3f}, {jump}. Saw: {past['scene']}")
        parts.append("Your recent decisions, newest first:\n" + "\n".join(lines))
    parts.append(f"Your note to yourself: {note or '(none)'}")
    parts.append("Body state:\n" + body)
    if table is not None:
        parts.append("Depth table:\n" + table)
    attached = {"depth": "The depth image is attached. ", "rgb": "The colour image is attached. ", "table": "",
                "rgbd": "The colour image and the depth image are attached, in that order. "}[view]
    parts.append(attached + "Decide the next commands.")
    return "\n\n".join(parts)
