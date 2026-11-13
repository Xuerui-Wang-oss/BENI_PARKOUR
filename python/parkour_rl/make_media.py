# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""Clips and a picture of the parkour project for a README, written to parkour_rl/media.

    python -m parkour_rl.make_media skills     # the two frozen low-level policies on a flat floor: four clips
    python -m parkour_rl.make_media stages     # three training videos of the env1 run, one per row
    python -m parkour_rl.make_media layouts    # the four training layouts of env2, drawn as a stack
    python -m parkour_rl.make_media results    # the final policy on the test layout of env2: six lanes in one clip

Run it from the ``python`` directory. Every clip is written as .mp4 and as .gif, which a README can show in place.

skills    The hierarchy of the parkour task (hierarchy.ParkourVecEnvWrapper) with scripted commands in place of the
          high-level policy: the frozen crawl policy drives and crouches, the frozen jump policy jumps. One robot per
          clip, all in one simulation, on a flat env1 lane that is drawn as an open checker floor.
stages    The videos that ``train --video`` recorded during the env1 run, STAGES of them. Each is one row of the clip:
          its tiles STAGE_TILES (pit, cube, hollow, ramp_cube) under a line with its PPO iteration and the curriculum
          rows its robots were on.
layouts   The lanes of env2 as the terrain generator builds them, the training layouts moved on top of each other.
          The lanes are drawn up to the nearest goal of the longest one; behind it every lane is empty.
results   One episode of RESULT_CHECKPOINT, with deterministic actions, on each of the 16 lanes of the test layout of
          env2, which training never used. The outcome of every lane is printed. The clip shows six of them at once,
          each until its episode ends: lanes on which the robot reached the goal, the first six in lane order, the
          two lanes at the edge of the terrain counted last.
"""

import math
import subprocess
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import imageio
import imageio_ffmpeg
import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import EventTermCfg
from mjlab.terrains import TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import TerrainGenerator
from mjlab.utils.spec_config import LightCfg, MaterialCfg, TextureCfg
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig
from PIL import Image, ImageDraw, ImageFont

from beni_rl_crawl.recovery import H_LOW, H_REF

from . import hierarchy
from .check import to_action
from .curriculum import CONTROL_STEPS_PER_ITERATION
from .env1 import ENV1_LANE, make_env1_cfg
from .env2 import (ENV2_LANES, ENV2_TRAIN_LAYOUTS, Env2Runner, env2_ppo_runner_cfg, make_env2_cfg,
                   make_env2_terrain_cfg)
from .evaluate import record_episodes
from .hierarchy import SUB_STEPS, ParkourVecEnvWrapper
from .parkour_terrain import BLUE, DARK, GOAL_MARGIN, GREEN, LANE, ORANGE, RED, ParkourLaneCfg

MEDIA = Path(__file__).parent / "media"

# skills
FRAME_SIZE = (1280, 720)  # px
FRAME_STEPS = 2  # Control steps (0.01 s) per frame: 50 frames per second.
SKILL_GIF_WIDTH = 480  # px
LEAD_IN_S = 1.5  # Simulated before the first frame, so that a driving clip starts at speed.
CLIP_S = 4.5
# Clip -> speed command (m/s), times of the jump triggers (s into the clip), start and end of the crouch command (s
# into the clip; None: no crouch).
CLIPS = {
    "skill_jump": (0., (.4, 2.4), None),
    "skill_crawl": (0., (), (.8, 3.)),
    "skill_jump_drive": (1., (.4, 2.4), None),
    "skill_crawl_drive": (.8, (), (.8, 3.)),
}
# The camera follows the robot over the floor at a fixed height, so a jump is seen against a still horizon.
CAMERA = dict(distance=.95, azimuth=125., elevation=-14.)
LOOK_AT_HEIGHT = .2  # m
# The renderer draws shadows up to 1.5 times the viewer's distance setting from the world origin, where the lane has
# its centre: with 5.4 m, over the whole lane. The camera of the clips is CAMERA; nothing else reads the setting.
SHADOW_DISTANCE = 5.4

# stages
STAGE_VIDEOS = Path("logs/rsl_rl/beni_parkour_env1/2026-11-05_22-54-04/videos/train")
# Control step at which a video starts (its file name), and the curriculum rows its robots are on (the tile labels).
STAGES = ((0, "curriculum rows 0-1"), (30_000, "curriculum row 4 (hardest)"),
          (330_000, "curriculum row 4 (hardest)"))
STAGE_FRAME_STEP = 4  # The videos have 100 frames per second; every fourth is kept.
# px; a video is two rows of three tiles: pit, cube, hollow; ramp_cube, thin_cube and an empty one.
TILE = (320, 240)
STAGE_TILES = (0, 1, 2, 3)  # The tiles that are shown, counted along the rows of a video.
HEADER = 40  # Height of the line above each row of tiles (px).
STAGE_GIF_WIDTH = 800  # px

# results
# The released policy: iteration 1000 of the env2 run 2026-11-10_23-34-36. Relative to the ``python`` directory.
RESULT_CHECKPOINT = "parkour_rl/checkpoints/env2_model_1000.pt"
RESULT_GRID = (2, 3)  # Rows and columns of lanes in the clip.
RESULT_TILE = (640, 480)  # px
RESULT_FRAME_STEPS = 4  # Control steps (0.01 s) per frame: 25 frames per second.
RESULT_CAMERA = dict(distance=2.2, azimuth=90., elevation=-45.)  # The view of the training videos, 0.2 m farther.
# The camera looks at the centre line of the lane, this far ahead of the robot (m): more of the tile shows what comes.
RESULT_LOOK_AHEAD = .7
RESULT_HOLD_S = 1.5  # The clip goes on this long after the last episode has ended.
RESULT_GIF_WIDTH = 960  # px

# layouts
LAYER_GAP = 14.  # Height between two layouts of the stack (m).
VIEW = dict(azimuth=45., elevation=-35.)
PIXELS_PER_METRE = 44
SUPERSAMPLING = 2  # The picture is rendered this many times larger and scaled down, which smooths the edges.
MARGIN = (11., 3., 3., 6.5)  # Left (layout names), top, right, bottom (legend) of the picture (m).
LEGEND = (("pit", DARK), ("cube", ORANGE), ("bar (thin cube)", RED), ("tunnel (hollow)", BLUE),
          ("ramp + cube", GREEN))


def open_video(name: str, fps: float):
    """Writer of media/<name>.mp4: ``append_data`` takes one frame (H, W, 3 uint8), ``close`` finishes the file."""
    MEDIA.mkdir(exist_ok=True)
    return imageio.get_writer(MEDIA / f"{name}.mp4", fps=fps, quality=8, macro_block_size=1)


def write_gif(name: str, fps: float, width: int) -> None:
    """Convert media/<name>.mp4 to a GIF ``width`` px wide with ``fps`` frames per second; print both file sizes."""
    video = MEDIA / f"{name}.mp4"
    filters = (f"fps={fps:g},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];"
               "[b][p]paletteuse=dither=bayer:bayer_scale=5")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video),
                    "-filter_complex", filters, str(video.with_suffix(".gif"))], check=True)
    for path in (video, video.with_suffix(".gif")):
        print(f"{path}  {path.stat().st_size / 1e6:.1f} MB")


class FloorLaneCfg(ParkourLaneCfg):
    """A lane with an open checker floor and a sky to film it on. The lane itself is unchanged, in geom group 0, for
    the physics and the depth camera; the floor is in group 2, and the clips are filmed without group 0."""

    def function(self, difficulty, spec, rng):
        output = super().function(difficulty, spec, rng)
        TextureCfg(name="sky", type="skybox", builtin="gradient", rgb1=(.62, .76, .9), rgb2=(.2, .34, .55),
                   width=512, height=3072).edit_spec(spec)
        TextureCfg(name="floor", type="2d", builtin="checker", mark="edge", rgb1=(.2, .3, .4), rgb2=(.1, .2, .3),
                   markrgb=(.8, .8, .8), width=300, height=300).edit_spec(spec)
        MaterialCfg(name="floor", texuniform=True, texrepeat=(4., 4.), reflectance=.15,
                    texture="floor").edit_spec(spec)
        # Only seen: no contacts. The lane ground has its top at z = 0 too.
        spec.body("terrain").add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0., 0., .01), material="floor",
                                      rgba=(1., 1., 1., 1.), contype=0, conaffinity=0, group=2)
        return output


@torch.inference_mode()
def skills() -> None:
    """Film the clips of CLIPS: env i runs clip i."""
    cfg = make_env1_cfg(play=True)
    cfg.scene.num_envs = len(CLIPS)
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        size=ENV1_LANE, num_rows=1, curriculum=True, seed=0,
        sub_terrains={"floor": FloorLaneCfg(kind="pit", num_rows=1)})  # One row: row 0, which has no obstacle.
    cfg.events["reset_base"].params["pose_range"]["y"] = (0., 0.)
    # No episode may end inside a clip: a robot that jumps on the spot is "stuck", one that drives on is past its goal.
    # goal_reached stays because the rewards read it; it needs the robot at rest at the goal for hold_steps.
    cfg.terminations = {name: cfg.terminations[name] for name in ("time_out", "goal_reached", "nan")}
    cfg.terminations["goal_reached"].params["hold_steps"] = 10 ** 9

    env = ManagerBasedRlEnv(cfg, device="cpu")
    wrapper = ParkourVecEnvWrapper(env)
    # The wrapper ignores a trigger on a lane without obstacles.
    env.command_manager.get_term("goal").on_flat_lane = lambda: torch.zeros(env.num_envs, dtype=torch.bool)

    # Geom groups 1 and 2: the robot and the floor, without the lane (group 0).
    viewer = replace(env.cfg.viewer, origin_type=ViewerConfig.OriginType.WORLD, width=FRAME_SIZE[0],
                     height=FRAME_SIZE[1], max_extra_envs=0, geom_group=(0, 1, 1, 0, 0, 0),
                     distance=SHADOW_DISTANCE)
    renderers = [OffscreenRenderer(env.sim.mj_model, replace(viewer, env_idx=i), env.scene, sim_model=env.sim.model,
                                   expanded_fields=env.sim.expanded_fields) for i in range(env.num_envs)]
    for renderer in renderers:
        renderer.initialize()
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = CAMERA["distance"], CAMERA["azimuth"], CAMERA["elevation"]

    robot = env.scene["robot"]
    fps = 1 / (FRAME_STEPS * env.step_dt)
    videos = [open_video(name, fps) for name in CLIPS]
    heights = []  # Root height above the floor (m) of every env at every frame.
    lead_in_steps = round(LEAD_IN_S / env.step_dt)
    control_steps = 0
    step = env.step

    def step_and_film(action):
        """One control step of the env, then a frame of every clip."""
        nonlocal control_steps
        result = step(action)
        control_steps += 1
        if control_steps > lead_in_steps and control_steps % FRAME_STEPS == 0:
            root = robot.data.root_link_pos_w.cpu().numpy()  # (N, 3)
            heights.append(root[:, 2] - env.scene.env_origins[:, 2].cpu().numpy())
            for i, renderer in enumerate(renderers):
                camera.lookat[:] = root[i, 0], root[i, 1], LOOK_AT_HEIGHT
                renderer.update(env.sim.data, camera=camera)
                videos[i].append_data(renderer.render())
        return result

    env.step = step_and_film  # The wrapper runs SUB_STEPS control steps per high-level step through this.

    dt = SUB_STEPS * env.step_dt  # High-level step (s).
    speed = torch.tensor([clip[0] for clip in CLIPS.values()])
    start_x = robot.data.root_link_pos_w[:, 0].clone()
    for high_level_step in range(round((LEAD_IN_S + CLIP_S) / dt)):
        time = high_level_step * dt - LEAD_IN_S  # s into the clip.
        trigger = torch.tensor([any(abs(time - jump) < dt / 2 for jump in clip[1]) for clip in CLIPS.values()])
        crouch = torch.tensor([clip[2] is not None and clip[2][0] <= time < clip[2][1] for clip in CLIPS.values()])
        wrapper.step(to_action(speed, torch.where(crouch, H_LOW, H_REF), trigger))

    heights = np.stack(heights)  # (frames, N)
    jumps = env.command_manager.get_term("hl_state").metrics["jumps_accepted"]
    driven = robot.data.root_link_pos_w[:, 0] - start_x
    for i, name in enumerate(CLIPS):
        videos[i].close()
        write_gif(name, fps / 2, SKILL_GIF_WIDTH)
        print(f"  jumps {int(jumps[i])}, root height {heights[:, i].min():.3f} to {heights[:, i].max():.3f} m, "
              f"driven {driven[i]:.2f} m (lead-in included)")
    for renderer in renderers:
        renderer.close()
    wrapper.close()


def stages() -> None:
    """Stack the env1 training videos of STAGES, one per row: a header line, then the video's tiles STAGE_TILES."""
    font = ImageFont.load_default(24)
    width = len(STAGE_TILES) * TILE[0]
    # (rows, columns) of every shown tile in a video frame, as slices.
    tiles = [(slice(i // 3 * TILE[1], (i // 3 + 1) * TILE[1]), slice(i % 3 * TILE[0], (i % 3 + 1) * TILE[0]))
             for i in STAGE_TILES]
    headers = []
    for step, rows in STAGES:
        header = Image.new("RGB", (width, HEADER))
        draw = ImageDraw.Draw(header)
        draw.text((12, HEADER // 2), f"Iteration {step // CONTROL_STEPS_PER_ITERATION}", font=font, anchor="lm")
        draw.text((width - 12, HEADER // 2), rows, font=font, anchor="rm", fill=(190, 190, 190))
        headers.append(np.asarray(header))
    readers = [imageio.get_reader(STAGE_VIDEOS / f"rl-video-step-{step}.mp4") for step, _ in STAGES]
    fps = readers[0].get_meta_data()["fps"] / STAGE_FRAME_STEP
    video = open_video("env1_training_stages", fps)
    for index, frames in enumerate(zip(*readers)):
        if index % STAGE_FRAME_STEP:
            continue
        video.append_data(np.concatenate([part for header, frame in zip(headers, frames) for part in
                                          (header, np.concatenate([frame[tile] for tile in tiles], axis=1))]))
    video.close()
    for reader in readers:
        reader.close()
    write_gif("env1_training_stages", fps / 2, STAGE_GIF_WIDTH)


def layouts() -> None:
    """Draw the training layouts of env2 as a stack, layout 1 on top: media/env2_layouts.png."""
    spec = mujoco.MjSpec()
    cfg = make_env2_terrain_cfg()
    TerrainGenerator(cfg).compile(spec)
    terrain = spec.compile()  # Every geom is a box; its pose is in the world frame, as the terrain body sits there.
    assert (terrain.geom_type == mujoco.mjtGeom.mjGEOM_BOX).all()
    # The generator lays the rows (layouts) end to end along x, the whole grid centred on the world origin.
    start_x = -cfg.num_rows * LANE[0] / 2
    row = ((terrain.geom_pos[:, 0] - start_x) // LANE[0]).astype(int)  # Layout of each geom.
    # The lanes are drawn up to here (m): the nearest goal of the training lane whose last obstacle lies farthest.
    length = max(lane.layouts[layout][-1]["x_end"] for lane in cfg.sub_terrains.values()
                 for layout in range(ENV2_TRAIN_LAYOUTS)) + GOAL_MARGIN[0]

    stack = mujoco.MjSpec()
    for i in np.flatnonzero(row < ENV2_TRAIN_LAYOUTS):
        position, half_size = terrain.geom_pos[i].copy(), terrain.geom_size[i].copy()
        position[0] -= start_x + row[i] * LANE[0]  # Every layout starts at x = 0.
        # Cut at ``length``. Only ground boxes and walls reach that far, and their first axis is the lane's.
        low, high = position[0] - half_size[0], min(position[0] + half_size[0], length)
        if low >= length:
            continue
        position[0], half_size[0] = (low + high) / 2, (high - low) / 2
        position[2] += (ENV2_TRAIN_LAYOUTS - 1 - row[i]) * LAYER_GAP
        stack.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=half_size, pos=position,
                                 quat=terrain.geom_quat[i], rgba=terrain.geom_rgba[i])
    stack.worldbody.add_light(type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, dir=(.3, -.5, -1.), castshadow=False,
                              diffuse=(.45, .45, .45), specular=(0., 0., 0.))
    model = stack.compile()

    # The camera axes in the world frame, and the corners of the stack along them (m).
    azimuth, elevation = math.radians(VIEW["azimuth"]), math.radians(VIEW["elevation"])
    right = np.array((math.sin(azimuth), -math.cos(azimuth), 0.))
    up = np.array((-math.cos(azimuth) * math.sin(elevation), -math.sin(azimuth) * math.sin(elevation),
                   math.cos(elevation)))
    width = ENV2_LANES * LANE[1]
    top = (ENV2_TRAIN_LAYOUTS - 1) * LAYER_GAP
    corners = np.array([(x, y, z) for x in (0., length) for y in (-width / 2, width / 2) for z in (-.5, top + .5)])
    low = np.array(((corners @ right).min() - MARGIN[0], (corners @ up).min() - MARGIN[3]))
    high = np.array(((corners @ right).max() + MARGIN[2], (corners @ up).max() + MARGIN[1]))
    size = np.round((high - low) * PIXELS_PER_METRE).astype(int)  # Width, height of the picture (px).

    scale = SUPERSAMPLING
    model.vis.global_.offwidth, model.vis.global_.offheight = int(size[0] * scale), int(size[1] * scale)
    model.vis.global_.orthographic = True
    model.vis.global_.fovy = high[1] - low[1]  # Of an orthographic camera: the height of the view (m).
    model.vis.quality.offsamples = 0  # No multisampling: its edge pixels would mix with the black background.
    model.vis.headlight.ambient[:], model.vis.headlight.diffuse[:], model.vis.headlight.specular[:] = .45, .5, 0.
    model.stat.extent = 100.  # The clipping planes lie at 0.01 and 50 times this distance from the camera (m).
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    centre = (low + high) / 2
    camera.lookat[:] = centre[0] * right + centre[1] * up
    camera.distance, camera.azimuth, camera.elevation = 300., VIEW["azimuth"], VIEW["elevation"]

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=int(size[1] * scale), width=int(size[0] * scale)) as renderer:
        renderer.update_scene(data, camera=camera)
        colour = renderer.render()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=camera)
        background = renderer.render()[..., 0] < 0  # (H, W): the pixels that show no geom.
    colour[background] = 255
    picture = Image.fromarray(colour).resize(tuple(int(v) for v in size), Image.LANCZOS)

    def pixel(point):
        """Picture position (px) of the world point ``point`` (m)."""
        point = np.asarray(point)
        return (point @ right - low[0]) * PIXELS_PER_METRE, (high[1] - point @ up) * PIXELS_PER_METRE

    draw = ImageDraw.Draw(picture)
    name, note = ImageFont.load_default(64), ImageFont.load_default(40)
    ink, soft = (30, 30, 30), (95, 95, 95)
    for layout in range(ENV2_TRAIN_LAYOUTS):
        # The left corner of the layout: the far end of the lanes' start line.
        x, y = pixel((0., width / 2, (ENV2_TRAIN_LAYOUTS - 1 - layout) * LAYER_GAP))
        draw.text((x - 30, y - 24), f"Layout {layout + 1}", font=name, fill=ink, anchor="rm")
        draw.text((x - 30, y + 34), f"{ENV2_LANES} lanes", font=note, fill=soft, anchor="rm")
    # Legend along the bottom: the colour of every obstacle kind.
    x, y = 60, size[1] - 80
    for label, rgba in LEGEND:
        draw.rectangle((x, y - 22, x + 66, y + 22), fill=tuple(round(255 * c) for c in rgba[:3]))
        draw.text((x + 84, y), label, font=note, fill=ink, anchor="lm")
        x += 84 + draw.textlength(label, font=note) + 70
    MEDIA.mkdir(exist_ok=True)
    picture.save(MEDIA / "env2_layouts.png")
    print(MEDIA / "env2_layouts.png", picture.size)


def _to_test_layout(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
    """Startup event: env i runs lane i of the test layout of env2, the row after the training layouts."""
    del env_ids
    terrain = env.scene.terrain
    terrain.terrain_types[:] = torch.arange(env.num_envs, device=env.device)
    terrain.terrain_levels[:] = ENV2_TRAIN_LAYOUTS
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]


def results() -> None:
    """Film one episode of RESULT_CHECKPOINT on every lane of the test layout of env2; show RESULT_GRID of them."""
    if torch.backends.mps.is_available():
        hierarchy.DEPTH_ENCODER_DEVICE = "mps"  # As in evaluate.py: the simulation of a Mac runs on the CPU.
    cfg = make_env2_cfg(play=True)
    cfg.seed = 0  # The same spawns and goals on every run.
    cfg.scene.num_envs = ENV2_LANES
    cfg.events["test_layout"] = EventTermCfg(func=_to_test_layout, mode="startup")
    # The renderer draws shadows only around a light. The terrain's light stands over the world origin, far from the
    # test layout, so this one, which stays above the robot, replaces it: a jump is then seen by its shadow.
    cfg.scene.terrain.lights = ()
    cfg.scene.entities["robot"].lights = (LightCfg(name="above", body="base_link", mode="track", type="directional",
                                                   pos=(0., 0., 1.5), dir=(.25, .2, -1.)),)
    env = ManagerBasedRlEnv(cfg, device="cpu")
    wrapper = ParkourVecEnvWrapper(env)
    runner = Env2Runner(wrapper, asdict(env2_ppo_runner_cfg()), device="cpu")
    runner.load(RESULT_CHECKPOINT, load_cfg={"actor": True}, strict=True, map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")  # The mean action: no exploration noise.

    viewer = replace(env.cfg.viewer, width=RESULT_TILE[0], height=RESULT_TILE[1], max_extra_envs=0,
                     distance=RESULT_CAMERA["distance"])
    renderers = [OffscreenRenderer(env.sim.mj_model, replace(viewer, env_idx=i), env.scene, sim_model=env.sim.model,
                                   expanded_fields=env.sim.expanded_fields) for i in range(env.num_envs)]
    for renderer in renderers:
        renderer.initialize()
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = (RESULT_CAMERA[key] for key in
                                                         ("distance", "azimuth", "elevation"))
    robot = env.scene["robot"]
    origins = env.scene.env_origins.cpu().numpy()  # (N, 3): a point on the centre line of every lane, on its ground.
    font = ImageFont.load_default(24)
    fps = 1 / (RESULT_FRAME_STEPS * env.step_dt)
    goal = env.command_manager.get_term("goal")
    episodes = record_episodes(env, 1)  # Filled as the first episode of every lane ends.
    running = [True] * env.num_envs  # The first episode of the lane has not ended.
    frames = [0] * env.num_envs  # Frames filmed of every lane.
    control_steps = 0
    step = env.step

    with tempfile.TemporaryDirectory() as folder:
        lanes = [imageio.get_writer(Path(folder) / f"{i}.mp4", fps=fps, quality=8, macro_block_size=1)
                 for i in range(env.num_envs)]

        def step_and_film(action):
            """One control step of the env, then a frame of every lane whose first episode still runs."""
            nonlocal control_steps
            result = step(action)
            control_steps += 1
            for i in (result[2] | result[3]).nonzero().flatten().tolist():
                running[i] = False  # The env has been reset inside the step: its robot is back at the spawn.
            if control_steps % RESULT_FRAME_STEPS == 0:
                kind, passed = goal.obstacle_outcomes()
                root_x = robot.data.root_link_pos_w[:, 0].cpu().numpy()
                for i in range(env.num_envs):
                    if not running[i]:
                        continue
                    camera.lookat[:] = root_x[i] + RESULT_LOOK_AHEAD, origins[i, 1], origins[i, 2]
                    renderers[i].update(env.sim.data, camera=camera)
                    tile = Image.fromarray(renderers[i].render())
                    draw = ImageDraw.Draw(tile)
                    outline = dict(font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
                    draw.text((12, 10), f"test lane {i + 1}", **outline)
                    draw.text((RESULT_TILE[0] - 12, 10), f"obstacles {int(passed[i].sum())}/{int((kind[i] >= 0).sum())}"
                              f"   {float(env.episode_length_buf[i]) * env.step_dt:4.1f} s", anchor="ra", **outline)
                    lanes[i].append_data(np.asarray(tile))
                    frames[i] += 1
            return result

        env.step = step_and_film  # The wrapper runs SUB_STEPS control steps per high-level step through this.
        observation = wrapper.get_observations()
        with torch.inference_mode():
            while any(running):
                observation, _, dones, _ = wrapper.step(policy(observation))
                policy.reset(dones)  # Clear the policy's memory of the envs that start a new episode.
        for lane in lanes:
            lane.close()

        episode = {e["column"]: e for e in episodes}  # Lane -> its episode.
        reached = [lane for lane in sorted(episode) if "goal_reached" in episode[lane]["outcome"]]
        seconds = {lane: frames[lane] / fps for lane in episode}  # Length of every lane's episode.
        for lane in sorted(episode):
            print(f"test lane {lane + 1:2d}: passed {episode[lane]['passed']} of {len(episode[lane]['kinds'])} "
                  f"({', '.join(episode[lane]['kinds'])}), {episode[lane]['outcome']}, {seconds[lane]:.1f} s")
        print(f"goal reached on {len(reached)} of {len(episode)} test lanes")

        # The lanes of the clip: those that reached the goal come first, and among them the two outer lanes last,
        # where the camera sees past the edge of the terrain; otherwise in lane order.
        shown = sorted(sorted(episode, key=lambda lane: (lane not in reached, lane in (0, ENV2_LANES - 1), lane))
                       [:RESULT_GRID[0] * RESULT_GRID[1]])
        print("lanes in the clip:", ", ".join(str(lane + 1) for lane in shown))
        clips = [iter(imageio.get_reader(Path(folder) / f"{lane}.mp4")) for lane in shown]
        tiles = [None] * len(shown)  # The current frame of every lane; after its episode, the last one with its result.
        video = open_video("env2_test_results", fps)
        for _ in range(round((max(seconds[lane] for lane in shown) + RESULT_HOLD_S) * fps)):
            for index, lane in enumerate(shown):
                if clips[index] is None:
                    continue
                frame = next(clips[index], None)
                if frame is not None:
                    tiles[index] = frame
                    continue
                clips[index] = None
                success = lane in reached
                text = (f"goal reached in {seconds[lane]:.1f} s" if success else
                        f"{episode[lane]['outcome'].replace('_', ' ')} after {seconds[lane]:.1f} s")
                tile = Image.fromarray(tiles[index])
                draw = ImageDraw.Draw(tile)
                box = draw.textbbox((RESULT_TILE[0] / 2, RESULT_TILE[1] - 44), text, font=font, anchor="mm")
                draw.rounded_rectangle((box[0] - 16, box[1] - 10, box[2] + 16, box[3] + 10), radius=10,
                                       fill=(34, 139, 69) if success else (190, 50, 45))
                draw.text((RESULT_TILE[0] / 2, RESULT_TILE[1] - 44), text, font=font, anchor="mm")
                tiles[index] = np.asarray(tile)
            rows = [np.concatenate(tiles[row * RESULT_GRID[1]:(row + 1) * RESULT_GRID[1]], axis=1)
                    for row in range(RESULT_GRID[0])]
            video.append_data(np.concatenate(rows))
        video.close()
    write_gif("env2_test_results", fps / 2, RESULT_GIF_WIDTH)
    for renderer in renderers:
        renderer.close()
    wrapper.close()


if __name__ == "__main__":
    {"skills": skills, "stages": stages, "layouts": layouts, "results": results}[sys.argv[1]]()
