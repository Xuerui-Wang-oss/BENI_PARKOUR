"""Run the LLM high-level policy on parkour lanes, log every call, and print the score.

    python -m parkour_llm.run env1 --lanes pit:2 --view depth              # one robot, one lane
    python -m parkour_llm.run env1 --lanes all --view rgb --max-calls 2500
    python -m parkour_llm.run env2 --lanes test --view depth --episodes 1
    python -m parkour_llm.run env1 --lanes all --backend oracle            # no API: the scripted test driver

``--lanes`` takes ``all``, ``test`` (the held-out layout of env2), an env1 kind (``pit``), or a list such as
``pit:2,hollow:4`` or ``3:4`` (lane column:row). ``--view`` is what the LLM sees: ``depth``, ``rgb`` or ``rgbd``.
Each robot runs ``--episodes`` episodes on its lane and then parks. The simulation is paused while the LLM answers.

Every call is paid for. ``--max-calls`` (default 400) stops a run before it has made more calls than that; raise it
on purpose. The run directory gets ``decisions.jsonl`` (every prompt and reply), the frames, ``episodes.json`` and
``summary.json``, and with ``--video`` a ``video.mp4`` of the first lane or with ``--videos`` one clip per lane under
``videos/``. The Claude backend reads ``ANTHROPIC_API_KEY`` from the environment.

``--replay DIR`` runs an earlier run again from its logged decisions, without calling a model, to add videos to it:

    python -m parkour_llm.run env1 --replay runs/sonnet_env1_all25 --videos
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv

from parkour_rl import hierarchy
from parkour_rl.env2 import ENV2_TRAIN_LAYOUTS
from parkour_rl.evaluate import print_report, record_episodes
from parkour_rl.hierarchy import SUB_STEPS, ParkourVecEnvWrapper

from .backends import (DEFAULT_MODEL, JEV_MODELS, JEV_OPTIONS, MODEL_LABELS, OracleBackend, ReplayBackend, cost_usd,
                       has_credentials, load_dotenv, make_backend)
from .policy import CallBudgetExhausted, LlmPolicy
from .prompt import DEFAULT_PROMPT, PROMPTS, VIEWS
from . import starts as fixed_starts
from .scene import PPO_SIZE, cell_name, make_llm_env_cfg, parse_cells
from .video import RunVideo
from .views import TABLE_STYLES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("task", choices=("env1", "env2"))
    parser.add_argument("--lanes", default="all", help="all, test, an env1 kind, or column:row,... (default all)")
    parser.add_argument("--view", choices=VIEWS, default=None,
                        help="what the model sees (default depth; table, the text alone, for Jev)")
    parser.add_argument("--backend", choices=("claude", "oracle"), default="claude")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"model id: {', '.join(MODEL_LABELS)} (default {DEFAULT_MODEL}). Jev takes text only, so "
                             "it always gets the table view")
    parser.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"), default="low",
                        help="how much the model thinks per call (default low)")
    parser.add_argument("--table", choices=TABLE_STYLES, default="camera",
                        help="how the depth table is written: heights from the camera (default) or above the floor "
                             "with the step between rows; the same frame either way")
    parser.add_argument("--prompt", choices=PROMPTS, default=None,
                        help=f"version of the instructions (default {DEFAULT_PROMPT}; v1 for Jev, whose questions were "
                             "tuned with it). v1 is the text of the runs in RESULTS.md up to 2026-11-22")
    parser.add_argument("--jev", default="", metavar="OPTIONS",
                        help=f"comma-separated variations of a Jev call: {', '.join(JEV_OPTIONS)}")
    parser.add_argument("--fast", action="store_true",
                        help="fast mode: quicker answers at twice the price, on the models that have it (Opus 5.5)")
    parser.add_argument("--period", type=int, default=5,
                        help="high-level steps of 0.05 s per LLM call (default 5: one call per 0.25 s)")
    parser.add_argument("--episodes", type=int, default=1, help="episodes per lane (default 1)")
    parser.add_argument("--max-calls", type=int, default=None,
                        help="stop before more model calls than this (default 400; 20000 for Jev, whose calls cost "
                             "about a hundredth of a cent)")
    parser.add_argument("--workers", type=int, default=16,
                        help="most API requests in flight at once (default 16); lower it if you hit rate limits")
    parser.add_argument("--camera", default=f"{PPO_SIZE[0]}x{PPO_SIZE[1]}",
                        help="LLM camera frame as ROWSxCOLUMNS (default 60x80, what the PPO policy gets)")
    parser.add_argument("--out", type=Path, default=None, help="run directory (default runs/<task>_<view>_<time>)")
    parser.add_argument("--no-frames", action="store_true", help="do not save the frames sent to the LLM")
    parser.add_argument("--video", action="store_true",
                        help="record video.mp4 of the first lane: the robot from outside, beside what the LLM was "
                             "shown and what it answered")
    parser.add_argument("--videos", action="store_true", help="record such a video of every lane, under videos/")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed of the start and goal positions (default: the env's own, 42). The courses stay "
                             "the same")
    parser.add_argument("--starts", type=Path, default=None, metavar="CSV",
                        help="start and goal of every episode from this episode file of the trained policy's "
                             "evaluation (lane, episode, spawn_x_m, spawn_y_m, goal_x_m) instead of a random draw")
    parser.add_argument("--replay", type=Path, default=None, metavar="DIR",
                        help="repeat the run logged in DIR from its decisions, calling no model: same lanes, view "
                             "and period. Writes to DIR/replay unless --out is given")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    if args.replay is not None:
        logged = json.loads((args.replay / "summary.json").read_text())
        if logged["task"] != args.task:
            sys.exit(f"{args.replay} is a run of {logged['task']}, not {args.task}")
        args.backend, args.lanes, args.view, args.table = "replay", logged["lanes"], logged["view"], logged["table"]
        args.prompt = logged.get("prompt", "v1")  # Runs from before the versions were all v1.
        args.period, args.camera = logged["period"], "x".join(str(n) for n in logged["camera"])
        args.episodes = logged["episodes_wanted"] // len(parse_cells(args.task, args.lanes))
        args.seed = logged.get("seed")
        if (args.replay / "starts.csv").exists():
            args.starts = args.replay / "starts.csv"
        args.out = args.out or args.replay / "replay"
    jev = args.backend == "claude" and args.model in JEV_MODELS
    if args.view is None:
        args.view = "table" if jev else "depth"
    options = tuple(option for option in args.jev.split(",") if option)
    if set(options) - set(JEV_OPTIONS):
        sys.exit(f"--jev takes {', '.join(JEV_OPTIONS)}; got {args.jev}")
    if jev and args.view in ("rgb", "rgbd"):
        sys.exit("Jev runs with --view table (text only) or --view depth (the table and the depth picture).")
    if args.prompt is None:
        args.prompt = "v1" if jev else DEFAULT_PROMPT
    if args.max_calls is None:
        args.max_calls = 20000 if jev else 400
    if args.backend == "claude" and not has_credentials(args.model):
        sys.exit(f"No API key found for {args.model}. Put {'JEV_KEY' if jev else 'ANTHROPIC_API_KEY'}=... in "
                 "BENI_PARKOUR/.env (it is ignored by git), then run again. --backend oracle needs no key.")
    cells = parse_cells(args.task, args.lanes)
    labels = [cell_name(args.task, cell) for cell in cells]
    size = tuple(int(n) for n in args.camera.lower().split("x"))
    out = args.out or Path("runs") / f"{args.task}_{args.view}_{args.backend}_{time.strftime('%m%d-%H%M%S')}"

    if torch.backends.mps.is_available():
        # The wrapper still encodes the PPO depth image on every step; nothing reads the result here.
        hierarchy.DEPTH_ENCODER_DEVICE = "mps"
    cfg = make_llm_env_cfg(args.task, cells, size)
    if args.seed is not None:
        cfg.seed = args.seed
    starts = None
    if args.starts is not None:
        starts = fixed_starts.Starts(args.starts)
        if args.episodes > starts.episodes:
            sys.exit(f"{args.starts} has {starts.episodes} episodes per lane; --episodes {args.episodes} asks more")
        fixed_starts.install(cfg, starts)
    env = ManagerBasedRlEnv(cfg, device="cpu")
    if starts is not None:
        fixed_starts.attach(env, starts)
    wrapper = ParkourVecEnvWrapper(env)
    backend = (OracleBackend(env, args.period) if args.backend == "oracle"
               else ReplayBackend(args.replay) if args.backend == "replay"
               else make_backend(args.model, args.effort, args.fast, args.period,
                                 workers=max(1, min(len(cells), args.workers)), options=options, table=args.table))
    policy = LlmPolicy(wrapper, backend, args.view, period=args.period, labels=labels, log_dir=out,
                       save_frames=not args.no_frames, max_calls=args.max_calls if args.backend == "claude" else None,
                       table_style=args.table, prompt=args.prompt)

    def lane_video(env_id: int, episode: int) -> RunVideo:
        """The recorder of one lane's episode: one file per lane, or per episode when a lane runs several."""
        name, title = labels[env_id].replace(" ", "_"), labels[env_id]
        if args.episodes > 1:
            name, title = f"{name}_ep{episode + 1}", f"{title} · attempt {episode + 1}"
        return RunVideo(env, out / "videos" / f"{name}.mp4", env_id, title=f"{title} · {backend.model} · {args.view}")

    videos = {}  # env id: its recorder.
    if args.videos:
        (out / "videos").mkdir(parents=True, exist_ok=True)
        for env_id in range(len(labels)):
            videos[env_id] = lane_video(env_id, 0)
    elif args.video:
        videos[0] = RunVideo(env, out / "video.mp4", title=f"{labels[0]} · {backend.model} · {args.view}")
    episodes = record_episodes(env, args.episodes)
    finished = [0] * env.num_envs
    began = [0] * env.num_envs  # The step on which each env's running episode began.
    wanted = args.episodes * env.num_envs
    max_steps = args.episodes * (env.max_episode_length // SUB_STEPS + 1)
    if starts is not None:
        wrapper.get_observations()
        for label, (x, y, goal_x, budget) in zip(labels, fixed_starts.placed(env)):
            print(f"  start {label}: x {x:+.4f} y {y:+.4f} goal {goal_x:.4f} budget {budget:.3f}")
    print(f"{args.task}: {len(cells)} lanes x {args.episodes} episodes, view {args.view}, instructions {args.prompt}, "
          "backend "
          f"{backend.model if args.backend != 'claude' or jev else f'{args.model} (effort {args.effort})'}"
          f"{' (replayed from ' + str(args.replay) + ')' if args.replay else ''}, one call per "
          f"{args.period * 0.05:.2f} s. Logging to {out}", flush=True)

    start, stopped = time.perf_counter(), None
    try:
        with torch.inference_mode():
            for step in range(max_steps):
                action = policy.act()
                _, _, dones, _ = wrapper.step(action)
                for env_id, video in videos.items():
                    # Not on the step that ended the episode: the env has already put the robot back at the start.
                    if finished[env_id] < args.episodes and not dones[env_id]:
                        video.frame(policy.last[env_id], (step + 1 - began[env_id]) * SUB_STEPS * env.step_dt)
                policy.episode_ended(dones)
                for env_id in dones.nonzero().flatten().tolist():
                    finished[env_id] += 1
                    began[env_id] = step + 1
                    if args.videos and finished[env_id] < args.episodes:
                        videos[env_id].close()
                        videos[env_id] = lane_video(env_id, finished[env_id])
                    if finished[env_id] <= args.episodes:
                        print(f"  step {step:4d}  {labels[env_id]}: episode ended ({len(episodes)}/{wanted})",
                              flush=True)
                    policy.active[env_id] = finished[env_id] < args.episodes
                if step % (20 * args.period) == 0:
                    print(f"  step {step:4d}  {backend.calls} calls, {len(episodes)}/{wanted} episodes, "
                          f"{time.perf_counter() - start:.0f} s", flush=True)
                if len(episodes) >= wanted:
                    break
    except CallBudgetExhausted as error:
        stopped = f"call budget exhausted ({error}); raise --max-calls to go on"
    except KeyboardInterrupt:
        stopped = "interrupted"
    finally:
        policy.close()
        for video in videos.values():
            video.close()

    label_of = dict(zip(cells, labels))
    for episode in episodes:
        episode["lane"] = label_of[(episode["column"], episode["row"])]
    print()
    for episode in episodes:
        print(f"{episode['lane']:14s} {episode['outcome']:14s} passed {episode['passed']}/{len(episode['kinds'])}"
              f"  {' '.join(episode['kinds'])}")
    print()
    generator = env.scene.terrain.cfg.terrain_generator
    print_report(episodes, generator.num_rows, ENV2_TRAIN_LAYOUTS if args.task == "env2" else None)

    cost = cost_usd(getattr(backend, "model", ""), backend.usage, getattr(backend, "fast", False))
    summary = {
        "task": args.task, "lanes": args.lanes, "view": args.view, "backend": args.backend, "table": args.table,
        "prompt": args.prompt, "starts": None if starts is None else "starts.csv",
        "seed": env.cfg.seed,
        "jev_options": list(options),
        "model": backend.model, "effort": args.effort if args.backend == "claude" else None, "period": args.period,
        "camera": list(size), "episodes_wanted": wanted, "episodes_finished": len(episodes),
        "goal_reached": sum("goal_reached" in e["outcome"] for e in episodes),
        "outcomes": dict(collections.Counter(e["outcome"] for e in episodes)),
        "calls": backend.calls, "unusable_answers": policy.errors, "usage": backend.usage,
        "estimated_cost_usd": None if cost is None else round(cost, 4),
        "wall_time_s": round(time.perf_counter() - start, 1), "stopped": stopped,
    }
    if starts is not None and starts.path.resolve() != (out / "starts.csv").resolve():
        (out / "starts.csv").write_text(starts.path.read_text())
    (out / "episodes.json").write_text(json.dumps(episodes, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\ncalls {backend.calls}, unusable answers {policy.errors}, tokens {backend.usage}")
    if cost is not None and backend.calls:
        print(f"estimated cost ${cost:.2f} (${cost / backend.calls:.4f} per call)")
    if stopped:
        print(f"STOPPED EARLY: {stopped}")
    print(f"log: {out}")
    wrapper.close()


if __name__ == "__main__":
    main()
