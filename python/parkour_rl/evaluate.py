# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""Measure a trained high-level parkour policy on every lane, with deterministic actions and no viewer.

    python -m parkour_rl.evaluate env2 logs/rsl_rl/beni_parkour_env2/<run>/model_<iteration>.pt
    python -m parkour_rl.evaluate env1 logs/rsl_rl/beni_parkour_env1/<run>/model_<iteration>.pt 3

The first argument, ``env1`` or ``env2``, is optional as in ``parkour_rl.train``; the last, also optional, is the
number of episodes per lane (default 2). The script builds the play form of the task, one robot per lane, and lets
the actor of the checkpoint drive each robot through that many episodes on its lane; every episode starts from a new
random spawn with a new goal. An env1 checkpoint also runs on env2 (the actor's inputs are the same), which gives the
score of env2 before any training on it.

It prints, per lane row and in total, how many episodes reached the goal, how many passed every obstacle of the lane,
and how many of the obstacles that the robots got to they passed; then the same per obstacle kind and the ways the
episodes ended. For env2 the training layouts (rows 0-3) and the test layout (row 4) are also totalled apart: the
policy generalises if the two totals are close. A robot got to an obstacle if it passed the one before it; an
episode that ends between two obstacles counts against the second.

On a Mac the simulation runs on the CPU and the depth encoder on MPS (``hierarchy.DEPTH_ENCODER_DEVICE``), which
makes a step about three times as fast; 2 episodes on the 80 lanes of env2 then take about a quarter of an hour.
"""

import collections
import sys
from dataclasses import asdict

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from . import ENV2_TASK_ID, hierarchy, task_from_argv  # Importing the package registers the tasks.
from .env2 import ENV2_TRAIN_LAYOUTS
from .hierarchy import SUB_STEPS, ParkourVecEnvWrapper


def record_episodes(env: ManagerBasedRlEnv, per_lane: int) -> list[dict]:
    """Record the first ``per_lane`` episodes of every env as they end; returns the list that is filled in.

    One dict per episode: ``row`` and ``column`` of its lane, ``outcome`` (the termination terms that ended it, joined
    by "+"), ``kinds`` (kind of every obstacle of the lane, in lane order) and ``passed`` (how many of them the robot
    root passed). The termination manager is wrapped, as in check.py, because the robot is still where the episode
    ended only until the reset that follows on the same control step.
    """
    manager, terrain = env.termination_manager, env.scene.terrain
    goal = env.command_manager.get_term("goal")
    compute = manager.compute
    episodes: list[dict] = []
    counts = [0] * env.num_envs

    def recording_compute() -> torch.Tensor:
        finished = compute()
        if finished.any():
            kind, passed = goal.obstacle_outcomes()
            for env_id in finished.nonzero().flatten().tolist():
                if counts[env_id] < per_lane:
                    counts[env_id] += 1
                    episodes.append(dict(
                        row=int(terrain.terrain_levels[env_id]), column=int(terrain.terrain_types[env_id]),
                        outcome="+".join(name for name in manager.active_terms if manager.get_term(name)[env_id]),
                        kinds=[goal.kinds[int(k)] for k in kind[env_id].tolist() if k >= 0],
                        passed=int(passed[env_id].sum())))
        return finished

    manager.compute = recording_compute
    return episodes


def summary(episodes: list[dict]) -> str:
    """Goal, all-passed and per-obstacle counts of ``episodes`` as one line of text."""
    goal = sum("goal_reached" in e["outcome"] for e in episodes)
    complete = sum(bool(e["kinds"]) and e["passed"] == len(e["kinds"]) for e in episodes)  # Not on a flat lane.
    passed = sum(e["passed"] for e in episodes)
    reached = sum(min(e["passed"] + 1, len(e["kinds"])) for e in episodes)
    count = len(episodes)
    return (f"{count} episodes, goal {goal} ({goal / max(count, 1):.0%}), all obstacles passed {complete} "
            f"({complete / max(count, 1):.0%}), obstacles passed {passed}/{reached} ({passed / max(reached, 1):.0%})")


def print_report(episodes: list[dict], num_rows: int, split: int | None) -> None:
    """Print the counts of the module docstring. ``split``: first row of the test layouts, None without any."""
    for row in range(num_rows):
        print(f"row {row}: {summary([e for e in episodes if e['row'] == row])}")
    if split is not None:
        print(f"training layouts (rows 0-{split - 1}): {summary([e for e in episodes if e['row'] < split])}")
        print(f"test layout (row {split}{'' if split == num_rows - 1 else f'-{num_rows - 1}'}):"
              f"        {summary([e for e in episodes if e['row'] >= split])}")
    print(f"all lanes: {summary(episodes)}")
    by_kind = collections.defaultdict(lambda: [0, 0])  # Kind -> obstacles passed, obstacles the robots got to.
    for episode in episodes:
        for index, kind in enumerate(episode["kinds"][:episode["passed"] + 1]):
            by_kind[kind][0] += index < episode["passed"]
            by_kind[kind][1] += 1
    print("passed / got to, per kind:", ", ".join(f"{kind} {a}/{b}" for kind, (a, b) in sorted(by_kind.items())))
    print("episodes ended with:", dict(collections.Counter(e["outcome"] for e in episodes).most_common()))


def main() -> None:
    """Build the play env, run the checkpoint's actor on it and print the report."""
    task_id = task_from_argv()
    checkpoint = sys.argv[1]
    per_lane = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and torch.backends.mps.is_available():
        # On a Mac the depth encoder takes three quarters of a CPU step; the simulation itself cannot run on MPS.
        hierarchy.DEPTH_ENCODER_DEVICE = "mps"
    env = ManagerBasedRlEnv(load_env_cfg(task_id, play=True), device=device)
    wrapper = ParkourVecEnvWrapper(env)
    runner = (load_runner_cls(task_id) or MjlabOnPolicyRunner)(wrapper, asdict(load_rl_cfg(task_id)), device=device)
    runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)  # The mean action: no exploration noise.

    episodes = record_episodes(env, per_lane)
    wanted = per_lane * env.num_envs
    # No episode outlasts max_episode_length control steps, so this many high-level steps end every episode wanted.
    max_steps = per_lane * (env.max_episode_length // SUB_STEPS + 1)
    print(f"{task_id}, {checkpoint}: {per_lane} episodes on each of {env.num_envs} lanes, deterministic actions",
          flush=True)
    observation = wrapper.get_observations()
    with torch.inference_mode():
        for step in range(max_steps):
            observation, _, dones, _ = wrapper.step(policy(observation))
            policy.reset(dones)  # Clear the policy's memory of the envs that start a new episode, as in training.
            if step % 200 == 0:
                print(f"high-level step {step}: {len(episodes)} of {wanted} episodes", flush=True)
            if len(episodes) == wanted:
                break
    print_report(episodes, env.scene.terrain.cfg.terrain_generator.num_rows,
                 ENV2_TRAIN_LAYOUTS if task_id == ENV2_TASK_ID else None)
    wrapper.close()


if __name__ == "__main__":
    main()
