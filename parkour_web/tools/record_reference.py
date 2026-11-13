"""Record reference episodes of the Python hierarchy (mjlab + mujoco_warp) for the JavaScript port to be checked against.

    cd open-beni/python
    PYTHONPATH=. python ../parkour_web/tools/record_reference.py [checkpoint] [env ids, comma separated] [max high-level steps]

Runs the play form of Beni-Parkour-Env2 (80 envs, one per lane) with the deterministic high-level policy, as
parkour_rl.evaluate does, and records the first episode of the chosen envs. Per episode it writes
``tools/ref/rollouts/env<id>.json`` (lane, spawn, goal, outcome, and the layout of the binary file) and
``env<id>.bin`` (float32, little endian). The arrays, with T control steps, 5 T physics steps and H high-level steps:

    every physics step     ctrl (8) the motor torques applied in the step, qpos (15) and qvel (14) after it, lag (4)
                           the command delay of the four delayed motor groups in physics steps, wheel_force (2, 3)
                           and body_force (6, 3) the contact sensors after it
    every control step     crawl_obs and jump_obs (40) the low-level observation as each frozen policy gets it,
                           crawl_action and jump_action (7) their outputs, action (7) the one applied, state (see
                           STATE_KEYS) the jump command, the latch and the jump measurement after the env step,
                           done (1)
    every high-level step  hl_obs (24), features (440), hl_action (3), depth (60, 80)
    once                   qpos0 (15), qvel0 (14): the state the episode starts from

The depth encoder runs on MPS on a Mac, as in parkour_rl.evaluate; its features differ from the CPU's by about 1e-4.
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402

from parkour_rl import ENV2_TASK_ID, hierarchy  # noqa: E402
from parkour_rl.depth import DEPTH_SENSOR  # noqa: E402
from parkour_rl.hierarchy import ParkourVecEnvWrapper  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tools/ref/rollouts"
CHECKPOINT = "logs/rsl_rl/beni_parkour_env2/2026-11-10_23-34-36/model_1000.pt"
# Env i of the play form sits on lane column i // 5 and row i % 5. Six lanes of the training rows and two of the
# test row.
ENV_IDS = (0, 6, 12, 18, 21, 27, 4, 39)
# Entries of the per-control-step ``state`` array, read after the env step.
STATE_KEYS = ("ready_steps", "yaw_ready_steps", "cool_down", "jump_flag", "jump_count", "preparing", "tracking",
              "gate", "lifted", "land_s", "age", "heading_active", "command_hold", "failed", "ready_hold",
              "pending", "trigger_high", "in_jump", "jump_ready", "air", "contact_left", "contact_right", "body",
              "load", "height", "tilt", "error", "margin", "twist_vx", "twist_vy", "twist_wz", "height_command",
              "goal_distance", "position_x", "lateral_offset", "heading")


class Recorder:
    """Per-env lists of arrays, filled by the hooks while the env is being recorded."""

    def __init__(self, env_ids: tuple[int, ...]) -> None:
        self.ids = list(env_ids)
        self.active = set(env_ids)
        self.data = {i: {} for i in env_ids}

    def add(self, name: str, values: torch.Tensor) -> None:
        """Append row ``i`` of ``values`` (N, ...) to the list ``name`` of every env still being recorded."""
        values = values.detach().to("cpu", torch.float32)
        for i in self.active:
            self.data[i].setdefault(name, []).append(values[i].numpy().copy())


class PolicyProxy:
    """A frozen low-level policy whose outputs are recorded."""

    def __init__(self, policy, recorder: Recorder, name: str) -> None:
        self._policy, self._recorder, self._name = policy, recorder, name

    def __call__(self, observation):
        action = self._policy(observation)
        self._recorder.add(f"{self._name}_obs", observation["actor"])
        self._recorder.add(f"{self._name}_action", action)
        return action

    def reset(self, *args, **kwargs):
        return self._policy.reset(*args, **kwargs)


def install_hooks(env: ManagerBasedRlEnv, wrapper: ParkourVecEnvWrapper, recorder: Recorder) -> None:
    robot = env.scene["robot"]
    jump = env.command_manager.get_term("jump")
    goal = env.command_manager.get_term("goal")
    hl_state = env.command_manager.get_term("hl_state")
    twist = env.command_manager.get_term("twist")
    delayed = [a for a in robot.actuators if getattr(a, "_delay_buffer", None) is not None]
    assert len(delayed) == 4, [type(a).__name__ for a in robot.actuators]

    scene_update = env.scene.update

    def recording_update(dt):  # After every physics step.
        scene_update(dt)
        recorder.add("ctrl", env.sim.data.ctrl)
        recorder.add("qpos", env.sim.data.qpos)
        recorder.add("qvel", env.sim.data.qvel)
        recorder.add("lag", torch.stack([a._delay_buffer.current_lags for a in delayed], dim=1))
        recorder.add("wheel_force", env.scene["wheel_ground_contact"].data.force)
        recorder.add("body_force", env.scene["body_ground_contact"].data.force)

    env.scene.update = recording_update

    wrapper._crawl_policy = PolicyProxy(wrapper._crawl_policy, recorder, "crawl")
    wrapper._jump_policy = PolicyProxy(wrapper._jump_policy, recorder, "jump")

    step = env.step

    def recording_step(action):
        recorder.add("action", action)
        result = step(action)
        s, m = jump.state, jump.sample
        values = dict(
            ready_steps=jump.ready_steps, yaw_ready_steps=jump.yaw_ready_steps, cool_down=jump.jump_cool_down,
            jump_flag=jump.jump_flag, jump_count=jump.jump_count, preparing=jump.preparing, tracking=s["tracking"],
            gate=s["gate"], lifted=s["lifted"], land_s=s["land_s"], age=s["age"], heading_active=jump.heading_active,
            command_hold=jump.command_hold, failed=s["failed"], ready_hold=s["ready_hold"],
            pending=hl_state.pending, trigger_high=hl_state.trigger_high, in_jump=hl_state.in_jump,
            jump_ready=m["jump_ready"], air=m["air"], contact_left=m["contacts"][:, 0],
            contact_right=m["contacts"][:, 1], body=m["body"], load=m["load"], height=m["height"], tilt=m["tilt"],
            error=m["error"], margin=m["margin"], twist_vx=twist.vel_command_b[:, 0],
            twist_vy=twist.vel_command_b[:, 1], twist_wz=twist.vel_command_b[:, 2],
            height_command=twist.base_pose_command[:, 2], goal_distance=goal.distance(),
            position_x=goal.position(), lateral_offset=robot.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1],
            heading=robot.data.heading_w)
        recorder.add("state", torch.stack([values[key].float() for key in STATE_KEYS], dim=1))
        recorder.add("done", (result[2] | result[3]).float().unsqueeze(1))
        return result

    env.step = recording_step


def save(env_id: int, arrays: dict, info: dict) -> None:
    """Write the arrays of one episode as a float32 file with a JSON index."""
    layout, offset, chunks = {}, 0, []
    for name, rows in arrays.items():
        block = np.stack(rows).astype("<f4")
        layout[name] = dict(offset=offset, shape=list(block.shape))
        offset += block.size
        chunks.append(block.ravel())
    OUT.mkdir(parents=True, exist_ok=True)
    np.concatenate(chunks).tofile(OUT / f"env{env_id}.bin")
    (OUT / f"env{env_id}.json").write_text(json.dumps(dict(info, arrays=layout, state_keys=STATE_KEYS), indent=1))


def main() -> None:
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else CHECKPOINT
    env_ids = tuple(int(i) for i in sys.argv[2].split(",")) if len(sys.argv) > 2 else ENV_IDS
    max_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1400
    torch.manual_seed(0)
    if torch.backends.mps.is_available():
        hierarchy.DEPTH_ENCODER_DEVICE = "mps"
    env = ManagerBasedRlEnv(load_env_cfg(ENV2_TASK_ID, play=True), device="cpu")
    wrapper = ParkourVecEnvWrapper(env)
    runner = (load_runner_cls(ENV2_TASK_ID) or MjlabOnPolicyRunner)(wrapper, asdict(load_rl_cfg(ENV2_TASK_ID)),
                                                                    device="cpu")
    runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")

    recorder = Recorder(env_ids)
    goal, terrain, manager = env.command_manager.get_term("goal"), env.scene.terrain, env.termination_manager
    info = {i: dict(env_id=i, checkpoint=checkpoint, row=int(terrain.terrain_levels[i]),
                    column=int(terrain.terrain_types[i]), origin=env.scene.env_origins[i].tolist(),
                    spawn_x=float(goal.spawn_x[i]), goal_x=float(goal.goal_x[i]),
                    time_budget=float(goal.time_budget[i])) for i in env_ids}
    for i in env_ids:
        recorder.data[i]["qpos0"] = [env.sim.data.qpos[i].float().numpy().copy()]
        recorder.data[i]["qvel0"] = [env.sim.data.qvel[i].float().numpy().copy()]
    install_hooks(env, wrapper, recorder)

    compute = manager.compute

    def recording_compute():  # The termination terms that end an episode, read before the reset clears them.
        finished = compute()
        for i in finished.nonzero().flatten().tolist():
            if i in recorder.active:
                info[i]["outcome"] = "+".join(n for n in manager.active_terms if manager.get_term(n)[i])
        return finished

    manager.compute = recording_compute

    observation = wrapper.get_observations()
    with torch.inference_mode():
        for step in range(max_steps):
            recorder.add("hl_obs", observation["hl_actor"])
            recorder.add("features", observation["depth"])
            recorder.add("depth", env.scene[DEPTH_SENSOR].data.depth[..., 0])
            action = policy(observation)
            recorder.add("hl_action", action)
            observation, _, dones, _ = wrapper.step(action)
            policy.reset(dones)
            for i in dones.nonzero().flatten().tolist():
                if i in recorder.active:
                    recorder.active.discard(i)
                    info[i]["high_level_steps"] = step + 1
                    save(i, recorder.data.pop(i), info[i])
                    print(f"env {i} (row {info[i]['row']}, column {info[i]['column']}): {info[i].get('outcome')} "
                          f"after {step + 1} high-level steps", flush=True)
            if not recorder.active:
                break
            if step % 100 == 0:
                print(f"high-level step {step}: {len(recorder.active)} envs still recording", flush=True)
    for i in list(recorder.active):  # Not finished within max_steps: saved as far as it got.
        info[i]["high_level_steps"] = max_steps
        info[i]["outcome"] = "unfinished"
        save(i, recorder.data.pop(i), info[i])
        print(f"env {i}: unfinished after {max_steps} high-level steps")
    wrapper.close()


if __name__ == "__main__":
    main()
