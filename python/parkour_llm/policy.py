"""The LLM in the seat of the high-level policy.

``LlmPolicy.act()`` returns the high-level action (N, 3) that ``ParkourVecEnvWrapper.step`` takes, as the PPO actor
does. Every ``period`` high-level steps it asks the backend for a plan per robot and holds it in between: speed and
body height stay as planned, and the jump trigger is raised on the one step the plan names.
"""

import json
from collections import deque
from pathlib import Path

import torch

from beni_rl_crawl.recovery import H_LOW, H_REF
from parkour_rl.check import to_action
from parkour_rl.hierarchy import VX_RANGE, ParkourVecEnvWrapper

from .backends import Decision
from .observe import build_request
from .prompt import DEFAULT_PROMPT, system_prompt
from .scene import LLM_SENSOR


class CallBudgetExhausted(RuntimeError):
    """The next decision tick would take the run past its call budget."""


class LlmPolicy:
    """Plans for every active robot, refreshed every ``period`` steps.

    ``view`` selects what the backend is shown: "depth" (depth picture and depth table), "rgb" (colour picture) or
    "rgbd" (both). ``history`` is how many of its own past decisions a robot's request repeats. ``labels`` names
    each env's lane in the log. ``log_dir`` receives ``decisions.jsonl``, one line per call, and the frames as PNG
    files under ``frames/``. ``max_calls`` stops the run before the backend has answered more calls than that.
    """

    def __init__(self, wrapper: ParkourVecEnvWrapper, backend, view: str, period: int = 5, history: int = 3,
                 labels: list[str] | None = None, log_dir: Path | None = None, save_frames: bool = True,
                 max_calls: int | None = None, table_style: str = "camera", prompt: str = DEFAULT_PROMPT) -> None:
        self._wrapper, self._env, self._backend = wrapper, wrapper.unwrapped, backend
        self.view, self.period, self._table_style, self._prompt = view, period, table_style, prompt
        self._num_envs = self._env.num_envs
        self._labels = labels or [f"env {i}" for i in range(self._num_envs)]
        self._max_calls = max_calls
        self._goal = self._env.command_manager.get_term("goal")
        self._goal_clip = tuple(self._env.cfg.observations["hl_actor"].terms["goal_distance"].clip)
        camera = self._env.scene[LLM_SENSOR].cfg
        self._system = system_prompt(view, period, (camera.height, camera.width), table=table_style, version=prompt)

        self._step = 0
        self._plans = [Decision() for _ in range(self._num_envs)]
        self._history = [deque(maxlen=history) for _ in range(self._num_envs)]
        self._notes = [""] * self._num_envs
        self._new_episode = [True] * self._num_envs
        self._episode = [0] * self._num_envs
        self._call = [0] * self._num_envs
        self.active = [True] * self._num_envs  # run.py clears an entry when that robot has run its episodes.
        # Per robot, its latest call: {"call", "images": [(name, PNG)], "depth": (rows, columns) m, "decision"};
        # None before the first.
        self.last: list[dict | None] = [None] * self._num_envs
        self.errors = 0

        self._log = self._frames = None
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "system_prompt.txt").write_text(self._system)
            self._log = (log_dir / "decisions.jsonl").open("a")
            if save_frames:
                self._frames = log_dir / "frames"
                self._frames.mkdir(exist_ok=True)

    def act(self) -> torch.Tensor:
        """The high-level action (N, 3) of the next step; asks the backend first when a new plan is due."""
        step_in_plan = self._step % self.period
        if step_in_plan == 0:
            self._replan()
        self._step += 1
        device = self._wrapper.device
        speed = torch.tensor([plan.speed for plan in self._plans], device=device).clamp(*VX_RANGE)
        height = torch.tensor([plan.height for plan in self._plans], device=device).clamp(H_LOW, H_REF)
        trigger = torch.tensor([plan.jump_at_step == step_in_plan for plan in self._plans], device=device)
        return to_action(speed, height, trigger)

    def episode_ended(self, dones: torch.Tensor) -> None:
        """Forget what the robots of ``dones`` (N,) knew: their episode ended and the env has reset them."""
        for env_id in dones.nonzero().flatten().tolist():
            self._history[env_id].clear()
            self._notes[env_id] = ""
            self._new_episode[env_id] = True
            self._episode[env_id] += 1
            self._call[env_id] = 0
            # Until the next plan the robot stands still: the old plan belonged to another place in the lane.
            self._plans[env_id] = Decision()

    def close(self) -> None:
        if self._log is not None:
            self._log.close()

    def _replan(self) -> None:
        env_ids = [i for i in range(self._num_envs) if self.active[i]]
        for env_id in range(self._num_envs):
            if not self.active[env_id]:
                self._plans[env_id] = Decision()  # Parked: zero speed, standing, no jump.
        if not env_ids:
            return
        if self._max_calls is not None and self._backend.calls + len(env_ids) > self._max_calls:
            raise CallBudgetExhausted(f"{self._backend.calls} calls made, budget {self._max_calls}")

        hl_actor = self._wrapper.get_observations()["hl_actor"].cpu()
        camera = self._env.scene[LLM_SENSOR].data
        depth = camera.depth[..., 0].cpu().numpy()
        rgb = camera.rgb.cpu().numpy()
        truth = self._truth()

        requests, texts, pictures = [], [], []
        for env_id in env_ids:
            request, images = build_request(
                self.view, self.period, hl_actor[env_id], depth[env_id], rgb[env_id], list(self._history[env_id]),
                self._notes[env_id], self._new_episode[env_id], self._goal_clip, self._table_style, self._prompt)
            requests.append(request)
            texts.append(request.text)
            pictures.append(images)
            if self._frames is not None:
                for name, png in images:
                    (self._frames / f"{self._tag(env_id)}_{name}.png").write_bytes(png)

        decisions = self._backend.decide(requests, env_ids)
        for env_id, text, images, decision in zip(env_ids, texts, pictures, decisions):
            previous = self._plans[env_id]
            self.last[env_id] = {"call": self._call[env_id], "images": images, "depth": depth[env_id],
                                 "decision": decision}
            if decision.error is None:
                self._plans[env_id] = decision
                self._history[env_id].append(decision.commands())
                self._notes[env_id] = decision.note
            else:
                # No usable answer: keep driving as before, without a jump, and ask again at the next tick.
                self.errors += 1
                self._plans[env_id] = Decision(speed=previous.speed, height=previous.height)
            self._new_episode[env_id] = False
            if self._log is not None:
                self._log.write(json.dumps({
                    "step": self._step, "env": env_id, "lane": self._labels[env_id],
                    "episode": self._episode[env_id], "call": self._call[env_id], "tag": self._tag(env_id),
                    "truth": truth[env_id], "decision": decision.commands(), "error": decision.error,
                    "confidence": decision.confidence,
                    "model": decision.model, "latency_s": round(decision.latency_s, 3), "usage": decision.usage,
                    "text": text}) + "\n")
                self._log.flush()
            self._call[env_id] += 1

    def _tag(self, env_id: int) -> str:
        return f"env{env_id:02d}_ep{self._episode[env_id]}_call{self._call[env_id]:03d}"

    def _truth(self) -> list[dict]:
        """What the simulator knows at this tick, for the log only: it is never shown to the backend."""
        kinds = self._goal.kinds
        obstacle = self._goal.next_obstacles(1)[:, 0].cpu()
        goal = self._goal.distance().cpu()
        records = []
        for env_id in range(self._num_envs):
            one_hot = obstacle[env_id, :len(kinds)]
            records.append({
                "next": kinds[int(one_hot.argmax())] if one_hot.any() else None,
                "front_m": round(float(obstacle[env_id, len(kinds)]), 3),
                "end_m": round(float(obstacle[env_id, len(kinds) + 1]), 3),
                "goal_m": round(float(goal[env_id]), 3)})
        return records
