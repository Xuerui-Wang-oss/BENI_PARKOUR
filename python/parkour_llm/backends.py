"""Who answers a decision request: Claude, or a scripted oracle that reads the simulator.

Both take the requests of one decision tick, one per robot that needs a plan, and return one ``Decision`` each.
"""

import base64
import json
import math
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import torch
from mjlab.envs import ManagerBasedRlEnv

from beni_rl_crawl.recovery import H_LOW, H_REF
from parkour_rl.check import (BRAKE_GAIN, BRAKE_MARGIN, CROUCH_DISTANCE, CROUCH_HOLD_STEPS, CRUISE_SPEED, LEAD_KINDS,
                              THIN_CUBE_SPEED, TRIGGER_DISTANCE)
from parkour_rl.hierarchy import PRESETTLE_STEPS

from .prompt import DECISION_SCHEMA, STEP_S

DEFAULT_MODEL = "claude-opus-5-5"
# Models that take the server-side refusal fallback: a request a safety classifier declines is re-run on another
# model inside the same call instead of coming back empty.
FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-2", "claude-fable-5-1")
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# USD per million tokens: input, output, cache read (Anthropic's list prices as of 2026-11; the Haiku cache-read
# price is assumed at 0.1 x input). A cache write is counted at 1.25 x input. Used only for the cost estimate that
# a run prints; the bill is what counts.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-fable-5-2": (8.00, 40.00, 0.20),
    "jev-latest": (0.042, 0.00, 0.042),  # TypeSafe: billed per input token, output free.
}
# Models with a fast mode: the same model with quicker output, at FAST_PRICE_FACTOR times the price.
FAST_MODELS = ("claude-opus-5-5", "claude-fable-5-2")
FAST_PRICE_FACTOR = 2.0
MODEL_LABELS = {
    "claude-opus-5-5": "Claude Opus 5.5",
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "claude-haiku-5-5": "Claude Haiku 5.5",
    "claude-fable-5-2": "Claude Fable 5.2",
    "jev-latest": "Jev (TypeSafe)",
}
JEV_MODELS = ("jev-latest",)
JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_SPEEDS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5)  # The speeds Jev chooses between (m/s).
# Variations of a Jev call that a run can switch on, to measure what each is worth:
#   split     the obstacle question tells a box from a bar (otherwise one answer, "box_or_bar")
#   examples  the state carries solved cases: tables from other lanes with the right answer for each
#   ranges    the jump question offers "jump" and "wait" as separate answers per obstacle, each with its distance,
#             instead of one "now" answer that lists every trigger distance
#   barspeed  every speed answer says when it applies, so that 1.2 m/s is spelled out as the speed for a bar
#   oneface   one jump rule for every low face, box or bar alike: the same speed and the same trigger distance
JEV_OPTIONS = ("split", "examples", "ranges", "barspeed", "oneface")
# A box and a bar of every env1 height were cleared at 1.0 m/s when the trigger came with the face 0.55 to 0.70 m
# from the root (scripted triggers, one run per height and distance, 2026-11-22). From the camera, 0.07 m ahead of
# the root, the middle of that window is:
FACE_TRIGGER = 0.555
JEV_FACE_RULE = ("One jump rule for every low face on the floor, box or bar alike, in place of any other box or bar "
                 "distance in the manual: trigger the jump when the face is {face:.2f} m ahead of the camera or nearer. A face "
                 "that is still farther away is not jumped yet. A bar needs no other speed than a box. Braking for "
                 "the goal stays as the manual says.")
JEV_PICTURE = "<depth_picture>"  # Where the picture stands in the state, when a view with a picture is sent.
EXAMPLES = Path(__file__).parent / "examples"  # examples/<table style>.json, written by make_examples.py.
USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


def load_dotenv() -> None:
    """Set the variables of the repository's ``.env`` file (KEY=value lines) that the environment does not have."""
    path = Path(__file__).resolve().parents[2] / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            key, separator, value = line.strip().partition("=")
            if separator and key and not key.startswith("#"):
                os.environ.setdefault(key.removeprefix("export ").strip(), value.strip().strip("'\""))


def has_credentials(model: str = DEFAULT_MODEL) -> bool:
    """Whether the environment (after ``load_dotenv``) holds a key for ``model``'s provider."""
    if model in JEV_MODELS:
        return bool(jev_key())
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def jev_key() -> str:
    """The TypeSafe API key: ``JEV_KEY`` or ``TYPESAFE_API_KEY``."""
    return os.environ.get("JEV_KEY") or os.environ.get("TYPESAFE_API_KEY") or ""


@dataclass
class Request:
    """One robot's question: the system prompt, the observation text, and the PNG images in order."""
    system: str
    text: str
    images: list[bytes]


@dataclass
class Decision:
    """One answer. ``jump_at_step`` is -1 for no jump. ``error`` is set when no usable answer came back."""
    scene: str = ""
    speed: float = 0.
    height: float = H_REF
    jump_at_step: int = -1
    note: str = ""
    error: str | None = None
    model: str = ""
    latency_s: float = 0.
    usage: dict = field(default_factory=dict)
    # For a model that answers with probabilities (Jev): per question, its choice and how sure it was.
    confidence: dict = field(default_factory=dict)

    def commands(self) -> dict:
        return {"scene": self.scene, "speed": self.speed, "height": self.height,
                "jump_at_step": self.jump_at_step, "note": self.note}


def cost_usd(model: str, usage: dict, fast: bool = False) -> float | None:
    """Estimated cost of ``usage`` (summed token counts) on ``model``; None for a model without a listed price."""
    if model not in PRICES:
        return None
    price_in, price_out, price_read = PRICES[model]
    factor = FAST_PRICE_FACTOR if fast and model in FAST_MODELS else 1.
    return factor * (usage.get("input_tokens", 0) * price_in + usage.get("output_tokens", 0) * price_out
                     + usage.get("cache_read_input_tokens", 0) * price_read
                     + usage.get("cache_creation_input_tokens", 0) * price_in * 1.25) / 1e6


class ClaudeBackend:
    """Claude through the Messages API, one request per robot, sent in parallel.

    Credentials come from the environment (``ANTHROPIC_API_KEY``). The simulation waits for the answers, so the
    SDK retries rate limits and server errors many times before giving up. A request that still fails, or is
    declined, comes back as a ``Decision`` with ``error`` set, and the policy keeps the robot on its previous
    commands; a request the API rejects outright (bad key, bad parameters) stops the run.
    """

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "low", workers: int = 16,
                 fast: bool = False) -> None:
        self.model, self.effort = model, effort
        self.fast = fast and model in FAST_MODELS  # Fast mode, where the model has one.
        self._client = anthropic.Anthropic(max_retries=10)
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self.calls = 0
        self.usage = dict.fromkeys(USAGE_KEYS, 0)

    def decide(self, requests: list[Request], env_ids: list[int]) -> list[Decision]:
        del env_ids
        decisions = list(self._pool.map(self.ask, requests))
        self.calls += len(decisions)
        for decision in decisions:
            for key in USAGE_KEYS:
                self.usage[key] += decision.usage.get(key, 0)
        return decisions

    def ask(self, request: Request) -> Decision:
        """Answer one request; safe to call from several threads."""
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                "data": base64.standard_b64encode(image).decode()}}
                   for image in request.images]
        content.append({"type": "text", "text": request.text})
        arguments = dict(
            model=self.model,
            max_tokens=16000,
            # The system prompt is the same on every call of a run, so it is cached; the frame and state follow it.
            system=[{"type": "text", "text": request.system, "cache_control": {"type": "ephemeral"}}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": DECISION_SCHEMA}},
            messages=[{"role": "user", "content": content}],
        )
        if self.fast:
            arguments["speed"] = "fast"
        start = time.perf_counter()
        try:
            if self.model in FALLBACK_MODELS:
                response = self._client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **arguments)
            else:
                response = self._client.messages.create(**arguments)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError,
                anthropic.BadRequestError):
            raise  # A wrong key, model or parameter fails every call alike.
        except anthropic.RateLimitError as error:
            return Decision(error=f"rate limited: {error.message}", latency_s=time.perf_counter() - start)
        except anthropic.APIStatusError as error:
            return Decision(error=f"API error {error.status_code}: {error.message}",
                            latency_s=time.perf_counter() - start)
        except anthropic.APIConnectionError as error:
            return Decision(error=f"connection error: {error}", latency_s=time.perf_counter() - start)

        decision = Decision(model=response.model, latency_s=time.perf_counter() - start,
                            usage={key: getattr(response.usage, key, 0) or 0 for key in USAGE_KEYS})
        if response.stop_reason != "end_turn":
            decision.error = f"stopped with {response.stop_reason}"
            return decision
        try:
            answer = json.loads(next(block.text for block in response.content if block.type == "text"))
            decision.scene, decision.note = str(answer["scene"]), str(answer["note"])
            decision.speed, decision.height = float(answer["speed"]), float(answer["height"])
            decision.jump_at_step = int(answer["jump_at_step"])
        except (StopIteration, ValueError, KeyError, TypeError) as error:
            decision.error = f"unreadable answer: {error}"
        return decision


class JevBackend:
    """TypeSafe's Jev, a model that answers typed questions about a state in one fast parallel pass.

    Jev writes no text, so the decision is asked as five separate questions over the same state: the manual (the
    system prompt without its reply section) and the observation text. Three are the commands, each a choice among
    fixed options; two describe the scene and serve as the memory that the next call is told. Jev answers every
    question on its own: no answer can use another answer of the same call, and there is no reasoning step. The
    request carries text only, so it goes with the "table" view.
    """

    fast = False

    def __init__(self, model: str = "jev-latest", period: int = 5, workers: int = 16, options=(),
                 table: str = "camera") -> None:
        self.model, self.effort = model, None
        self._key = jev_key()
        self.options = tuple(options)
        self._questions = jev_questions(period, split="split" in self.options, ranges="ranges" in self.options,
                                        barspeed="barspeed" in self.options, oneface="oneface" in self.options)
        self._face_rule = (JEV_FACE_RULE.format(face=FACE_TRIGGER + period * STEP_S / 2)
                           if "oneface" in self.options else None)
        self._examples = None
        if "examples" in self.options:
            self._examples = json.loads((EXAMPLES / f"{table}.json").read_text())
            for example in self._examples:
                answers = example["answers"]
                if "split" not in self.options and answers["obstacle"] in ("box", "bar"):
                    answers["obstacle"] = "box_or_bar"
                if "ranges" in self.options:  # The jump answer in the words of the "ranges" question.
                    kind = answers["obstacle"] if answers["obstacle"] in ("pit", "box", "bar") else None
                    if answers["obstacle"] == "box_or_bar":
                        kind = "bar" if "bar" in example["situation"] else "box"
                    answers["jump"] = "no_jump" if kind is None else (
                        f"jump_{kind}" if answers["jump"] == "now" else f"wait_{kind}")
            for question in self._questions.values():
                question["instructions"] += (" `examples` holds tables like the one in `now`, from other lanes, "
                                             "each with its right answers.")
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self.calls = 0
        self.usage = dict.fromkeys(USAGE_KEYS, 0)

    def decide(self, requests: list[Request], env_ids: list[int]) -> list[Decision]:
        del env_ids
        decisions = list(self._pool.map(self.ask, requests))
        self.calls += len(decisions)
        for decision in decisions:
            for key in USAGE_KEYS:
                self.usage[key] += decision.usage.get(key, 0)
        return decisions

    def ask(self, request: Request) -> Decision:
        """Answer one request; safe to call from several threads."""
        # The manual is the system prompt up to its section on the reply's fields.
        state = {"manual": request.system.split("\n## Your reply")[0], "now": request.text}
        payload = {"model": self.model, "questions": self._questions, "state": state}
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        if self._examples is not None:
            state["examples"] = self._examples
        if self._face_rule is not None:
            state["jump_rule"] = self._face_rule
        if request.images:
            # The picture of the request's view, as an asset that stands in the state under "picture".
            state["picture"] = JEV_PICTURE
            payload["assets"] = {JEV_PICTURE: {"type": "image", "format": "png",
                                               "data_base64": base64.standard_b64encode(request.images[0]).decode()}}
            headers["TypeSafe-Danger-Zone-Images"] = "true"
        body = json.dumps(payload).encode()
        start = time.perf_counter()
        answer, error = None, "no answer"
        for attempt in range(4):
            call = urllib.request.Request(JEV_URL, body, headers)
            try:
                with urllib.request.urlopen(call, timeout=30) as response:
                    answer = json.load(response)
                break
            except urllib.error.HTTPError as failure:
                detail = failure.read().decode(errors="replace")[:300]
                if failure.code == 402:  # Out of credits: every further call would fail the same way.
                    raise RuntimeError("The TypeSafe account has no API credits left; add credits in the TypeSafe "
                                       "console to use Jev again.") from None
                if failure.code in (400, 401, 403, 404, 413, 422):
                    raise RuntimeError(f"TypeSafe API rejected the request (HTTP {failure.code}): {detail}") from None
                error = f"TypeSafe API error {failure.code}: {detail}"  # 429, 529, 5xx: wait and try again.
            except (urllib.error.URLError, TimeoutError) as failure:
                error = f"connection error: {failure}"
            time.sleep(0.5 * 2 ** attempt)
        decision = Decision(model=self.model, latency_s=time.perf_counter() - start)
        if answer is None:
            decision.error = error
            return decision
        decision.model = answer.get("model", self.model)
        decision.usage = {"input_tokens": answer.get("usage", {}).get("input_tokens", 0),
                          "output_tokens": answer.get("usage", {}).get("output_tokens", 0)}
        try:
            answers = answer["answers"]
            pick = {name: answers[name]["choice"] for name in self._questions}
            decision.speed = float(pick["speed"])
            decision.height = H_LOW if pick["height"] == "crouch" else H_REF
            decision.jump_at_step = 0 if pick["jump"] == "now" or pick["jump"].startswith("jump_") else -1
            where = "" if pick["obstacle"] == "none" else f", {pick['distance'].replace('_', ' ')}"
            decision.scene = f"{pick['obstacle'].replace('_', ' ')}{where}"
            decision.confidence = {name: {"choice": pick[name], "confidence": round(answers[name]["confidence"], 3)}
                                   for name in self._questions}
        except (KeyError, ValueError, TypeError) as failure:
            decision.error = f"unreadable answer: {failure!r}"
        return decision


def jev_questions(period: int, split: bool = False, ranges: bool = False, barspeed: bool = False,
                  oneface: bool = False) -> dict:
    """The five questions of a Jev call. Every one stands alone: Jev cannot use one answer in another.

    The jump is asked as "now or not": in our runs Jev never placed a trigger on a later step of its plan, it
    fired when the obstacle was already close, so its triggers came late by up to one call's travel. Each trigger
    distance in the question is therefore the tuned one plus half the ground covered before the next call, and the
    way to make the timing finer is to call more often (a smaller ``period``).
    """
    # Half the ground covered before the next call (m): a trigger then lands within that much of the tuned point.
    reach = {speed: speed * period * STEP_S / 2 for speed in (1.0, 1.2)}
    faces = {"box_or_bar": "a low vertical face standing on the floor, under 0.15 m tall"}
    if split:
        faces = {"box": "a low vertical face standing on the floor whose top runs on away from the robot: the rows "
                        "above the face stay raised at growing distances",
                 "bar": "a low vertical face standing on the floor that is only 0.1 m deep: the rows just above it "
                        "are back at floor level, farther away"}
    pit, box, bar = 0.38 + reach[1.0], 0.53 + reach[1.0], 0.62 + reach[1.2]  # Trigger ranges (m from the camera).
    jump = {
        "now": f"a pit's near edge is {pit:.2f} m ahead or nearer, or a box's face is {box:.2f} m ahead or nearer, or "
               f"a bar is {bar:.2f} m ahead or nearer, and the obstacle is still ahead of the robot",
        "none": "nothing to jump, or the obstacle is still farther away than that, or it is a tunnel or a ramp, or a "
                "jump is already in progress or pending",
    }
    if oneface:
        face = FACE_TRIGGER + reach[1.0]
        jump = {
            "now": f"a pit's near edge is {pit:.2f} m ahead or nearer, or a low face on the floor (a box or a bar, "
                   f"the same rule for both) is {face:.2f} m ahead or nearer, and the obstacle is still ahead of the "
                   "robot",
            "none": f"nothing to jump, or a pit's edge is still farther than {pit:.2f} m, or a low face is still "
                    f"farther than {face:.2f} m, or it is a tunnel or a ramp, or a jump is already in progress or "
                    "pending",
        }
    if ranges:
        jump = {
            "jump_pit": f"trigger: a pit's near edge is {pit:.2f} m ahead or nearer",
            "wait_pit": f"not yet: a pit is ahead, its near edge still farther than {pit:.2f} m",
            "jump_box": f"trigger: a box's face is {box:.2f} m ahead or nearer",
            "wait_box": f"not yet: a box is ahead, its face still farther than {box:.2f} m",
            "jump_bar": f"trigger: a bar is {bar:.2f} m ahead or nearer",
            "wait_bar": f"not yet: a bar is ahead, still farther than {bar:.2f} m",
            "no_jump": "no trigger: open floor, a tunnel, a ramp, or a jump already in progress or pending",
        }
    speeds = {f"{speed:.1f}": None for speed in JEV_SPEEDS}
    if barspeed:
        speeds = {f"{speed:.1f}": "braking for the goal: about 2 x (distance to goal - 0.1)" for speed in JEV_SPEEDS}
        speeds |= {"0.0": "stopped: inside the goal window, 0.1 m or less to the goal",
                   "1.0": "cruising: open floor, a pit, a box, a tunnel or a ramp ahead, and the goal 0.6 m away or more",
                   "1.2": "a bar is the next obstacle, within about 1.5 m: a bar is cleared best at this speed",
                   "1.5": "never needed"}
    return {
        "obstacle": {
            "type": "choice",
            "instructions": "What is the nearest obstacle ahead of the robot, according to the depth table in `now`? "
                            "Read the table as `manual` explains.",
            "criteria": {
                "none": "open floor ahead: nothing but floor, side walls and perhaps the far end wall",
                "pit": "a gap in the floor: the distance jumps between neighbouring rows and rows read below floor",
                **faces,
                "ramp": "the floor rises steadily over several rows",
                "tunnel": "something overhead with floor running on underneath",
                "mid_jump": "the robot is in the air or landing and the view is not level",
            },
        },
        "distance": {
            "type": "choice",
            "instructions": "How far ahead of the camera, in metres, is the nearest obstacle in the depth table in "
                            "`now`: the face of a box or bar, the near edge of a pit, the foot of a ramp, the slab "
                            "of a tunnel?",
            "criteria": {"no_obstacle": None, "under_0.3_m": None, "0.3_to_0.45_m": None, "0.45_to_0.6_m": None,
                         "0.6_to_0.8_m": None, "0.8_to_1.2_m": None, "1.2_to_2_m": None, "over_2_m": None},
        },
        "speed": {
            "type": "choice",
            "instructions": "Forward speed in m/s for the next steps, by the rules in `manual`: cruise at 1.0, take "
                            "a bar at 1.2, and near the goal slow to about 2 x (distance to goal - 0.1), stopping "
                            "inside the goal window.",
            "criteria": speeds,
        },
        "height": {
            "type": "choice",
            "instructions": "Body height for the next steps, by the rules in `manual` and the recent decisions in "
                            "`now`.",
            "criteria": {
                "stand": "standing height, 0.110 m: everywhere except at a tunnel",
                "crouch": "lowest crouch, 0.067 m: a tunnel slab is within about 0.7 m ahead, or the robot is under "
                          "it, or it went under it less than 0.5 s ago",
            },
        },
        "jump": {
            "type": "choice",
            "instructions": "Should a jump be triggered right now? Look at the depth table in `now`. You will not be "
                            f"asked again for {period * STEP_S:.2f} s, in which the robot moves on, so trigger as "
                            "soon as the obstacle is inside its trigger range; a trigger that comes late drops the "
                            "robot into a pit or runs it into a face.",
            "criteria": jump,
        },
    }


def make_backend(model: str, effort: str = "low", fast: bool = False, period: int = 5, workers: int = 16,
                 options=(), table: str = "camera"):
    """The backend that asks ``model``: Jev through TypeSafe's API, anything else through Anthropic's.

    ``options`` (JEV_OPTIONS) and the table style ``table`` matter to Jev only.
    """
    if model in JEV_MODELS:
        return JevBackend(model, period, workers, options, table)
    return ClaudeBackend(model, effort, workers, fast)


class OracleBackend:
    """The scripted driver of ``parkour_rl.check`` answering in the LLM's place.

    It ignores the requests and reads the next obstacle and the goal from the simulator, so it is not a fair
    controller. It exists to test the plumbing without an API key, and its score is what perfect perception gets
    out of the same interface: one plan every ``period`` steps, held, with at most one jump trigger in it.
    """

    model = "oracle"

    def __init__(self, env: ManagerBasedRlEnv, period: int) -> None:
        self._goal = env.command_manager.get_term("goal")
        self._kinds = self._goal.kinds
        self._period = period
        self._presettle_s = PRESETTLE_STEPS * env.step_dt
        self._hold_calls = math.ceil(CROUCH_HOLD_STEPS / period)
        self._crouch_calls = torch.zeros(env.num_envs, dtype=torch.long)
        self.calls = 0
        self.usage = dict.fromkeys(USAGE_KEYS, 0)

    def decide(self, requests: list[Request], env_ids: list[int]) -> list[Decision]:
        del requests
        obstacle = self._goal.next_obstacles(1)[:, 0].cpu()  # (N, K + 3): one-hot kind, distance to front, ...
        goal_distance = self._goal.distance().cpu()
        decisions = []
        for env_id in env_ids:
            one_hot, front = obstacle[env_id, :len(self._kinds)], float(obstacle[env_id, len(self._kinds)])
            kind = self._kinds[int(one_hot.argmax())] if one_hot.any() else None
            cruise = THIN_CUBE_SPEED if kind == "thin_cube" else CRUISE_SPEED
            speed = min(max(BRAKE_GAIN * (float(goal_distance[env_id]) - BRAKE_MARGIN), 0.), cruise)
            travel = speed * self._period * STEP_S  # Distance covered before the next call (m).

            # Low by the time the slab is CROUCH_DISTANCE ahead, and for the hold after the root has left it.
            near_slab = kind == "hollow" and front - travel < CROUCH_DISTANCE
            crouch = near_slab or self._crouch_calls[env_id] > 0
            self._crouch_calls[env_id] = self._hold_calls if near_slab else max(int(self._crouch_calls[env_id]) - 1, 0)

            jump_at_step = -1
            if kind in TRIGGER_DISTANCE and front > 0. and speed > 0.:
                lead = speed * self._presettle_s if kind in LEAD_KINDS else 0.
                steps = max(math.ceil((front - lead - TRIGGER_DISTANCE[kind]) / (speed * STEP_S)), 0)
                if steps < self._period:
                    jump_at_step = steps
            scene = f"{kind} {front:+.2f} m ahead" if kind else "no obstacle left"
            decisions.append(Decision(scene=scene, speed=speed, height=H_LOW if crouch else H_REF,
                                      jump_at_step=jump_at_step, model=self.model))
        self.calls += len(decisions)
        return decisions


class ReplayBackend:
    """Answers with the decisions an earlier run logged, in their order: no model is called and nothing is paid.

    The simulation is seeded, so the same lanes with the same decisions give the same episodes again. That makes a
    replay the way to get videos of a run after the fact (``run.py --replay DIR --videos``).
    """

    def __init__(self, run_dir: Path) -> None:
        self._logged: dict[int, list[dict]] = {}
        with (run_dir / "decisions.jsonl").open() as log:
            for line in log:
                row = json.loads(line)
                self._logged.setdefault(row["env"], []).append(row)
        self._next = dict.fromkeys(self._logged, 0)
        self.model = next((row["model"] for rows in self._logged.values() for row in rows if row["model"]), "replay")
        self.calls = 0
        self.usage = dict.fromkeys(USAGE_KEYS, 0)

    def decide(self, requests: list[Request], env_ids: list[int]) -> list[Decision]:
        del requests
        decisions = []
        for env_id in env_ids:
            rows, index = self._logged.get(env_id, []), self._next.get(env_id, 0)
            if index >= len(rows):
                raise RuntimeError(f"the replayed run logged {len(rows)} calls of env {env_id}; this run asks for "
                                   "one more, so it has left the logged course")
            row = rows[index]
            self._next[env_id] = index + 1
            decisions.append(Decision(**row["decision"], error=row["error"], model=row["model"],
                                      latency_s=row["latency_s"], confidence=row.get("confidence") or {}))
        self.calls += len(env_ids)
        return decisions
