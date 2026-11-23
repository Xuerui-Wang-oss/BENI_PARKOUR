"""The local server behind the "Claude" driver of the web demo (``parkour_web``).

    python -m parkour_llm.server                 # http://127.0.0.1:8787

The page simulates the robot in the browser. When Claude drives, it posts the depth frame and the 24 body-state
numbers here every few steps; this server builds the same request as ``parkour_llm.run`` does, asks Claude, and
answers with the decision and with everything that was sent, so the page can show both.

    GET  /api/config                    models, default model, efforts, whether a key is set
    GET  /api/system?period=5           the system prompt
    POST /api/decide                    {model, effort, fast, period, hl_actor[24], depth[rows*columns], rows, columns,
                                         history, note, new_episode} -> {decision, input, usage, cost_usd, ...}

It also serves the built page (``parkour_web/dist``, after ``npm run build``) so that one process is enough; during
development the Vite dev server proxies ``/api`` here instead.

The API key stays on this machine (``ANTHROPIC_API_KEY`` in ``BENI_PARKOUR/.env``). The server listens on 127.0.0.1
only unless ``--host`` says otherwise: whoever can reach it can spend the key's money.
"""

import argparse
import base64
import json
import mimetypes
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import anthropic
import numpy as np

from .backends import (DEFAULT_MODEL, FAST_MODELS, FAST_PRICE_FACTOR, JEV_MODELS, MODEL_LABELS, PRICES, cost_usd,
                       has_credentials, jev_questions, load_dotenv, make_backend)
from .observe import build_request
from .prompt import system_prompt
from .views import depth_png

REPOSITORY = Path(__file__).resolve().parents[2]
DIST = REPOSITORY / "parkour_web" / "dist"
EFFORTS = ("low", "medium", "high")
# The model the page preselects: in our runs it answered in about 2 s, against 3 to 7 s for the default Opus.
WEB_MODEL = "claude-sonnet-5-5"
# Instructions for the Claude models on the page. "v2" is being tested; the page changes over when it has proved itself.
WEB_PROMPT = "v1"
# High-level steps (0.05 s) per call on the page. Jev answers in about 0.15 s and never schedules a jump for a later
# step, so it is asked more often: on the env1 pits it crossed 15 of 20 at 2 steps against 11 of 20 at 5.
PERIOD, JEV_PERIOD = 5, 2
GOAL_CLIP = (-1.0, 2.0)  # Range of the goal distance in the page's observation (parkour_web/src/sim/hierarchy.js).
MAX_BODY = 2_000_000  # Bytes of a request body; a depth frame is about 40 kB of JSON.
# The page is cross-origin isolated (WebAssembly threads), so every file needs these.
ISOLATION = {"Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp",
             "Cross-Origin-Resource-Policy": "same-origin"}

_backends: dict[tuple[str, str, bool, int], object] = {}
_lock = threading.Lock()
_log_path: Path | None = None


def prompt_for(model: str) -> str:
    """The version of the instructions that ``model`` gets on the page (``prompt.PROMPTS``)."""
    return "v1" if model in JEV_MODELS else WEB_PROMPT  # Jev's questions were tuned with v1.


def backend_for(model: str, effort: str, fast: bool, period: int):
    with _lock:
        if (model, effort, fast, period) not in _backends:
            _backends[model, effort, fast, period] = make_backend(model, effort, fast, period, workers=1)
        return _backends[model, effort, fast, period]


def decide(body: dict) -> dict:
    """Answer one ``/api/decide`` request body; raises ValueError for a body the page should not have sent."""
    model, effort = body.get("model", DEFAULT_MODEL), body.get("effort", "low")
    if model not in MODEL_LABELS:
        raise ValueError(f"unknown model {model!r}")
    if effort not in EFFORTS:
        raise ValueError(f"unknown effort {effort!r}")
    period = int(body.get("period", 5))
    rows, columns = int(body.get("rows", 60)), int(body.get("columns", 80))
    if not 1 <= period <= 20 or rows * columns > 320 * 240:
        raise ValueError("period or frame size out of range")
    hl_actor = [float(x) for x in body["hl_actor"]]
    depth = np.asarray(body["depth"], dtype=np.float32)
    if len(hl_actor) != 24 or depth.size != rows * columns:
        raise ValueError("hl_actor needs 24 numbers and depth rows x columns numbers")
    history = [{"scene": str(h["scene"]), "speed": float(h["speed"]), "height": float(h["height"]),
                "jump_at_step": int(h["jump_at_step"])} for h in body.get("history", [])][-3:]

    frame = depth.reshape(rows, columns)
    # Jev is sent text only: the table view. The page still gets the picture, marked as not sent.
    view = "table" if model in JEV_MODELS else "depth"
    request, images = build_request(view, period, hl_actor, frame, None, history,
                                    str(body.get("note", "")), bool(body.get("new_episode")), GOAL_CLIP,
                                    prompt=prompt_for(model))
    fast = bool(body.get("fast")) and model in FAST_MODELS
    decision = backend_for(model, effort, fast, period).ask(request)
    cost = cost_usd(model, decision.usage, fast)
    encode = lambda png: base64.standard_b64encode(png).decode()  # noqa: E731
    answer = {
        "decision": decision.commands(), "error": decision.error, "model": decision.model or model,
        "latency_s": round(decision.latency_s, 3), "usage": decision.usage,
        "cost_usd": None if cost is None else round(cost, 6), "confidence": decision.confidence,
        "input": {"text": request.text, "view": view,
                  "images": [{"name": name, "png_base64": encode(png)} for name, png in images],
                  "preview_png_base64": None if images else encode(depth_png(frame))},
    }
    if _log_path is not None:
        with _lock, _log_path.open("a") as log:
            log.write(json.dumps({"time": time.time(), "run": body.get("run"), "call": body.get("call"),
                                  **{key: value for key, value in answer.items() if key != "input"},
                                  "text": request.text}) + "\n")
    return answer


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        if self.path.startswith("/api/"):
            sys.stderr.write(f"{self.log_date_time_string()}  {format % args}\n")

    def _send(self, status: int, body: bytes, content_type: str, cache: str = "no-store") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        for name, value in ISOLATION.items():
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        if url.path == "/api/config":
            models = [{"id": model, "label": label, "input_usd_per_mtok": PRICES[model][0],
                       "output_usd_per_mtok": PRICES[model][1], "fast": model in FAST_MODELS,
                       "typed": model in JEV_MODELS, "available": has_credentials(model),
                       "period": JEV_PERIOD if model in JEV_MODELS else PERIOD}
                      for model, label in MODEL_LABELS.items()]
            available = [model["id"] for model in models if model["available"]]
            self._json({
                "key": bool(available), "efforts": list(EFFORTS), "period": PERIOD,
                "default_model": WEB_MODEL if WEB_MODEL in available else (available or [WEB_MODEL])[0],
                "fast_price_factor": FAST_PRICE_FACTOR, "models": models})
        elif url.path == "/api/system":
            query = parse_qs(url.query)
            period = int(query.get("period", ["5"])[0])
            typed = query.get("model", [""])[0] in JEV_MODELS
            version = prompt_for(query.get("model", [""])[0])
            self._json({"system": system_prompt("table" if typed else "depth", period, reply=not typed,
                                                version=version),
                        "questions": jev_questions(period) if typed else None})
        elif url.path.startswith("/api/"):
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        else:
            self._static(url.path)

    do_HEAD = do_GET  # noqa: N815

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/decide":
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", 0))
        if not 0 < length <= MAX_BODY:
            self._json({"error": "request body missing or too large"}, HTTPStatus.BAD_REQUEST)
            return
        try:
            self._json(decide(json.loads(self.rfile.read(length))))
        except (ValueError, KeyError, TypeError) as error:
            self._json({"error": f"bad request: {error}"}, HTTPStatus.BAD_REQUEST)
        except anthropic.AuthenticationError:
            self._json({"error": "The Anthropic API rejected the key in BENI_PARKOUR/.env."}, HTTPStatus.BAD_GATEWAY)
        except anthropic.AnthropicError as error:
            self._json({"error": f"Anthropic API: {error}"}, HTTPStatus.BAD_GATEWAY)
        except RuntimeError as error:  # The TypeSafe API rejected the request (backends.JevBackend).
            self._json({"error": str(error)}, HTTPStatus.BAD_GATEWAY)

    def _static(self, path: str) -> None:
        file = (DIST / path.lstrip("/")).resolve()
        if file.is_dir():
            file = file / "index.html"
        if DIST.resolve() not in file.parents or not file.is_file():
            hint = b"" if DIST.is_dir() else b" Build the page first: cd parkour_web && npm run build"
            self._send(HTTPStatus.NOT_FOUND, b"Not found." + hint, "text/plain")
            return
        kind = {".wasm": "application/wasm", ".onnx": "application/octet-stream", ".js": "text/javascript",
                ".mjs": "text/javascript", ".obj": "text/plain"}.get(
                    file.suffix, mimetypes.guess_type(file.name)[0] or "application/octet-stream")
        cache = "public, max-age=31536000, immutable" if path.startswith("/assets/") else "no-cache"
        self._send(HTTPStatus.OK, file.read_bytes(), kind, cache)


def main() -> None:
    global _log_path
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on (default 127.0.0.1: this machine)")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--no-log", action="store_true", help="do not append the calls to runs/web_calls.jsonl")
    args = parser.parse_args()
    load_dotenv()
    if not args.no_log:
        _log_path = Path("runs") / "web_calls.jsonl"
        _log_path.parent.mkdir(exist_ok=True)
    if not any(has_credentials(model) for model in MODEL_LABELS):
        print("No API key found (ANTHROPIC_API_KEY or JEV_KEY in BENI_PARKOUR/.env): the page offers no model.")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"parkour_llm server on http://{args.host}:{args.port}"
          f"{'' if DIST.is_dir() else '  (no built page: API only)'}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
