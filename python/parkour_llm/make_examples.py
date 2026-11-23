"""Pick the solved cases that a Jev call can carry (``--jev examples``) out of a run of the scripted oracle.

    python -m parkour_llm.run env2 --lanes 0:0,1:0,2:0,3:0,4:0,5:0 --backend oracle --view table --table floor \\
        --period 2 --no-frames --out runs/examples_floor
    python -m parkour_llm.make_examples runs/examples_floor floor

The oracle run logs, for every call, the depth table that was sent and what the simulator knew. This script keeps
one table per situation below and writes ``examples/<table style>.json``: the table with the right answers for it.
The lanes are training layouts of env2, so no example comes from a lane of the env1 benchmark. The answers come
from the simulator, which is how the examples can be right; the model that reads them at run time is given nothing
of the lane it drives on beyond its own depth frame.
"""

import json
import sys
from pathlib import Path

from beni_rl_crawl.recovery import H_LOW

from .backends import EXAMPLES

CAMERA_AHEAD = 0.07  # The camera sits this far ahead of the robot's root, which the simulator measures from (m).
NAMES = {"pit": "pit", "cube": "box", "thin_cube": "bar", "hollow": "tunnel", "ramp_cube": "ramp"}
# (label, obstacle kind or None for open floor, nearest and farthest distance from the camera, jump answer)
CASES = (
    ("open floor", None, 2.5, 99., "none"),
    ("pit, still too far to jump", "pit", 0.62, 0.80, "none"),
    ("pit, at the trigger point", "pit", 0.33, 0.42, "now"),
    ("box, still too far to jump", "cube", 0.85, 1.10, "none"),
    ("box, at the trigger point", "cube", 0.46, 0.56, "now"),
    ("bar, still too far to jump", "thin_cube", 0.95, 1.20, "none"),
    ("bar, at the trigger point", "thin_cube", 0.56, 0.66, "now"),
    ("tunnel, close enough to crouch", "hollow", 0.40, 0.70, "none"),
    ("tunnel, still far", "hollow", 1.10, 1.60, "none"),
    ("ramp ahead", "ramp_cube", 0.40, 0.80, "none"),
)
BUCKETS = ((0.3, "under_0.3_m"), (0.45, "0.3_to_0.45_m"), (0.6, "0.45_to_0.6_m"), (0.8, "0.6_to_0.8_m"),
           (1.2, "0.8_to_1.2_m"), (2.0, "1.2_to_2_m"), (float("inf"), "over_2_m"))


def main() -> None:
    run, style = Path(sys.argv[1]), sys.argv[2]
    calls = [json.loads(line) for line in (run / "decisions.jsonl").open()]
    examples = []
    for label, kind, near, far, jump in CASES:
        for call in calls:
            truth, decision = call["truth"], call["decision"]
            distance = truth["front_m"] - CAMERA_AHEAD
            # A level view: not in or just after a jump, where the oracle's own commands are still standing ones.
            steady = decision["jump_at_step"] < 0 or jump == "now"
            if truth["next"] == kind and near <= distance <= far and steady and "Depth table:" in call["text"]:
                table = call["text"].split("Depth table:\n")[1].split("\n\n")[0]
                bucket = "no_obstacle" if kind is None else next(name for limit, name in BUCKETS if distance < limit)
                examples.append({
                    "situation": label, "depth_table": table,
                    "answers": {"obstacle": NAMES.get(kind, "none"), "distance": bucket, "jump": jump,
                                "height": "crouch" if decision["height"] < H_LOW + 0.005 else "stand"}})
                print(f"{label:34s} {call['lane']:10s} call {call['call']:3d}  {distance:5.2f} m")
                break
        else:
            print(f"{label:34s} no call found")
    EXAMPLES.mkdir(exist_ok=True)
    (EXAMPLES / f"{style}.json").write_text(json.dumps(examples, indent=1))
    print(f"{len(examples)} examples -> {EXAMPLES / f'{style}.json'}")


if __name__ == "__main__":
    main()
