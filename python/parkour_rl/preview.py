# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""Drive a low-level BENI policy through the parkour lanes with the Viser joystick.

    python -m parkour_rl.preview Beni-Parkour-Preview-Jump --checkpoint-file assets/models/policy_best.pt
    python -m parkour_rl.preview Beni-Parkour-Preview-Crawl \\
        --checkpoint-file logs/rsl_rl/beni_crawl_flat/2026-10-11_17-48-27/model_15500.pt

The viewer's environment selector (Scene > Environment > Select) walks the 36 cells: env i sits on
column i // 9 (0 pit, 1 cube, 2 hollow, 3 ramp_cube) and row i % 9 (0 flat, 1-4 one obstacle of growing
size, 5 two at the limit, 6 the column's kind mixed with others, 7 random with 3-4, 8 random with 5-6).
Keep the default --num-envs; another count breaks this mapping.
"""

import sys

from beni_rl import play as jump_play
from beni_rl_crawl import play as crawl_play

from . import CRAWL_TASK_ID, JUMP_TASK_ID  # Importing the package registers the tasks before mjlab parses the task.


def main():
    if sys.argv[1:] in (["-h"], ["--help"]):
        print(__doc__.rstrip())
        print(f"\nTasks: {JUMP_TASK_ID}, {CRAWL_TASK_ID}. Run '<TASK> --help' for the play options.")
        return
    # Each package's play restores the Jump settings of its own command class from the checkpoint.
    play = jump_play if sys.argv[1:2] == [JUMP_TASK_ID] else crawl_play
    play.main()


if __name__ == "__main__":
    main()
