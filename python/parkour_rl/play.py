# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""Watch a trained high-level parkour policy on every lane cell in the Viser viewer.

    python -m parkour_rl.play --checkpoint-file logs/rsl_rl/beni_parkour/<run>/model_<iteration>.pt
    python -m parkour_rl.play env1 --checkpoint-file logs/rsl_rl/beni_parkour_env1/<run>/model_<iteration>.pt
    python -m parkour_rl.play env2 --checkpoint-file logs/rsl_rl/beni_parkour_env2/<run>/model_<iteration>.pt

The options are those of mjlab's play script (``PlayConfig``). An optional first argument ``env1`` or ``env2`` plays
Beni-Parkour-Env1 (env1.py) or Beni-Parkour-Env2 (env2.py) instead of Beni-Parkour.

One robot runs on every cell of the 4 x 9 lane grid. The viewer's environment selector (Scene > Environment >
Select) walks the cells: env i sits on column i // 9 (0 pit, 1 cube, 2 hollow, 3 ramp_cube) and row i % 9 (the
difficulty steps described in parkour_terrain.py). Keep the default --num-envs: the cell layout needs exactly 36
envs and refuses any other count. With env1 one robot runs on every cell of the 5 x 5 grid: env i sits on column
i // 5 (0 pit, 1 cube, 2 hollow, 3 ramp_cube, 4 thin_cube) and row i % 5; keep the default --num-envs (25). With
env2 one robot runs on every lane: env i sits on lane column i // 5 and row i % 5, rows 0-3 the training layouts and
row 4 the test layout; keep the default --num-envs (80).

"Reset Environment" in the viewer restarts all envs and clears the memory of the high-level policy and of both
low-level policies. The viewer offers no reset of a single env for this task.

On a CPU the playback is slower than real time and the viewer reports CAPPED: one high-level step of the 36 envs
took about 0.8 s on an Apple M5, which is 1/16 of real time.

Not supported:
    Viewers other than Viser. mjlab's default ``--viewer auto`` picks its native viewer on a Linux desktop (DISPLAY
        or WAYLAND_DISPLAY is set); pass ``--viewer viser`` there. Only the viewer of this file accounts for an env
        step of SUB_STEPS control steps and clears the policy's memory when an episode ends.
    ``--agent zero`` and ``--agent random``. mjlab builds them for the 7 joint actions of the wrapped env, not for
        the 3 high-level actions, and they have no memory to clear: the viewer stops with an error at the first step.
    ``--no-terminations True``. The rewards read the goal_reached termination, so every step fails.
"""

import mjlab
import tyro
from mjlab.scripts import play as mjlab_play

from beni_rl.play import BeniViserPlayViewer

from . import task_from_argv  # Importing the package registers the tasks.
from .hierarchy import SUB_STEPS, ParkourVecEnvWrapper


class ParkourViserPlayViewer(BeniViserPlayViewer):
    """Viser play viewer for ``ParkourVecEnvWrapper``, whose step is one high-level step: SUB_STEPS control steps.

    mjlab's viewer treats every env step as one control step, and it clears the memory of a recurrent policy only
    when all envs are reset. ``_execute_step`` makes up for both after every step. The base class, the viewer of
    beni_rl.play, opens the browser.
    """

    def _execute_step(self) -> bool:
        """Let the policy act for one high-level step in all envs; returns False if the step raised an error."""
        succeeded = super()._execute_step()
        if succeeded:
            # The wrapper simulated SUB_STEPS control steps, of which mjlab's viewer books one.
            unbooked_steps = SUB_STEPS - 1
            # Simulated time (s) the viewer may still spend to stay in real time. Without this the playback runs
            # SUB_STEPS times too fast.
            self._sim_budget -= unbooked_steps * self.env.unwrapped.step_dt
            # Control steps behind the viewer's "Actual RT" readout.
            self._stats_steps += unbooked_steps
            # Zero the policy's memory of the envs whose episode ended in this step, as rsl_rl does in training.
            self.policy.reset(self.env.last_dones)
        return succeeded


def main() -> None:
    """Run mjlab's play on the task with the env wrapper of hierarchy.py and the viewer of this file."""
    task_id = task_from_argv()
    # run_play looks both names up in mjlab's play module when it wraps the env and opens the viewer.
    mjlab_play.RslRlVecEnvWrapper = ParkourVecEnvWrapper
    mjlab_play.ViserPlayViewer = ParkourViserPlayViewer
    mjlab_play.run_play(task_id, tyro.cli(mjlab_play.PlayConfig, config=mjlab.TYRO_FLAGS))


if __name__ == "__main__":
    main()
