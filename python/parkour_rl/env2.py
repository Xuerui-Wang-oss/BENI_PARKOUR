# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""env2 of the high-level parkour task: lanes of 4-6 mixed obstacles, no curriculum, warm start from env1.

The env is env1 (env1.py) with these changes:
    lanes        5 rows x 16 columns. A row is a layout of 16 lanes: rows 0-3 are the training layouts, row 4 is the
                 test layout, which no training env is ever put on. Every lane has ENV2_COUNT obstacles of the five
                 env1 kinds, each kind equally likely, with its parameter uniform in the range env1 trained
                 (ENV2_VALUES) and ENV2_GAP metres between obstacles. No lane is flat.
    layouts      env i stays on lane column i % 16. Whenever its episode ends, it moves to one of the training rows
                 at random (LayoutSampler), so at any time about a quarter of the envs is on each layout and every
                 PPO update holds all 64 training lanes.
    curriculum   none: the sizes are random from the first episode on.
    success      the robot may stop up to GOAL_OVERSHOOT past the goal. Of the episodes in which the env1 policy
                 passed every obstacle of a multi-obstacle lane, 28 of 66 stopped 0.33-0.85 m past the goal, and vx
                 cannot be negative (model_2999, 2026-11-10).

Training (Env2Runner) starts from the env1 actor and a new critic. The critic of env1 is not reused: its input
normaliser averages over all 92 million samples of env1, so it hardly moves any more, and in env2 the number of
obstacles left (up to 6, env1: 0 or 1) and the distance to the goal (up to 30 m, env1 mean 2.8 m) lie 11-12 standard
deviations outside what it saw. For the first critic_warm_up_iterations PPO iterations only the critic learns; then
the actor is released at fine_tune_learning_rate.

The play form runs one robot on every lane: env i sits on column i // 5 and row i % 5 (80 envs).
"""

from dataclasses import dataclass, fields

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg
from mjlab.terrains import TerrainGeneratorCfg

from .curriculum import _blend
from .env import EPISODE_LENGTH_S, GOAL_OVERSHOOT
from .env1 import ENV1_KINDS, ENV1_ROWS, make_env1_cfg
from .parkour_terrain import LANE, ParkourLaneCfg
from .rl_cfg import parkour_ppo_runner_cfg

ENV2_TRAIN_LAYOUTS = 4  # Rows 0-3.
ENV2_TEST_LAYOUTS = 1  # Row 4.
ENV2_LANES = 16  # Lane columns: the lanes of one layout.
ENV2_COUNT = (4, 6)  # Obstacles of a lane.
# End of an obstacle to the start of the next (m). With gaps of 1.5-2.2 m the env1 policy kept failing the second of
# two jumps: after a landing the next jump is accepted only about 1 s later.
ENV2_GAP = (2., 4.)
# Parameter range of each kind: the smallest and the largest value of its env1 rows.
ENV2_VALUES = {kind: (min(value for row in rows for value in row), max(value for row in rows for value in row))
               for kind, rows in ENV1_ROWS.items()}
# The actor that training starts from: the final checkpoint of the env1 run, relative to the working directory.
ENV1_CHECKPOINT = "logs/rsl_rl/beni_parkour_env1/2026-11-05_22-54-04/model_2999.pt"


def make_env2_terrain_cfg(seed: int = 0) -> TerrainGeneratorCfg:
    """The env2 lanes: one row per layout, training layouts first, and ENV2_LANES columns of random lanes."""
    rows = ENV2_TRAIN_LAYOUTS + ENV2_TEST_LAYOUTS
    return TerrainGeneratorCfg(size=LANE, num_rows=rows, curriculum=True, seed=seed, sub_terrains={
        f"lane_{column}": ParkourLaneCfg(kind="mixed", num_rows=rows, count=ENV2_COUNT, values=ENV2_VALUES,
                                         gap=ENV2_GAP) for column in range(ENV2_LANES)})


class LayoutSampler:
    """Curriculum term of env2: an env whose episode has ended moves to a random training layout.

    mjlab runs the curriculum terms first when it resets envs, before the reset events and the command terms, so the
    spawn and the goal of the new episode are drawn for the new lane. The term also keeps running rates of how the
    episodes end, blended as in curriculum.py, and returns them for the log.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv) -> None:
        del cfg
        self._terrain = env.scene.terrain
        self._kinds = env.command_manager.get_term("goal").kinds
        # Running fractions: episodes that reached the goal, over all layouts (1,) and per layout (layouts,); obstacles
        # of the lane that the robot passed (1,); obstacles of each kind that it passed, of those it got to (kinds,).
        self._success = torch.zeros(1, device=env.device)
        self._layout_success = torch.zeros(ENV2_TRAIN_LAYOUTS, device=env.device)
        self._passed = torch.zeros(1, device=env.device)
        self._kind_passed = torch.zeros(len(self._kinds), device=env.device)

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> dict[str, float]:
        """Record the episodes that just ended in the envs ``env_ids`` (n,; None: all envs) and move those envs.

        Returns the values to log: ``success``, ``success_layout<r>``, ``obstacles_passed`` and, per kind,
        ``passed_<kind>`` (see ``__init__``).
        """
        terrain = self._terrain
        env_ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
        # mjlab also resets every env once before the first step. No episode has ended then, so nothing is recorded.
        if env.common_step_counter > 0:
            # Read before the envs move and before the reset: the robot is still where the episode ended.
            success = env.termination_manager.get_term("goal_reached")[env_ids]
            kind, passed = (table[env_ids] for table in env.command_manager.get_term("goal").obstacle_outcomes())
            count = passed.sum(1, keepdim=True)  # (n, 1) obstacles passed; they are the first entries of the lane.
            # The obstacles that the robot got to: those it passed and the one after them, where the episode ended.
            reached = (kind >= 0) & (torch.arange(kind.shape[1], device=kind.device) <= count)
            zeros = torch.zeros_like(env_ids)
            _blend(self._success, zeros, success)
            _blend(self._layout_success, terrain.terrain_levels[env_ids], success)
            _blend(self._passed, zeros, count.squeeze(1) / (kind >= 0).sum(1))
            _blend(self._kind_passed, kind[reached].long(), passed[reached])

        rows = torch.randint(ENV2_TRAIN_LAYOUTS, (len(env_ids),), device=env.device)
        terrain.terrain_levels[env_ids] = rows
        terrain.env_origins[env_ids] = terrain.terrain_origins[rows, terrain.terrain_types[env_ids]]

        log = {"success": self._success.item(), "obstacles_passed": self._passed.item()}
        log.update({f"success_layout{row}": rate for row, rate in enumerate(self._layout_success.tolist())})
        log.update({f"passed_{kind}": rate for kind, rate in zip(self._kinds, self._kind_passed.tolist())})
        return log


def make_env2_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the env configuration of env2: ``make_env1_cfg(play)`` with the changes of the module docstring.

    ``play=False`` is the training env: 1280 envs, env i on lane column i % 16, on a random training layout.
    ``play=True`` has no layout sampler; the registered play form puts one env on every lane, the test layout included.
    """
    cfg = make_env1_cfg(play)
    terrain = cfg.scene.terrain
    terrain.terrain_generator = make_env2_terrain_cfg()
    # Training starts every env on one of the training rows; the sampler keeps it on them.
    terrain.max_init_terrain_level = None if play else ENV2_TRAIN_LAYOUTS - 1
    cfg.episode_length_s = EPISODE_LENGTH_S  # The lanes have the length of the Beni-Parkour lanes again.

    # The lanes mix kinds, so the goal term cannot read the kinds from the columns. The order of env1 keeps the
    # critic's observation as it is in env1.
    cfg.commands["goal"].kinds = ENV1_KINDS
    cfg.observations["hl_critic"].terms["goal_distance_full"].clip = (-GOAL_OVERSHOOT, LANE[0])
    cfg.terminations["goal_reached"].params["tolerance_past"] = GOAL_OVERSHOOT
    cfg.curriculum = {} if play else {"layout": CurriculumTermCfg(func=LayoutSampler)}
    return cfg


@dataclass
class Env2RunnerCfg(RslRlOnPolicyRunnerCfg):
    """The PPO configuration of Beni-Parkour with the warm start of ``Env2Runner``."""

    # Checkpoint whose actor a new run starts from; None trains from scratch, without the warm-up.
    init_checkpoint: str | None = ENV1_CHECKPOINT
    # PPO iterations at the start of a warm-started run in which only the critic learns, at the configured learning
    # rate, without the adaptive schedule and without the clipping of the value loss.
    critic_warm_up_iterations: int = 50
    # Learning rate from the iteration that releases the actor; the adaptive schedule takes over from there. The env1
    # run started at 1e-3 and its schedule kept the rate at about 1.3e-4 from iteration 50 on.
    fine_tune_learning_rate: float = 1e-4


def env2_ppo_runner_cfg() -> Env2RunnerCfg:
    """The PPO configuration of Beni-Parkour (rl_cfg.py) with the warm start, a checkpoint every 500 iterations and
    its own log directory, beni_parkour_env2."""
    base = parkour_ppo_runner_cfg()
    cfg = Env2RunnerCfg(**{field.name: getattr(base, field.name) for field in fields(base)})
    cfg.experiment_name = "beni_parkour_env2"
    cfg.save_interval = 500
    return cfg


class Env2Runner(MjlabOnPolicyRunner):
    """mjlab's runner with the warm start of env2 (module docstring): the actor of ``init_checkpoint``, a new critic.

    A run that resumes an env2 checkpoint (``--agent.resume``) and a runner that only plays are left as they are.
    """

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        checkpoint = self.cfg["init_checkpoint"]
        if checkpoint is not None and self.current_learning_iteration == 0:
            print(f"[INFO]: Starting from the actor of {checkpoint}; the critic and the optimiser are new.")
            # mjlab's load also sets the env's step counter to that of the checkpoint, which belongs to the env1 run.
            steps = self.env.unwrapped.common_step_counter
            self.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=self.device)
            self.env.unwrapped.common_step_counter = steps
            self._warm_up_critic(self.cfg["critic_warm_up_iterations"], self.cfg["fine_tune_learning_rate"])
        super().learn(num_learning_iterations, init_at_random_ep_len)

    def _warm_up_critic(self, iterations: int, learning_rate: float) -> None:
        """Freeze the actor for the first ``iterations`` PPO updates, then release it at ``learning_rate``."""
        algorithm, update, schedule = self.alg, self.alg.update, self.alg.schedule
        clipped = algorithm.use_clipped_value_loss
        updates = 0

        def update_with_warm_up() -> dict[str, float]:
            nonlocal updates
            frozen = updates < iterations
            algorithm.actor.requires_grad_(not frozen)
            # Fixed while the actor is frozen: its KL divergence is then 0 up to round-off, on which the adaptive
            # schedule would raise the rate by half on every mini-batch.
            algorithm.schedule = "fixed" if frozen else schedule
            # Not clipped while only the critic learns: the clipped value loss has no gradient once a value is
            # clip_param (0.2) away from its value in the rollout, so a value moves by about 0.2 per iteration, and
            # the new critic starts near 0 where the returns of env2 are about -5 to 20.
            algorithm.use_clipped_value_loss = clipped and not frozen
            if updates == iterations:
                algorithm.learning_rate = learning_rate
                for group in algorithm.optimizer.param_groups:
                    group["lr"] = learning_rate
            updates += 1
            return update()

        algorithm.update = update_with_warm_up
