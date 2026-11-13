# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""Train the crawl task with mjlab, resuming from a jump checkpoint by default.

Seed the run directory once, then train:

    mkdir -p logs/rsl_rl/beni_crawl_flat/0_from_best
    cp assets/models/policy_best.pt logs/rsl_rl/beni_crawl_flat/0_from_best/model_10500.pt
    python -m beni_rl_crawl.train Beni-Crawl-Flat --env.scene.num-envs 4096
"""

import sys
from dataclasses import dataclass, replace
from typing import Literal

import mjlab
import tyro
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.scripts.train import TrainConfig as MjlabTrainConfig
from mjlab.scripts.train import launch_training
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.gpu import select_gpus
from mjlab.managers import EventTermCfg
from mjlab.managers.recorder_manager import RecorderTermCfg
from .recovery_sampling import AdaptiveRecoverySampler
from .recovery_curriculum import (DEFAULT_TRAINING_STATE_BANK, reset_curriculum_states,
                                  curriculum_frames, training_state_bank)
from .recovery import sampling_metrics
from . import TASK_ID
from .distributed import resolve_gpu_ids


@dataclass(frozen=True)
class TrainConfig(MjlabTrainConfig):
    gpu_ids: list[int] | Literal["all", "auto"] | None = "auto"
    """Auto selects up to eight visible GPUs, otherwise CPU. num_envs is per GPU."""

    @classmethod
    def from_task(cls, task_id):
        """Assemble BENI defaults before applying command-line overrides."""
        args = cls(env=load_env_cfg(task_id), agent=load_rl_cfg(task_id))
        configure_recovery_curriculum(args)
        args.env.events["recovery_sampling_metrics"] = EventTermCfg(func=sampling_metrics, mode="step")
        return args


def validate_training(task_id, args):
    if args.agent.max_iterations < 1:
        raise ValueError('Training requires a positive number of PPO updates')
    if 'reset_curriculum' in args.env.events:
        training_state_bank(args.env.events['reset_curriculum'].params['bank_path'])


def configure_recovery_curriculum(args):
    """Configure recovery resets with the default training bank."""
    args.env.curriculum = {}
    args.env.events["reset_base"].params["fallen_probability"] = 0.
    args.env.events["reset_curriculum"] = EventTermCfg(
        func=reset_curriculum_states, mode="reset",
        params={"bank_path": str(DEFAULT_TRAINING_STATE_BANK)})
    args.env.events["curriculum_frames"] = EventTermCfg(func=curriculum_frames, mode="step")
    args.env.recorders['recovery_sampling'] = RecorderTermCfg(func=AdaptiveRecoverySampler)


def main() -> None:
    maybe_print_top_level_help("python -m beni_rl_crawl.train")
    task_id, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices((TASK_ID,)),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    args = tyro.cli(
        TrainConfig,
        args=remaining_args,
        default=TrainConfig.from_task(task_id),
        prog=sys.argv[0] + f" {task_id}",
        config=mjlab.TYRO_FLAGS,
    )
    validate_training(task_id, args)
    # The reset schedule uses actual PPO updates, not an independent clock.
    args.env.events['reset_curriculum'].params['steps_per_iteration'] = args.agent.num_steps_per_env
    args = replace(args, gpu_ids=resolve_gpu_ids(args.gpu_ids))
    devices, count = select_gpus(args.gpu_ids)
    print(f"[Training devices] {devices or 'CPU'}; environments per worker: {args.env.scene.num_envs}; "
          f"total environments: {args.env.scene.num_envs * max(count, 1)}")
    if count == 0:
        print(
            "[WARNING] Training on CPU is slow. Use a small --env.scene.num-envs value (e.g. 4).",
            file=sys.stderr,
        )
    launch_training(task_id=task_id, args=args)


if __name__ == "__main__":
    main()
