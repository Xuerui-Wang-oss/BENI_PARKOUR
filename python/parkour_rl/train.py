# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""Train the high-level parkour policy (task Beni-Parkour or Beni-Parkour-Env1) with mjlab's PPO runner.

    python -m parkour_rl.train
    python -m parkour_rl.train --env.scene.num-envs 512 --agent.max-iterations 2
    python -m parkour_rl.train env1 --agent.max-iterations 2
    python -m parkour_rl.train env2 --agent.max-iterations 2

The options are those of mjlab's ``TrainConfig``; ``--help`` lists them. An optional first argument ``env1`` or
``env2`` trains Beni-Parkour-Env1 (env1.py) or Beni-Parkour-Env2 (env2.py) instead of Beni-Parkour. By default it
runs 1024 envs for 3000 iterations and writes TensorBoard logs and checkpoints to
``logs/rsl_rl/beni_parkour/<start time>/`` under the working directory; env1 and env2 run 1280 envs and write to
``logs/rsl_rl/beni_parkour_env1/<start time>/`` and ``logs/rsl_rl/beni_parkour_env2/<start time>/``.

env2 starts from the actor of the env1 run named in env2.py (``--agent.init-checkpoint <file>`` names another one,
``--agent.init-checkpoint None`` trains from scratch) and trains only the critic for its first 50 iterations.

Three weight files must be present under ``python/parkour_rl``; the repository holds them:

    low_level_policy/model_crawl.pt        frozen crawl policy
    low_level_policy/model_jump.pt         frozen jump policy
    depth_model/defm_regnet_y_400mf.pth    frozen depth encoder

On a new machine, first run with ``--agent.max-iterations 2`` and read the "Collection time" printed for each
iteration. Every high-level step simulates several control steps, renders a depth image and runs the depth encoder
for each env, so this time shows how long 3000 iterations will take and whether the number of envs has to change.
"""

from dataclasses import replace

import mjlab
import torch
import tyro
from mjlab.scripts import train as mjlab_train
from mjlab.utils.gpu import select_gpus

from . import task_from_argv  # Importing the package registers the tasks.
from .hierarchy import ParkourVecEnvWrapper
from .video import ParkourVideoRecorder


def main() -> None:
    """Read mjlab's training options from the command line and train the task on one GPU or on the CPU."""
    task_id = task_from_argv()

    # mjlab's run_train looks these names up when it wraps the env, so the assignments make it use the parkour
    # wrapper and, with --video, the recorder that films one robot per lane column. Nothing else is replaced: the
    # runner and the training loop are mjlab's own.
    mjlab_train.RslRlVecEnvWrapper = ParkourVecEnvWrapper
    mjlab_train.VideoRecorder = ParkourVideoRecorder

    # mjlab's default, gpu_ids=[0], raises on a machine without CUDA.
    gpu_ids = [0] if torch.cuda.is_available() else None
    default = replace(mjlab_train.TrainConfig.from_task(task_id), gpu_ids=gpu_ids)
    args = tyro.cli(mjlab_train.TrainConfig, default=default, config=mjlab.TYRO_FLAGS)

    # mjlab trains on several GPUs in new worker processes. They import mjlab afresh, so they would not see the
    # assignment above and would wrap the env with mjlab's own wrapper.
    _, num_gpus = select_gpus(args.gpu_ids)
    if num_gpus > 1:
        raise ValueError(f"{task_id} trains on one GPU or on the CPU; --gpu-ids selects {num_gpus} GPUs")

    mjlab_train.launch_training(task_id, args)


if __name__ == "__main__":
    main()
