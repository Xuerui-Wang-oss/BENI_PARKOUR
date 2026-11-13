# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""PPO configuration of the high-level parkour policy.

Actor and critic are recurrent rsl_rl models: normalised observation -> GRU -> MLP. They read the observation
groups returned by ``hierarchy.ParkourVecEnvWrapper``:

    actor  = hl_actor (24) + depth (440) = 464 inputs -> 3 outputs: vx, base height, jump trigger
    critic = hl_critic (42)              =  42 inputs -> 1 output: the value

``hl_critic`` holds the actor's terms plus privileged ones; ``depth`` is the feature of the frozen depth encoder.
The critic does without the depth feature: its privileged terms give the next obstacle, the number of obstacles
left and the robot's place in the lane exactly.
One runner step is one high-level step of the wrapper (0.05 s, 20 Hz). mjlab's own runner trains with this
configuration: there is no runner subclass.
"""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def parkour_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Runner, model and PPO settings of the task ``Beni-Parkour``."""
    recurrent_model_actor = {
        "class_name": "RNNModel",
        "rnn_type": "gru",
        "rnn_hidden_dim": 256,
        "rnn_num_layers": 1,
        "hidden_dims": (256, 128),
        "activation": "elu",
        "obs_normalization": True,
    }

    recurrent_model_critic = {
        "class_name": "RNNModel",
        "rnn_type": "gru",
        "rnn_hidden_dim": 512,
        "rnn_num_layers": 1,
        "hidden_dims": (512, 256, 128),
        "activation": "elu",
        "obs_normalization": True,        
    }

    return RslRlOnPolicyRunnerCfg(
        # rsl_rl concatenates the groups of each model in the order listed here.
        # actor has obsevations + depth
        # critic has observations
        obs_groups={"actor": ("hl_actor", "depth"), "critic": ("hl_critic",)},
        actor=RslRlModelCfg(
            **recurrent_model_actor,
            # rsl_rl starts all action dims with the same std (init_std is one float, not one per dim). The wrapper
            # therefore scales each dim (VX_SCALE, HEIGHT_SCALE, TRIGGER_THRESHOLD in hierarchy.py) so that a std
            # of 1 is a sensible exploration width for all three.
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        ),

        critic=RslRlModelCfg(**recurrent_model_critic),

        algorithm=RslRlPpoAlgorithmCfg(
            learning_rate=1e-3,
            schedule="adaptive",
            desired_kl=0.01,
            # Horizon 1 / (1 - gamma) = 200 high-level steps = 10 s. mjlab's default of 0.99 would look only 5 s
            # ahead at 20 Hz, and the goal bonus comes at the end of a lane.
            gamma=0.995,
            lam=0.95,
            # 0.005 let the action std grow from 0.72 to 1.69 after iteration 1250 in run 2026-10-14_21-35-41.
            entropy_coef=0.001,
            num_learning_epochs=5,
            num_mini_batches=4,
            clip_param=0.2,
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            max_grad_norm=1.0,
        ),

        num_steps_per_env=24,  # High-level steps per rollout: 1.2 s per env.
        max_iterations=3000,
        # Equal to max_iterations: rsl_rl saves model_0.pt (iteration 0) and the final model at the end of learn() only.
        save_interval=3000,
        seed=42,
        experiment_name="beni_parkour",
        logger="tensorboard",
        upload_model=False,
        clip_actions=None,  # The wrapper clamps each action dim to its own range.
    )
