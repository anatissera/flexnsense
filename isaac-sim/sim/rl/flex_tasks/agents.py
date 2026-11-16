"""PPO settings (rsl_rl) for the Flex'N Sense tasks."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class PullPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 600
    save_interval = 100
    experiment_name = "flex_pull"
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 64], activation="elu", obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.4),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[128, 128, 64], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.003,
        num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive",
        gamma=0.98, lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
    )


@configclass
class MovePPORunnerCfg(PullPPORunnerCfg):
    max_iterations = 1500
    save_interval = 250
    experiment_name = "flex_move"
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64], activation="elu", obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[256, 128, 64], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.004,
        num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive",
        gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
    )


@configclass
class Pull10PPORunnerCfg(PullPPORunnerCfg):
    max_iterations = 600
    experiment_name = "flex_pull10"


@configclass
class PullBenchPPORunnerCfg(PullPPORunnerCfg):
    max_iterations = 600
    experiment_name = "flex_pull_bench"


@configclass
class DuoPPORunnerCfg(MovePPORunnerCfg):
    max_iterations = 1500
    save_interval = 250
    experiment_name = "flex_duo"
