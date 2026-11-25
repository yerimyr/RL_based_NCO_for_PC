import os

import pytest

from rl4co.envs import PartConsolidationEnv, TSPEnv
from rl4co.models.rl import A2C, PPO, REINFORCE
from rl4co.models.nn.graph.pc_encoder import PCEdgeAwareEncoder
from rl4co.models.zoo import AttentionModelPolicy
from rl4co.utils import RL4COTrainer

# Get env variable MAC_OS_GITHUB_RUNNER
if "MAC_OS_GITHUB_RUNNER" in os.environ:
    accelerator = "cpu"
else:
    accelerator = "auto"


# Test out simple training loop and test with multiple baselines
@pytest.mark.parametrize("baseline", ["rollout", "exponential", "mean", "no", "critic"])
def test_reinforce(baseline):
    env = TSPEnv(generator_params=dict(num_loc=20))
    policy = AttentionModelPolicy(env_name=env.name)
    model = REINFORCE(
        env,
        policy,
        baseline=baseline,
        train_data_size=10,
        val_data_size=10,
        test_data_size=10,
    )
    trainer = RL4COTrainer(max_epochs=1, devices=1, accelerator=accelerator)
    trainer.fit(model)
    trainer.test(model)


def test_a2c():
    env = TSPEnv(generator_params=dict(num_loc=20))
    policy = AttentionModelPolicy(env_name=env.name)
    model = A2C(env, policy, train_data_size=10, val_data_size=10, test_data_size=10)
    trainer = RL4COTrainer(max_epochs=1, devices=1, accelerator=accelerator)
    trainer.fit(model)
    trainer.test(model)


def test_ppo():
    env = TSPEnv(generator_params=dict(num_loc=20))
    policy = AttentionModelPolicy(env_name=env.name)
    model = PPO(env, policy, train_data_size=10, val_data_size=10, test_data_size=10)
    trainer = RL4COTrainer(max_epochs=1, gradient_clip_val=None, devices=1, accelerator=accelerator)
    trainer.fit(model)
    trainer.test(model)


def test_pc_reinforce():
    env = PartConsolidationEnv(generator_params=dict(num_parts=20))
    policy = AttentionModelPolicy(env_name=env.name)
    model = REINFORCE(
        env,
        policy,
        baseline="no",
        train_data_size=10,
        val_data_size=10,
        test_data_size=10,
        batch_size=2,
    )
    trainer = RL4COTrainer(max_epochs=1, devices=1, accelerator="cpu")
    trainer.fit(model)
    trainer.test(model)


def test_pc_edge_reinforce():
    env = PartConsolidationEnv(generator_params=dict(num_parts=20))
    policy = AttentionModelPolicy(
        env_name=env.name,
        embed_dim=128,
        encoder=PCEdgeAwareEncoder(embed_dim=128, num_layers=1, env_name=env.name),
    )
    model = REINFORCE(
        env,
        policy,
        baseline="no",
        train_data_size=10,
        val_data_size=10,
        test_data_size=10,
        batch_size=2,
    )
    trainer = RL4COTrainer(max_epochs=1, devices=1, accelerator="cpu")
    trainer.fit(model)
    trainer.test(model)


def test_pc_edge_ppo():
    env = PartConsolidationEnv(generator_params=dict(num_parts=20))
    policy = AttentionModelPolicy(
        env_name=env.name,
        embed_dim=128,
        encoder=PCEdgeAwareEncoder(embed_dim=128, num_layers=1, env_name=env.name),
    )
    model = PPO(
        env,
        policy,
        train_data_size=10,
        val_data_size=10,
        test_data_size=10,
        batch_size=2,
        mini_batch_size=2,
        ppo_epochs=1,
    )
    trainer = RL4COTrainer(max_epochs=1, gradient_clip_val=None, devices=1, accelerator="cpu")
    trainer.fit(model)
    trainer.test(model)
