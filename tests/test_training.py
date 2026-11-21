import unittest

import jax.numpy as jnp
import numpy as np

from nanodreamer.actor_critic import lambda_returns
from nanodreamer.config import parse_args
from nanodreamer.data import make_replay


class TrainingTests(unittest.TestCase):
    def test_rollout_stops_at_terminal(self):
        values = lambda_returns(
            jnp.array([1.0, 2.0]), jnp.array([0.0, 1.0]), jnp.array([8.0, 3.0]), 1.0
        )
        np.testing.assert_allclose(values, [1.0, 5.0])

    def test_full_episode_window_includes_final_observation(self):
        episode = {
            "observations": np.arange(4),
            "actions": np.arange(3),
            "dones": np.array([False, False, True]),
            "rewards": np.array([0, 0, 1]),
        }
        replay = make_replay([episode], 3, 0)
        window = replay.sample_batch(1, 3)
        np.testing.assert_array_equal(window["observations"], [[0, 1, 2, 3]])
        np.testing.assert_array_equal(window["rewards"], [[0, 0, 1]])

    def test_online_uses_ppo(self):
        model, run = parse_args(
            [
                "--train",
                "online",
                "--model-checkpoint",
                "world.msgpack",
                "--output",
                "new-run",
            ]
        )
        self.assertEqual(model.latent_dim, 64)
        self.assertEqual(run.policy_method, "ppo")
        self.assertEqual(run.actor_learning_rate, 1e-4)

    def test_reward_sampling_recovers_natural_event_rate(self):
        from nanodreamer.data import RewardSampler

        episode = {
            "observations": np.arange(12),
            "actions": np.arange(11),
            "rewards": np.array([0] * 10 + [1]),
        }
        sampler = RewardSampler([episode], 2)
        _, _, points, weights = sampler.sample(np.random.default_rng(0), 8, 0.5)
        self.assertEqual(points.sum(), 4)
        self.assertAlmostEqual(float(np.mean(points * weights)), 0.1)
        self.assertAlmostEqual(float(weights.mean()), 1.0)
