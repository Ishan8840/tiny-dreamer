import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from tiny_dreamer.model import DynamicsModel
from tiny_dreamer.world_model_learning import noise_history


class DiffusionForcingTests(unittest.TestCase):
    def test_independent_frame_levels_and_interpolation(self):
        history = jnp.ones((64, 5, 8))
        key = jax.random.key(23)
        mixed, signal = noise_history(history, key)

        self.assertEqual(signal.shape, history.shape[:-1])
        self.assertTrue(np.any(np.ptp(signal, axis=1) > 0))
        self.assertTrue(np.any(signal == 0))
        self.assertTrue(np.any(signal == 1))
        _, noise_key = jax.random.split(key)
        noise = jax.random.normal(noise_key, history.shape)
        np.testing.assert_allclose(
            mixed, signal[..., None] * history + (1 - signal[..., None]) * noise
        )

    def test_clean_history_uses_the_existing_inference_path(self):
        net = DynamicsModel(nnx.Rngs(0), 3, latent_dim=8, d_model=16, n_heads=2)
        history = jax.random.normal(jax.random.key(0), (2, 5, 8))
        args = (
            history,
            jnp.zeros((2, 5), dtype=int),
            history[:, -1],
            jnp.full(2, 0.5),
            jnp.full(2, 0.25),
        )

        clean = net(*args)
        np.testing.assert_allclose(clean, net(*args, jnp.ones((2, 5))), atol=1e-6)
        self.assertFalse(np.allclose(clean, net(*args, jnp.zeros((2, 5)))))

    def test_inference_noise_only_perturbs_the_generated_suffix(self):
        from tiny_dreamer.model import predict_next

        class EchoContext:
            inference_history_noise = 1 / 16

            def __call__(self, history, actions, latent, level, step, history_signal):
                return history[..., 0] + history_signal

        net = EchoContext()
        key = jax.random.key(37)
        history = jnp.arange(9, dtype=jnp.float32).reshape(1, 3, 3)
        actions = jnp.zeros((1, 3), dtype=int)
        clean = history[..., 0] + 1

        np.testing.assert_allclose(
            predict_next(net, history, actions, key, 4, 0), clean
        )
        prediction = predict_next(net, history, actions, key, 4, 1)
        np.testing.assert_allclose(prediction[:, :-1], clean[:, :-1])

        noise = jax.random.normal(jax.random.fold_in(key, 0xDF), history.shape)
        expected = (15 / 16) * history[:, -1, 0] + noise[:, -1, 0] / 16 + 15 / 16
        np.testing.assert_allclose(prediction[:, -1], expected)
