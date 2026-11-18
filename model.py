import jax
import jax.numpy as jnp
from flax import nnx
import optax
import jax.numpy as jnp


class Encoder(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(4, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        self.linear = nnx.Linear(64 * 10 * 10, 128, rngs=rngs)

    def __call__(self, obs):
        x = jax.nn.relu(self.conv1(obs))
        x = jax.nn.relu(self.conv2(x))
        x = x.reshape(*x.shape[:-3], -1)
        x = self.linear(x)

        return x


class Decoder(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(128, 64 * 10 * 10, rngs=rngs)
        self.conv1 = nnx.Conv(64, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 4, kernel_size=(3, 3), rngs=rngs)

    def __call__(self, obs):
        x = jax.nn.relu(self.linear(obs))
        x = x.reshape(*x.shape[:-1], 10, 10, 64)

        x = jax.nn.relu(self.conv1(x))
        x = self.conv2(x)

        return x


def reconstruction_loss(logits, observations):
    loss = optax.sigmoid_binary_cross_entropy(
        logits,
        observations.astype(jnp.float32)
    )
    
    return jnp.mean(loss)