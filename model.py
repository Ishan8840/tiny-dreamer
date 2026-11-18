import jax
import jax.numpy as jnp
from flax import nnx
import optax
import jax.numpy as jnp


class Encoder(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, channels, latent_dim):
        self.conv1 = nnx.Conv(channels, 16, kernel_size=(3, 3),padding="VALID", rngs=rngs)
        self.conv2 = nnx.Conv(16, 32, kernel_size=(3, 3),padding="VALID", rngs=rngs)
        self.project = nnx.Conv(32, 8, kernel_size=(1, 1), rngs=rngs)
        self.linear = nnx.Linear(8 * 6 * 6, latent_dim, rngs=rngs)
        self.norm = nnx.LayerNorm(latent_dim, rngs=rngs)

    def __call__(self, obs):
        x = jax.nn.elu(self.conv1(obs))
        x = jax.nn.elu(self.conv2(x))
        x = self.project(x)
        x = x.reshape(*x.shape[:-3], -1)
        x = self.linear(x)

        return self.norm(x)


class Decoder(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, channels, latent_dim):
        self.linear = nnx.Linear(latent_dim, 8 * 6 * 6, rngs=rngs)
        self.project = nnx.Conv(8, 32, kernel_size=(1, 1), rngs=rngs)
        self.conv1 = nnx.ConvTranspose(32, 16, kernel_size=(3, 3),padding="VALID", rngs=rngs)
        self.conv2 = nnx.ConvTranspose(16, channels, kernel_size=(3, 3),padding="VALID", rngs=rngs)

    def __call__(self, obs):
        x = jax.nn.elu(self.linear(obs))
        x = x.reshape(*x.shape[:-1], 6, 6, 8)
        x = self.project(x)

        x = jax.nn.elu(self.conv1(x))
        x = self.conv2(x)

        return x


def reconstruction_loss(logits, observations):
    loss = optax.sigmoid_binary_cross_entropy(
        logits,
        observations.astype(jnp.float32)
    )

    return jnp.mean(loss)
