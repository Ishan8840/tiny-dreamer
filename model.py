import jax
import jax.numpy as jnp
from flax import nnx
import optax
import jax.numpy as jnp


def apply_rope(x):
    B, T, H, D = x.shape

    positions = jnp.arange(T)
    frequencies = 1.0 / (
        10000 ** (jnp.arange(0, D, 2) / D)
    )

    angles = positions[:, None] * frequencies[None, :]

    cos = jnp.cos(angles)[None, :, None, :]
    sin = jnp.sin(angles)[None, :, None, :]

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    return jnp.stack(
        [rotated_even, rotated_odd], axis=-1
    ).reshape(x.shape)


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


class RoPEAttention(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, d_model=128, n_heads=4):
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.k_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.v_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs)

    def __call__(self, x):
        B, T, _ = x.shape

        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, T, self.n_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, T, self.n_heads, self.head_dim)

        q = apply_rope(q)
        k = apply_rope(k)

        mask = jnp.tril(jnp.ones((T, T), dtype=bool))

        out = jax.nn.dot_product_attention(q, k, v, mask=mask)

        out = out.reshape(B, T, self.n_heads * self.head_dim)


class TransformerBlock(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, d_model=128, n_heads=4):
        self.norm1 = nnx.RMSNorm(d_model, rngs=rngs)

        self.attn = RoPEAttention(
            d_model=d_model,
            n_heads=n_heads,
        )

        self.norm2 = nnx.RMSNorm(d_model, rngs=rngs)

        self.fc1 = nnx.Linear(d_model, 4 * d_model, rngs=rngs)
        self.fc2 = nnx.Linear(4 * d_model, d_model, rngs=rngs)

    def __call__(self, x, mask):
        x = x + self.attn(self.norm1(x), mask=mask)

        h = self.fc1(self.norm2(x))
        h = jax.nn.silu(h)
        h = self.fc2(h)

        return x + h


class DynamicsModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, action_dim, latent_dim=64, d_model=128, n_heads=4):
        self.latent_emb = nnx.Linear(latent_dim, d_model, rngs=rngs)
        self.action_emb = nnx.Embed(num_embeddings=action_dim, features=d_model, rngs=rngs)
        self.noise_emb = nnx.Linear(1, d_model, rngs=rngs)
        self.step_emb = nnx.Linear(1, d_model, rngs=rngs)

        self.blocks = [TransformerBlock(rngs, d_model, n_heads) for _ in range(2)]

        self.out_proj = nnx.Linear(d_model, latent_dim, rngs=rngs)

    def __call__(self, history_latents, history_actions, noisy_latent, noise_level, step_size):
        x = self.latent_emb(history_latents) + self.action_emb(history_actions)

        tau_emb = self.noise_emb(noise_level[:, None])
        step_emb = self.step_emb(step_size[:, None])

        future_token = self.latent_emb(noisy_latent) + tau_emb + step_emb
        future_token = future_token[:, None, :]

        x = jnp.concatenate([x, future_token], axis=1)

        for block in self.blocks:
            x = block(x)

        future_hidden = x[:, -1, :]
        predicted_clean = self.out_proj(future_hidden)

        return predicted_clean