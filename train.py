import jax
import numpy as np
import optax
from flax import nnx
import jax.numpy as jnp

from collect import collect_episodes
from env import make_env
from model import Decoder, Encoder, reconstruction_loss


BATCH_SIZE = 64
LEARNING_RATE = 3e-4
EPOCHS = 20
LATENT_DIM = 128


def reconstruction_loss(logits, observations):
    loss = optax.sigmoid_binary_cross_entropy(
        logits,
        observations.astype(jnp.float32)
    )

    return jnp.mean(loss)


def batch_loss(model, observations):
    return reconstruction_loss(model(observations), observations)


def dynamics_loss(
    history_latents,
    history_actions,
    noisy_latent,
    noise_level,
    step_size,
    predicted_latent,
    target_latent,
    model,
    d_min,
):
    half_step = step_size / 2

    b_1 = (
        model(
            history_latents,
            history_actions,
            noisy_latent,
            noise_level,
            half_step,
        ) - noisy_latent
    ) / (1 - noise_level)[:, None]

    z_prime = noisy_latent + half_step[:, None] * b_1
    t_prime = noise_level + half_step

    b_2 = (
        model(
            history_latents,
            history_actions,
            z_prime,
            t_prime,
            half_step,
        ) - z_prime
    ) / (1 - t_prime)[:, None]

    target_velocity = jax.lax.stop_gradient((b_1 + b_2) / 2)

    predicted_velocity = (
        predicted_latent - noisy_latent
    ) / (1 - noise_level)[:, None]

    reconstruction_loss = jnp.mean(
        (predicted_latent - target_latent) ** 2,
        axis=-1,
    )

    shortcut_loss = ((1 - noise_level) ** 2) * jnp.mean(
        (predicted_velocity - target_velocity) ** 2,
        axis=-1,
    )

    loss = jnp.where(
        jnp.isclose(step_size, d_min),
        reconstruction_loss,
        shortcut_loss,
    )

    ramp_weight = 0.9 * noise_level + 0.1

    return jnp.mean(loss * ramp_weight)


validation_step = nnx.jit(batch_loss)


def sample_noisy_latents(target_latent, rng):
    B = target_latent.shape[0]

    noise_key, t_key = jax.random.split(rng)

    noise = jax.random.normal(
        noise_key, target_latent.shape
    )

    t = jax.random.uniform(
        t_key,
        shape=(B,),
        minval=0.0,
        maxval=1.0,
    )

    noisy_latent = (
        t[:, None] * target_latent
        + (1 - t[:, None]) * noise
    )

    return noisy_latent, t


def split_episodes(episodes, seed=0):
    if len(episodes) < 2:
        raise ValueError("A train/validation split requires at least two episodes")
    order = np.random.default_rng(seed).permutation(len(episodes))
    val_count = max(1, round(0.1 * len(episodes)))
    val_obs = np.concatenate([episodes[i]["observations"] for i in order[:val_count]])
    train_obs = np.concatenate([episodes[i]["observations"] for i in order[val_count:]])
    return train_obs, val_obs


@nnx.jit
def train_step(model, optimizer, batch):
    loss, grads = nnx.value_and_grad(batch_loss)(model, batch)
    optimizer.update(model, grads)
    return loss


def train(observations, seed=0, *, validation_observations=None):
    if len(observations) == 0:
        raise ValueError("Training requires at least one observation")
    if validation_observations is not None and len(validation_observations) == 0:
        raise ValueError("Validation requires at least one observation")

    rngs = nnx.Rngs(seed)
    channels = observations.shape[-1]
    model = nnx.Sequential(Encoder(rngs, channels, LATENT_DIM), Decoder(rngs, channels, LATENT_DIM))
    optimizer = nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt=nnx.Param)
    rng = np.random.default_rng(seed)

    for epoch in range(EPOCHS):
        indices = rng.permutation(len(observations))
        total_loss = 0.0
        for start in range(0, len(indices), BATCH_SIZE):
            batch = observations[indices[start:start + BATCH_SIZE]]
            loss = train_step(model, optimizer, batch)
            total_loss += float(loss) * len(batch)

        metrics = f"Epoch {epoch + 1}/{EPOCHS}  train BCE {total_loss / len(indices):.5f}"
        if validation_observations is not None:
            val_loss = 0.0
            for start in range(0, len(validation_observations), BATCH_SIZE):
                batch = validation_observations[start:start + BATCH_SIZE]
                val_loss += float(validation_step(model, batch)) * len(batch)
            metrics += f"  val BCE {val_loss / len(validation_observations):.5f}"
        print(metrics)
    return model


if __name__ == "__main__":
    env, params = make_env()
    key = jax.random.PRNGKey(0)
    episodes, _ = collect_episodes(env, params, key, num_episodes=10000)
    train_obs, val_obs = split_episodes(episodes)
    train(train_obs, validation_observations=val_obs)
