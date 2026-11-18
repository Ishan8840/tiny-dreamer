import jax
import numpy as np
import optax
from flax import nnx

from collect import collect_episodes
from env import make_env
from model import Decoder, Encoder, reconstruction_loss


BATCH_SIZE = 64
LEARNING_RATE = 3e-4
EPOCHS = 20


def batch_loss(model, observations):
    return reconstruction_loss(model(observations), observations)


@nnx.jit
def train_step(model, optimizer, batch):
    loss, grads = nnx.value_and_grad(batch_loss)(model, batch)
    optimizer.update(model, grads)
    return loss


def train(observations, seed=0):
    if len(observations) == 0:
        raise ValueError("Training requires at least one observation")

    rngs = nnx.Rngs(seed)
    model = nnx.Sequential(Encoder(rngs), Decoder(rngs))
    optimizer = nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt=nnx.Param)
    rng = np.random.default_rng(seed)

    for epoch in range(EPOCHS):
        indices = rng.permutation(len(observations))
        total_loss = 0.0
        for start in range(0, len(indices), BATCH_SIZE):
            batch = observations[indices[start:start + BATCH_SIZE]]
            loss = train_step(model, optimizer, batch)
            total_loss += float(loss) * len(batch)

        print(f"Epoch {epoch + 1}/{EPOCHS}  BCE {total_loss / len(indices):.5f}")
    return model


if __name__ == "__main__":
    env, params = make_env()
    key = jax.random.PRNGKey(0)
    episodes, _ = collect_episodes(env, params, key, num_episodes=5000)
    train(np.concatenate([episode["observations"] for episode in episodes]))
