import jax
import jax.numpy as jnp
from flax import nnx, serialization


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


class RoPEAttention(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, d_model=128, n_heads=4, qknorm=True, softcap=0.0):
        if n_heads < 1 or d_model < 1 or d_model % (2 * n_heads):
            raise ValueError("RoPE requires an even head dimension and equally sized heads")
        
        self.n_heads = n_heads
        self.softcap = softcap
        self.head_dim = d_model // n_heads
        self.qk_scale = nnx.Param(jnp.array(self.head_dim ** 0.5, dtype=jnp.float32)) if qknorm else None

        self.q_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.k_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.v_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs)

    def __call__(self, x, mask=None):
        B, T, _ = x.shape

        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, T, self.n_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, T, self.n_heads, self.head_dim)

        q = apply_rope(q)
        k = apply_rope(k)

        scale = None
        if self.qk_scale is not None:
            q = q * jax.lax.rsqrt(
                jnp.maximum(jnp.sum(q * q, axis=-1, keepdims=True), 1e-12)
            )
            k = k * jax.lax.rsqrt(
                jnp.maximum(jnp.sum(k * k, axis=-1, keepdims=True), 1e-12)
            )
            q = q * self.qk_scale[...]
            scale = 1.0

        if mask is None:
            mask = jnp.tril(jnp.ones((T, T), dtype=bool))

        mask = mask[None, None, :, :]

        if self.softcap:
            logits = jnp.einsum("bthd,bshd->bhts", q, k)

            if scale is None:
                logits = logits * self.head_dim ** -0.5

            logits = self.softcap * jnp.tanh(logits / self.softcap)

            weights = jax.nn.softmax(
                jnp.where(mask, logits, -jnp.inf),
                axis=-1,
            )
            out = jnp.einsum("bhts,bshd->bthd", weights, v)
        else:
            out = jax.nn.dot_product_attention(
                q, k, v,
                mask=mask,
                scale=scale,
            )

        out = out.reshape(B, T, self.n_heads * self.head_dim)
        return self.out_proj(out)


class TransformerBlock(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, d_model=128, n_heads=4, qknorm=True, swiglu=False, softcap=0.0):
        self.norm1 = nnx.RMSNorm(d_model, rngs=rngs)

        self.attn = RoPEAttention(
            rngs=rngs,
            d_model=d_model,
            n_heads=n_heads,
            qknorm=qknorm,
            softcap=softcap,
        )

        self.norm2 = nnx.RMSNorm(d_model, rngs=rngs)

        self.swiglu = swiglu
        hidden = round(8 * d_model / 3) if swiglu else 4 * d_model
        self.fc1 = nnx.Linear(d_model, 2 * hidden if swiglu else hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, d_model, rngs=rngs)

    def __call__(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask=mask,)

        h = self.fc1(self.norm2(x))

        if self.swiglu:
            gate, value = jnp.split(h, 2, axis=-1)
            h = jax.nn.silu(gate) * value
        else:
            h = jax.nn.silu(h)

        h = self.fc2(h)

        return x + h


# Ten rescue points plus up to two active friendly shots in default Seaquest.
REWARD_CLASSES = 13


def make_reward_head(width, rngs):
    def sparse_bias(key, shape, dtype=jnp.float32):
        return jnp.zeros(shape, dtype).at[0].set(7.)

    return nnx.Linear(width, REWARD_CLASSES, kernel_init=nnx.initializers.zeros_init(),
                      bias_init=sparse_bias, rngs=rngs)


class DynamicsModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, action_dim, latent_dim=128, d_model=128, n_heads=4, qknorm=True,
                 swiglu=False, softcap=0.0):
        self.latent_emb = nnx.Linear(latent_dim, d_model, rngs=rngs)
        self.action_emb = nnx.Embed(num_embeddings=action_dim, features=d_model, rngs=rngs)
        self.shortcut_emb = nnx.Linear(2, d_model, rngs=rngs)
        self.agent_token = nnx.Param(jnp.zeros((1, 1, d_model)))

        self.blocks = nnx.List([TransformerBlock(rngs, d_model, n_heads, qknorm, swiglu, softcap)
                                for _ in range(2)])

        self.out_proj = nnx.Linear(d_model, latent_dim, rngs=rngs)
        self.reward_head = make_reward_head(d_model, rngs)
        self.reward_target = "categorical_points"
        self.terminal_head = nnx.Linear(d_model, 1, rngs=rngs)

    def __call__(self, history_latents, history_actions, noisy_latent, noise_level, step_size):
        history_tokens = self.latent_emb(history_latents) + self.action_emb(history_actions)

        shortcut = jnp.stack([noise_level, step_size], axis=-1)
        shortcut_token = self.shortcut_emb(shortcut)[:, None, :]

        future_token = self.latent_emb(noisy_latent)
        future_token = future_token[:, None, :]

        x = jnp.concatenate([history_tokens, shortcut_token, future_token], axis=1)
        for block in self.blocks:
            x = block(x)
        return self.out_proj(x[:, -1])

    def outcome_outputs(self, history_latents, history_actions):
        history = self.latent_emb(history_latents) + self.action_emb(history_actions)
        batch, _, width = history.shape
        agent_token = jnp.broadcast_to(self.agent_token[...], (batch, 1, width))
        x = jnp.concatenate([history, agent_token], axis=1)

        for block in self.blocks:
            x = block(x)

        features = x[:, -1]
        return self.reward_head(features), self.terminal_head(features)[:, 0]

    def expected_reward(self, output):
        if self.reward_target == "event":
            return jax.nn.sigmoid(output[..., 0])
        if self.reward_target == "points":
            return output[..., 0]
        return jax.nn.softmax(output) @ jnp.arange(REWARD_CLASSES, dtype=output.dtype)

    def scoring_logit(self, output):
        if self.reward_target == "event":
            return output[..., 0]
        if self.reward_target == "points":
            # Legacy regression checkpoints have no event distribution.
            raise ValueError("A scalar points checkpoint cannot predict scoring probability")
        return jax.scipy.special.logsumexp(output[..., 1:], axis=-1) - output[..., 0]

    def predict_outcomes(self, history_latents, history_actions):
        """Positive-reward and game-over logits used by the event evaluator."""
        reward, terminal = self.outcome_outputs(history_latents, history_actions)
        return self.scoring_logit(reward), terminal


def make_tokenizer(rngs, channels, latent_dim=128):
    return nnx.Sequential(Encoder(rngs, channels, latent_dim), Decoder(rngs, channels, latent_dim))


class WorldModel(nnx.Module):
    def __init__(self, rngs, channels, action_dim, qknorm=True, swiglu=False, softcap=0.0):
        self.tokenizer = make_tokenizer(rngs, channels)
        self.dynamics = DynamicsModel(rngs, action_dim, latent_dim=128, qknorm=qknorm,
                                      swiglu=swiglu, softcap=softcap)


OUTCOME_COMPONENTS = ("agent_token", "reward_head", "terminal_head")


def restore_weights(model, path, *, require_outcomes=False):
    saved = serialization.msgpack_restore(path.read_bytes())
    state = nnx.state(model, nnx.Param)
    template = nnx.to_pure_dict(state)
    replacement_head = None
    reward_head = None
    if isinstance(model, WorldModel):
        reward_params = saved["dynamics"].get("reward_head")
        legacy = reward_params is not None and reward_params["bias"].shape == (1,)
        reward_target = saved.pop("_reward_target", "event" if legacy else "categorical_points")
        if reward_target not in (("event", "points") if legacy else ("categorical_points",)):
            raise ValueError(f"Incompatible reward target {reward_target} in {path}")
        width = 1 if legacy else REWARD_CLASSES
        if model.dynamics.reward_head.bias.shape != (width,):
            reward_head = nnx.Linear(model.dynamics.reward_head.kernel.shape[0], width, rngs=nnx.Rngs(0))
            template["dynamics"]["reward_head"] = nnx.to_pure_dict(nnx.state(reward_head, nnx.Param))
    # Actor-critic checkpoints predating the two-hot head store one output.
    if isinstance(model, nnx.Dict) and isinstance(model.get("critic"), Critic):
        kernel = saved.get("critic", {}).get("f3", {}).get("kernel")
        if kernel is not None and kernel.shape in ((128, 1), (128, VALUE_BINS)):
            if model.critic.f3.kernel.shape != kernel.shape:
                replacement_head = nnx.Linear(128, kernel.shape[1], rngs=nnx.Rngs(0))
                template["critic"]["f3"] = nnx.to_pure_dict(nnx.state(replacement_head, nnx.Param))
    expected = serialization.to_state_dict(template)
    if isinstance(model, WorldModel) and "dynamics" in saved and not require_outcomes:
        # Older world models predate the agent token and outcome heads.
        for name in OUTCOME_COMPONENTS:
            saved["dynamics"].setdefault(name, expected["dynamics"][name])
    if jax.tree.structure(saved) != jax.tree.structure(expected):
        raise ValueError(f"Incompatible checkpoint parameter keys: {path}")
    restored = serialization.from_state_dict(template, saved)

    for expected, actual in zip(jax.tree.leaves(template), jax.tree.leaves(restored), strict=True):
        if actual.shape != expected.shape:
            raise ValueError(f"Incompatible checkpoint: {actual.shape} vs {expected.shape}")

    if replacement_head is not None:
        model.critic.f3 = replacement_head
        state = nnx.state(model, nnx.Param)
    if isinstance(model, WorldModel):
        if reward_head is not None:
            model.dynamics.reward_head = reward_head
            state = nnx.state(model, nnx.Param)
        model.dynamics.reward_target = reward_target
    nnx.replace_by_pure_dict(state, jax.tree.map(jnp.asarray, restored))
    nnx.update(model, state)


def denoise_next(dynamics, history, actions, noise, denoising_steps):
    step_size = jnp.full((len(noise),), 1.0 / denoising_steps, dtype=noise.dtype)

    def advance(index, latent):
        clean = dynamics(history, actions, latent, index * step_size, step_size)
        return latent + (clean - latent) / (denoising_steps - index)

    return jax.lax.fori_loop(0, denoising_steps, advance, noise)


class Actor(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, action_dim, latent_dim):
        self.f1 = nnx.Linear(latent_dim, 256, rngs=rngs)
        self.f2 = nnx.Linear(256, 128, rngs=rngs)
        self.f3 = nnx.Linear(128, action_dim, rngs=rngs)

    def __call__(self, x):
        x = jax.nn.silu(self.f1(x))
        x = jax.nn.silu(self.f2(x))
        x = self.f3(x)

        return x


VALUE_BINS = 255


def return_bins():
    return jnp.expm1(jnp.linspace(0., jnp.log1p(100.), VALUE_BINS)).at[-1].set(100.)


class Critic(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, latent_dim):
        self.f1 = nnx.Linear(latent_dim, 256, rngs=rngs)
        self.f2 = nnx.Linear(256, 128, rngs=rngs)
        self.f3 = nnx.Linear(128, VALUE_BINS, kernel_init=nnx.initializers.zeros_init(), rngs=rngs)

    def logits(self, x):
        x = jax.nn.silu(self.f1(x))
        x = jax.nn.silu(self.f2(x))
        return self.f3(x)

    @staticmethod
    def value(logits):
        if logits.shape[-1] == 1:
            return logits  # Legacy scalar checkpoint.
        probs = jax.nn.softmax(logits.astype(jnp.float32))
        return (probs * return_bins()).sum(axis=-1, keepdims=True)

    def __call__(self, x):
        return self.value(self.logits(x))

    @staticmethod
    def return_loss(logits, targets):
        targets = jax.lax.stop_gradient(targets)
        if logits.shape[-1] == 1:
            return jnp.square(logits[..., 0] - targets)
        bins = return_bins()
        targets = jnp.clip(targets, bins[0], bins[-1])
        hi = jnp.clip(jnp.searchsorted(bins, targets), 1, VALUE_BINS - 1)
        lo = hi - 1
        upper_weight = (targets - bins[lo]) / (bins[hi] - bins[lo])
        logp = jax.nn.log_softmax(logits.astype(jnp.float32))
        lower = jnp.take_along_axis(logp, lo[..., None], axis=-1)[..., 0]
        upper = jnp.take_along_axis(logp, hi[..., None], axis=-1)[..., 0]
        return -(1 - upper_weight) * lower - upper_weight * upper
