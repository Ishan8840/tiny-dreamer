# TinyDreamer

Train a small world model to play MinAtar through imagination. A convolutional tokenizer encodes binary game frames. An action-conditioned transformer predicts future latents, reward points and episode termination. An actor chooses actions from the two most recent latents, and a distributional critic estimates their expected return.

## Installation

From the repository root, install the locked dependencies with Python 3.11 and uv.

```sh
uv sync --locked
source .venv/bin/activate
```

For a Linux machine with an NVIDIA GPU and a compatible CUDA 12 driver, install the CUDA plugin for the same JAX version into that environment.

```sh
uv pip install "jax[cuda12]==$(python -c 'import jax; print(jax.__version__)')"
python -c 'import jax; print(jax.devices())'
```

## Train and evaluate

These short Breakout commands exercise the full pipeline. Each stage writes to a fresh directory. Increase the budgets for a trained agent.

```sh
python scripts/train.py --train tokenizer --output outputs/tokenizer \
  --episodes 12 --epochs 1 --batch-size 8

python scripts/train.py --train dynamics-outcomes --output outputs/world \
  --tokenizer-checkpoint outputs/tokenizer/best.msgpack \
  --episodes 12 --epochs 1 --steps-per-epoch 1 --batch-size 4

python scripts/train.py --train online --output outputs/policy \
  --model-checkpoint outputs/world/best.msgpack --episodes 12 \
  --epochs 1 --collect-steps 32 --world-updates 1 --steps-per-epoch 1 \
  --batch-size 4 --eval-episodes 4

python scripts/evaluate.py --policy outputs/policy/best.msgpack --episodes 4
```

Set the same `--game` on each training command to train Asterix, Breakout, Freeway, Seaquest or SpaceInvaders. Evaluation reads the game from the saved configuration. Keep the policy directory together when moving it, including its tokenizer and configuration.

## Training modes

| Mode | What learns | Data and selection metric |
| --- | --- | --- |
| `tokenizer` | Encoder and decoder | Random gameplay, held-out reconstruction BCE |
| `dynamics-outcomes` | Dynamics, reward and terminal heads | Frozen tokenizer, held-out reward and terminal loss |
| `online` | Dynamics, outcome heads, actor and critic | Online and random replay for dynamics, imagined PPO for behavior, real-game return for selection |
| `actor-critic` | Actor and critic | Fixed world model and replay, real-game return for selection. Use `--policy-method ppo` for PPO. |
| `real-actor-critic` | Actor and critic | Real game transitions using a frozen encoder, real-game return for selection |
| `dynamics` / `both` | Dynamics alone or with tokenizer | Shortcut prediction, with reconstruction in joint mode |
| `heads` / `outcomes` | Outcome heads alone or with shared outcome features | Real reward points and natural game-over labels |
| `rollout-error` | Latent error probe | Error of imagined predictions against held-out encoded observations |

The default tokenizer has 64 latent dimensions. The world model uses five frames of context, three training targets and four denoising steps per predicted frame. In online mode each epoch collects `--collect-steps` real actions, takes `--world-updates` dynamics updates and trains on `--steps-per-epoch` imagined PPO batches. Each batch produces four optimizer updates. `--episodes` controls initial random collection. Policy validation uses separate real gameplay. Checkpoints contain weights, and loading a checkpoint starts a fresh optimizer.

For one-step world-model updates, set `--training-horizon 1 --reward-fraction 0.25` to draw extra scoring transitions for the reward head. Importance weights recover the original replay distribution. For multi-step training, `--detach-rollout` stops gradients through the predicted inputs to later steps. An offline imagined policy can use a fitted error probe with `--error-checkpoint PATH --error-threshold HIGH`; adding `--error-low LOW` gives 90% trust at LOW and 50% at HIGH.

Use `--diffusion-forcing` to independently mix each context latent with Gaussian noise during shortcut training. The transformer receives a signal level for each frame. The shortcut target is still the next latent at each supervised rollout step, and reward and terminal heads use the ordinary rollout context. Prediction starts from clean observed history at evaluation. This option adds no parameters and works with existing checkpoints.

For autoregressive inference with a diffusion-forcing checkpoint, `--inference-history-noise 0.0625` mixes each previously predicted context latent with 1/16 Gaussian noise. The denoiser receives its corresponding signal level. Observed frames stay clean, and each predicted frame uses one context-noise draw across all denoising substeps. The default is zero.

## Where to make changes

| File in `tiny_dreamer/` | Responsibility |
| --- | --- |
| `config.py` | Configuration and CLI validation |
| `model.py` | Tokenizer, transformer, prediction heads, actor and critic |
| `data.py` | Game creation, episode collection and replay windows |
| `world_model_learning.py` | Tokenizer, shortcut and outcome losses |
| `imagination.py` | Autoregressive imagined transitions |
| `actor_critic.py` | Returns, policy losses and critic updates |
| `training.py` | Training loops and model selection |
| `rollout_error.py` | Error-probe fitting and calibration |
| `checkpoints.py` | Parameter serialization and compatible weight loading |

Run `python scripts/train.py --help` for all flags or `python -m unittest discover -s tests` for the regression tests.

## License

[MIT](LICENSE)
