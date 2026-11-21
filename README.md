# nanoDreamer training

`train.py` selects trainable components with `--train`. New dynamics models use QKNorm, an ungated SiLU feed-forward block, uncapped attention, and the one-frame shortcut objective.

| Mode | Parameters updated | Training objective | Best checkpoint selected by |
|---|---|---|---|
| `dynamics` (default) | Dynamics transformer | Shortcut loss on frozen encoder latents | Next-frame validation BCE |
| `tokenizer` | Encoder and decoder | Frame reconstruction BCE | Reconstruction validation BCE |
| `both` | Encoder, decoder, and dynamics | Reconstruction BCE + weighted shortcut loss | Next-frame validation BCE |
| `outcomes` | Agent token, reward head, terminal head | Raw-points categorical CE + natural-terminal BCE | Sum of held-out outcome losses |
| `heads` | Reward and terminal heads only, including their biases | Reward CE + terminal BCE | Sum of held-out outcome losses |
| `dynamics-outcomes` | Dynamics transformer, agent token, reward and terminal heads | Weighted shortcut loss + reward CE + terminal BCE | Sum of held-out outcome losses |
| `actor-critic` | Actor and critic | Entropy-regularized categorical policy gradient and two-hot value cross-entropy on imagined λ-returns | Mean real episode return |
| `real-actor-critic` | Fresh actor and critic; frozen encoder | Clipped PPO policy loss and two-hot value cross-entropy on real transitions | Mean real episode return |

```bash
python train.py --train tokenizer --output runs/tokenizer
python train.py --train dynamics --tokenizer-checkpoint runs/tokenizer/best.msgpack --output runs/dynamics
python train.py --train outcomes --model-checkpoint runs/dynamics/best.msgpack --output runs/outcomes
python train.py --train both --model-checkpoint runs/dynamics/best.msgpack --output runs/joint
python train.py --train actor-critic --model-checkpoint runs/outcomes/best.msgpack --output runs/agent
python train.py --train real-actor-critic --model-checkpoint runs/outcomes/best.msgpack --output runs/real-agent
```

Each command takes a new output directory. Tokenizer mode writes standalone tokenizer weights. Dynamics and joint modes write full model weights, including the tokenizer. Loading weights starts a fresh optimizer.

Outcome training loads a full pretrained world model and saves the full model with the fitted heads. Older checkpoints initialize the missing agent token and heads. Each label refers to the final recorded action in the context. A time limit ends collection while a natural game over supplies the terminal target. All five games use the same 13 reward classes (integer points 0–12), fitted separately in each world model. Validation reports expected-points MSE and event precision and recall. Outcome training from an older binary or scalar checkpoint initializes a fresh categorical reward head.

Breakout dynamics defaults are batch size 64, context 16, AdamW at 3e-4, and four denoising steps per evaluated frame. Training runs for up to 50 intervals of 2,000 updates, with early stopping after five validations without improvement. Tokenizer pretraining uses Adam for 20 epochs. Run `python train.py --help` for overrides.

Actor–critic mode collects a fixed random-policy replay once, caches encoder latents and trains offline. Each sampled context ends at a real decision point. The actor and critic see the last two latents. The actor samples actions for five imagined transitions using four denoising steps each. Imagination uses the probability-weighted mean of the point classes. Older checkpoints retain their original binary or scalar reward interpretation during policy training. Continuation probability discounts the bootstrapped targets and weights both losses. Defaults are γ = 0.99, λ = 0.95 and entropy coefficient 0.001.

Fresh critics predict 255 probabilities over symmetric exponential return bins spanning symexp(−20) to symexp(20). The scalar value used for advantages and bootstrapping is the probability-weighted mean in reward units. The loss interpolates each detached return target between adjacent bins and applies cross-entropy. Older scalar critic checkpoints load with their original value function.

For imagined PPO, use `--train actor-critic --policy-method ppo` and set `--imagination-horizon` to the number of predicted transitions per replay start. This uses the real-game PPO objective and optimizer rates with cumulative continuation weighting. Each imagined batch produces four optimizer updates. To match the training budget of the earlier scalar-critic Breakout horizon experiment, use `--episodes 10000 --batch-size 32 --epochs 50 --steps-per-epoch 2000 --patience 0` with the frozen joint world-model checkpoint.

Validation samples the policy in 64 real episodes with fixed random seeds, duplicating the initial frame for its first decision. Validation episodes stay outside replay. The best checkpoint contains `actor` and `critic` weights together. Keep the input world-model checkpoint alongside it for the encoder and dynamics.

The real-game baseline is PPO with actor learning rate 1e-4 and critic learning rate 3e-4. The command above uses 32 environments, five transitions per rollout and four gradient passes, for 16 million real actions and 400,000 optimizer updates. It evaluates at initialization and every 320,000 real actions. The three selected policies averaged 15.15 points per held-out Breakout episode. The seed-2 checkpoint scored 16.50. [Per-seed results](runs/real-policy-grid/per-seed.csv) · [Reference checkpoint](runs/real-policy-grid/ppo-1e-4-s2/best.msgpack)

Use `--policy-method a2c|ppo|reinforce|pmpo` with `--train real-actor-critic` for the real-game ablations. Set `--actor-learning-rate` independently of the critic's `--learning-rate`. The real-action budget remains batch size × real horizon × steps per epoch × epochs. Each run writes validation diagnostics, the highest-validation checkpoint and episode returns on 256 held-out games.

A2C uses one advantage-weighted update per rollout. PPO takes four full-batch passes with clip width 0.2 and fixed behavior logits, advantages and value targets. REINFORCE fits a learned state baseline to discounted Monte Carlo returns from batches of episodes. Padded lanes are masked, and the last batch of each validation interval ends exactly at the action budget. PMPO uses the sign of the λ-return advantage, with equal weight on the positive and negative groups, zero advantage in the positive group, and zero contribution from an empty group. Its actor objective has no KL or entropy term.
