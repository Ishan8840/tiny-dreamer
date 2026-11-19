# nanoDreamer training

`train.py` selects trainable components with `--train`. New dynamics models use QKNorm, an ungated SiLU feed-forward block, uncapped attention, and the one-frame shortcut objective.

| Mode | Parameters updated | Training objective | Best checkpoint selected by |
|---|---|---|---|
| `dynamics` (default) | Dynamics transformer | Shortcut loss on frozen encoder latents | Next-frame validation BCE |
| `tokenizer` | Encoder and decoder | Frame reconstruction BCE | Reconstruction validation BCE |
| `both` | Encoder, decoder, and dynamics | Reconstruction BCE + weighted shortcut loss | Next-frame validation BCE |
| `outcomes` | Agent token, reward head, terminal head | Positive-reward event BCE + natural-terminal BCE | Sum of held-out event BCEs |
| `heads` | Reward and terminal heads only, including their biases | Both event BCEs | Sum of held-out event BCEs |
| `dynamics-outcomes` | Dynamics transformer, agent token, reward and terminal heads | Weighted shortcut loss + both event BCEs | Sum of held-out event BCEs |

```bash
python train.py --train tokenizer --output runs/tokenizer
python train.py --train dynamics --tokenizer-checkpoint runs/tokenizer/best.msgpack --output runs/dynamics
python train.py --train outcomes --model-checkpoint runs/dynamics/best.msgpack --output runs/outcomes
python train.py --train both --model-checkpoint runs/dynamics/best.msgpack --output runs/joint
```

Each command takes a new output directory. Tokenizer mode writes standalone tokenizer weights. Dynamics and joint modes write full model weights, including the tokenizer. Loading weights starts a fresh optimizer.

Outcome training loads a full pretrained world model and saves the full model with the fitted heads. Older checkpoints initialize the missing agent token and heads. Each label refers to the final recorded action in the context. A time limit ends collection while a natural game over supplies the terminal target. Validation reports precision, recall, and positive counts separately for both events over all eligible held-out transitions.

Breakout dynamics defaults are batch size 64, context 16, AdamW at 3e-4, and four denoising steps per evaluated frame. Training runs for up to 50 intervals of 2,000 updates, with early stopping after five validations without improvement. Tokenizer pretraining uses Adam for 20 epochs. Run `python train.py --help` for overrides.
