# nanoDreamer training

`train.py` selects trainable components with `--train`. New dynamics models use QKNorm, an ungated SiLU feed-forward block, uncapped attention, and the one-frame shortcut objective.

| Mode | Parameters updated | Training objective | Best checkpoint selected by |
|---|---|---|---|
| `dynamics` (default) | Dynamics transformer | Shortcut loss on frozen encoder latents | Next-frame validation BCE |
| `tokenizer` | Encoder and decoder | Frame reconstruction BCE | Reconstruction validation BCE |
| `both` | Encoder, decoder, and dynamics | Reconstruction BCE + weighted shortcut loss | Next-frame validation BCE |

```bash
python train.py --train tokenizer --output runs/tokenizer
python train.py --train dynamics --tokenizer-checkpoint runs/tokenizer/best.msgpack --output runs/dynamics
python train.py --train both --model-checkpoint runs/dynamics/best.msgpack --output runs/joint
```

Each command takes a new output directory. Tokenizer mode writes standalone tokenizer weights. Dynamics and joint modes write full model weights, including the tokenizer. Loading weights starts a fresh optimizer.

Breakout dynamics defaults are batch size 64, context 16, AdamW at 3e-4, and four denoising steps per evaluated frame. Training runs for up to 50 intervals of 2,000 updates, with early stopping after five validations without improvement. Tokenizer pretraining uses Adam for 20 epochs. Run `python train.py --help` for overrides.
