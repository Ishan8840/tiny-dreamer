"""Render and verify examples from the matched 60k Breakout evaluation."""
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flax import nnx
from matplotlib import font_manager

from evaluate_matched_breakout import BATCH, CONTEXT, HORIZONS, OUT, RUNS, WorldModel, predict, restore_tokenizer


BG = "#f6f7f9"
INK = "#172539"
CELL = 13
SIDE = CELL * 10
REG = font_manager.findfont("DejaVu Sans")
BOLD = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))


def label(draw, xy, value, size=16, bold=False):
    draw.text(xy, value, font=ImageFont.truetype(BOLD if bold else REG, size), fill=INK)


def board(draw, x, y, frame, target=None):
    draw.rectangle((x, y, x + SIDE - 1, y + SIDE - 1), fill="#172537")
    for row, col in np.ndindex(10, 10):
        left, top = x + col * CELL, y + row * CELL
        box = (left + 1, top + 1, left + CELL - 2, top + CELL - 2)
        if target is not None:
            missed = (target[row, col] & ~frame[row, col]).any()
            extra = (frame[row, col] & ~target[row, col]).any()
            if missed or extra:
                draw.rectangle(box, fill="#b58bff" if missed and extra else "#ff8d70" if missed else "#57d2cf")
            continue
        paddle, ball, trail, brick = frame[row, col]
        if brick:
            draw.rectangle(box, fill="#80bc92")
        if paddle:
            draw.rectangle((left, top + 8, left + CELL - 1, top + 12), fill="#5abbff")
        if trail:
            draw.line((left + 2, top + 2, left + 10, top + 10), fill="#cb9fe1", width=2)
        if ball:
            draw.ellipse((left + 3, top + 2, left + 10, top + 9), fill="#ffc562")


def panel(pairs, title, headings, row_names):
    im = Image.new("RGB", (660, 101 + len(pairs) * 149 + 58), BG)
    draw = ImageDraw.Draw(im)
    label(draw, (18, 11), title, 21, True)
    xs = (170, 335, 500)
    for x, heading in zip(xs, headings):
        label(draw, (x, 59), heading, 16, True)
    for i, ((truth, pred), name) in enumerate(zip(pairs, row_names)):
        y = 89 + i * 149
        label(draw, (18, y + 42), name, 16, True)
        bits = int((truth != pred).sum())
        label(draw, (18, y + 67), f"{bits} bit error" + ("s" if bits != 1 else ""), 14)
        board(draw, xs[0], y, truth)
        board(draw, xs[1], y, pred)
        board(draw, xs[2], y, pred, truth)
    y = im.height - 58
    label(draw, (18, y), "Board: gold ball, blue paddle, purple trail, green bricks", 15)
    label(draw, (18, y + 23), "Errors: coral missing, teal extra, violet both at one cell", 15)
    return im


def main():
    run = RUNS["AdamW 60k"]
    with np.load(OUT / f"{run}-scores.npz") as data:
        exact, distances = data["exact"], data["ball_distance"]
    perfect = exact.all(axis=0)
    masks = (perfect, ~perfect & exact[-1], ~exact[-1])
    names = ("Perfect", "Recovered", "Inexact at +20")
    rng = np.random.default_rng(60)
    ids = [int(rng.choice(np.flatnonzero(mask))) for mask in masks]
    with np.load(OUT / "windows.npz") as data:
        obs, actions = data["observations"], data["actions"]
    model = WorldModel(nnx.Rngs(0), 4, 3, qknorm=False)
    restore_tokenizer(model, Path("runs") / run / "best.msgpack")
    folder = OUT / "visuals-60k"
    folder.mkdir(exist_ok=True)
    examples = []
    with (folder / "examples.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(["outcome", "window_id", "group_size", "horizon", "bit_errors", "ball_error"])
        for window, name, mask in zip(ids, names, masks):
            start = window // BATCH * BATCH
            batch = np.minimum(np.arange(start, start + BATCH), len(obs)-1)
            logits = np.asarray(predict(model, obs[batch, :CONTEXT].astype(np.float32),
                                        actions[batch], batch, horizons=HORIZONS))[:,window-start]
            pred = logits >= 0
            truth = obs[window, CONTEXT:]
            errors = (pred != truth).sum(axis=(1, 2, 3))
            np.testing.assert_array_equal(errors == 0, exact[:,window])
            p = logits[...,1].reshape(20,100).argmax(axis=1)
            t = truth[...,1].reshape(20,100).argmax(axis=1)
            np.testing.assert_allclose(np.hypot(p//10-t//10,p%10-t%10), distances[:,window])
            examples.append(list(zip(truth, pred)))
            for h in HORIZONS:
                writer.writerow([name, window, int(mask.sum()), h, errors[h-1], distances[h-1, window]])
            # Include the first mismatch even when it falls between report horizons.
            horizons = sorted({1, 5, 10, 20} | ({int(np.flatnonzero(errors)[0])+1} if errors.any() else set()))
            panel([examples[-1][h-1] for h in horizons], f"{name}  |  Window {window}",
                  ("Real", "60k prediction", "Errors"), [f"+{h}" for h in horizons]).save(folder / f"window-{window}.png")
    frames = [panel([example[h-1] for example in examples], f"Breakout   /   60k checkpoint   /   Frame +{h:02d} of 20",
                    ("Real", "60k prediction", "Errors"), names) for h in HORIZONS]
    frames[0].save(folder / "rollouts.gif", save_all=True, append_images=frames[1:], duration=[650]*19+[1800], loop=0)
    frames[-1].save(folder / "frame-20.png")
    print(ids, [int(mask.sum()) for mask in masks])


if __name__ == "__main__":
    main()
