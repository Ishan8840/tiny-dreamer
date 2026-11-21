"""Train a TinyDreamer component or run the online learning loop."""

from tiny_dreamer.config import parse_args
from tiny_dreamer.training import run_training

if __name__ == "__main__":
    architecture, training = parse_args()
    run_training(architecture, training)
