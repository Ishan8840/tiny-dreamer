import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=1000, rng=None):
        self.capacity = capacity
        self.episodes = []
        self.rng = rng

    def add_episode(self, episode):
        self.episodes.append(episode)

        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)

    def sample_sequence(self, seq_len=16):
        valid_episodes = [
            ep for ep in self.episodes
            if len(ep["actions"]) >= seq_len
        ]

        if not valid_episodes:
            raise ValueError("No episodes long enough to sample")

        return self._sample_window(valid_episodes, seq_len)

    def _sample_window(self, valid_episodes, seq_len):
        randint = np.random.randint if self.rng is None else self.rng.integers
        episode = valid_episodes[randint(len(valid_episodes))]

        num_steps = len(episode["actions"])
        start = randint(0, num_steps - seq_len + 1)
        end = start + seq_len

        sequence = {
            "observations": episode["observations"][start:end + 1],
            "actions": episode["actions"][start:end],
            "rewards": episode["rewards"][start:end],
            "dones": episode["dones"][start:end],
        }
        if "terminals" in episode:
            sequence["terminals"] = episode["terminals"][start:end]
        return sequence

    def sample_batch(self, batch_size=32, seq_len=16):
        eligible = [ep for ep in self.episodes if len(ep["actions"]) >= seq_len]
        if not eligible:
            raise ValueError("No episodes long enough to sample")
        sequences = [self._sample_window(eligible, seq_len) for _ in range(batch_size)]

        return {
            key: np.stack([seq[key] for seq in sequences])
            for key in sequences[0] if all(key in seq for seq in sequences)
        }
