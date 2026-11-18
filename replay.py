import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.episodes = []

    def add_episode(self, episode):
        self.episodes.append(episode)

        if len(self.episodes > self.capacity):
            self.episodes.pop(0)

    def sample_sequence(self, seq_len=16):
        episode = self.episodes[np.random.randint(len(self.episodes))]

        num_steps = len(episode["actions"])
        start = np.random.randint(0, num_steps - seq_len + 1)
        end = start + seq_len

        return {
            "observations": episode["observations"][start:end + 1],
            "actions": episode["actions"][start:end],
            "rewards": episode["rewards"][start:end],
            "dones": episode["dones"][start:end],
        }

    def sample_batch(self, batch_size=32, seq_len=16):
        sequences = [self.sample_sequence(seq_len) for _ in range(batch_size)]

        return {
            key: np.stack([seq[key] for seq in sequences])
            for key in sequences[0]
        }