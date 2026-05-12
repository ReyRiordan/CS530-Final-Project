from collections import deque
import random
import numpy as np


class ReplayBuffer:
    """
    Regular ole replay buffer (queue), contains transitions of format:
    (troop_tensor, flat_vector, action, reward, next_troop_tensor, next_flat_vector, done)
    """

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, troop_tensor, flat_vector, action, reward, next_troop_tensor, next_flat_vector, done):
        self.buffer.append((
            troop_tensor.astype(np.float32),
            flat_vector.astype(np.float32),
            int(action),
            float(reward),
            next_troop_tensor.astype(np.float32),
            next_flat_vector.astype(np.float32),
            bool(done),
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        troop_tensors, flat_vectors, actions, rewards, next_troop_tensors, next_flat_vectors, dones = zip(*batch)
        return (
            np.stack(troop_tensors),
            np.stack(flat_vectors),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_troop_tensors),
            np.stack(next_flat_vectors),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
