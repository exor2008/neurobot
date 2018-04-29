import random
from collections import deque
import pandas as pd
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, columns=None):
        self.capacity = capacity
        self.columns = columns
        self._buffer = deque()

    def _get_size(self):
        return len(self._buffer)

    size = property(_get_size)

    def get_batch(self, batch_size):
        # Randomly sample batch_size examples
        batch_size = min(batch_size, self.size)
        batch = np.asarray(random.sample(self._buffer, batch_size))
        batch = pd.DataFrame(batch)
        if self.columns:
            batch.columns = self.columns
        return batch

    def add(self, state, actions, reward, new_state, done):
        if self.size < self.capacity:
            self._buffer.append([state, actions, reward, new_state, done])
        else:
            self._buffer.popleft()
            self._buffer.append([state, actions, reward, new_state, done])

    def clear(self):
        self._buffer = deque()
