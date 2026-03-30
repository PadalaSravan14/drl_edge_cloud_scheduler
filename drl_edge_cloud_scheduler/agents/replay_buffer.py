import numpy as np
from typing import Tuple


class ReplayBuffer:

    def __init__(self, capacity: int, state_dim: int, seed: int = 42):
        self.capacity  = capacity
        self.state_dim = state_dim
        self.rng       = np.random.default_rng(seed)

        # Pre-allocate arrays for efficiency
        self.states      = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions     = np.zeros(capacity, dtype=np.int64)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.dones       = np.zeros(capacity, dtype=np.float32)

        self._ptr  = 0   # write pointer
        self._size = 0   # current fill level

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store one transition. Overwrites oldest entry when full."""
        self.states[self._ptr]      = state
        self.actions[self._ptr]     = action
        self.rewards[self._ptr]     = reward
        self.next_states[self._ptr] = next_state
        self.dones[self._ptr]       = float(done)

        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:

        idx = self.rng.integers(0, self._size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    def __len__(self) -> int:
        return self._size

    @property
    def is_ready(self) -> bool:
        """True when buffer has at least one sample."""
        return self._size > 0