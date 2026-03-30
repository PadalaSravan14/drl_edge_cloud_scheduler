import numpy as np
from typing import Optional


class RoundRobinScheduler:
    """
    Round-Robin scheduler: cyclic assignment across valid resources.
    """

    def __init__(self, config: dict):
        self.config   = config
        self._counter = 0

    def select_action(
        self,
        state: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        training: bool = False,
    ) -> int:
        if action_mask is not None:
            valid = np.where(action_mask)[0]
        else:
            n = (self.config['environment']['num_edge_devices']
                 + self.config['environment']['num_cloud_servers'])
            valid = np.arange(n)

        if len(valid) == 0:
            return 0

        idx = self._counter % len(valid)
        self._counter += 1
        return int(valid[idx])

    def reset(self):
        self._counter = 0

    def store_transition(self, *args, **kwargs): pass
    def update(self): return None
    def decay_epsilon(self): pass
    def save(self, path: str): pass
    def load(self, path: str): pass