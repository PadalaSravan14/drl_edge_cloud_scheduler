
import numpy as np
from typing import Optional


class FIFOScheduler:

    def __init__(self, config: dict):
        self.config = config

    def select_action(
        self,
        state: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        training: bool = False,
    ) -> int:
        """Return the first valid resource index (FIFO order)."""
        if action_mask is not None:
            valid = np.where(action_mask)[0]
            if len(valid) > 0:
                return int(valid[0])
        n = self.config['environment']['num_edge_devices'] \
            + self.config['environment']['num_cloud_servers']
        return 0  # fallback

    # Stub methods for interface compatibility with Trainer/Evaluator
    def store_transition(self, *args, **kwargs): pass
    def update(self): return None
    def decay_epsilon(self): pass
    def save(self, path: str): pass
    def load(self, path: str): pass