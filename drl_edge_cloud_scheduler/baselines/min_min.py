import numpy as np
from typing import Optional, List
class MinMinScheduler:
    def __init__(self, config: dict, resources=None):
        self.config    = config
        self.resources = resources   # injected by evaluator for exec-time calc

    def select_action(
        self,
        state: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        training: bool = False,
        task=None,
    ) -> int:
        if action_mask is not None:
            valid = np.where(action_mask)[0]
        else:
            n = (self.config['environment']['num_edge_devices']
                 + self.config['environment']['num_cloud_servers'])
            valid = np.arange(n)

        if len(valid) == 0:
            return 0

        if self.resources is None or task is None:
            return int(valid[0])

        # Choose resource that minimises execution time = CPU / speed
        best_idx  = valid[0]
        best_time = float('inf')
        for idx in valid:
            r = self.resources[idx]
            exec_t = task.cpu_demand / max(r.cpu_speed, 1e-9)
            if exec_t < best_time:
                best_time = exec_t
                best_idx  = idx

        return int(best_idx)

    def set_resources(self, resources):
        self.resources = resources

    def store_transition(self, *args, **kwargs): pass
    def update(self): return None
    def decay_epsilon(self): pass
    def save(self, path: str): pass
    def load(self, path: str): pass