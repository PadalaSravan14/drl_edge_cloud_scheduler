import numpy as np
from typing import Optional


class GreedyEnergyScheduler:
    def __init__(self, config: dict, resources=None):
        self.config    = config
        self.resources = resources

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

        best_idx    = valid[0]
        best_energy = float('inf')

        for idx in valid:
            r      = self.resources[idx]
            exec_t = task.cpu_demand / max(r.cpu_speed, 1e-9)
            if r.resource_type == 'edge':
                energy = r.power_consumption * exec_t
            else:
                energy = 0.0   # cloud has no battery constraint

            if energy < best_energy:
                best_energy = energy
                best_idx    = idx

        return int(best_idx)

    def set_resources(self, resources):
        self.resources = resources

    def store_transition(self, *args, **kwargs): pass
    def update(self): return None
    def decay_epsilon(self): pass
    def save(self, path: str): pass
    def load(self, path: str): pass