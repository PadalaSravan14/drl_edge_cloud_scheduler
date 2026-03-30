import numpy as np
from typing import List, Optional
from sklearn.decomposition import PCA


class StateBuilder:
    """
    Implements Algorithm 2: State Representation Generation.
    Normalises and optionally reduces the state vector via PCA.
    """

    def __init__(self, config: dict):
        self.config  = config
        self._env    = config['environment']
        self._dqn    = config['dqn']

        self.K = self._env['max_pending_tasks']
        self.M = self._env['num_edge_devices']
        self.N = self._env['num_cloud_servers']

        # Normalisation constants from paper (Section 4.3)
        self.cpu_min, self.cpu_max   = self._env['task_cpu_range']
        self.mem_min, self.mem_max   = self._env['task_mem_range']
        self.dl_range                = self._env['task_deadline_range'][1]  # 10s
        self.lat_min, self.lat_max   = self._env['network_latency_range']
        self.bw_min,  self.bw_max    = self._env['network_bandwidth_range']
        self.q_max                   = self._env['queue_threshold'] * 2      # normalise queue

        # d_state = 4K + 4(M+N) + M + 2MN  (Section 4.3)
        self._raw_dim = (
            4 * self.K
            + 4 * (self.M + self.N)
            + self.M
            + 2 * self.M * self.N
        )

        # PCA
        self.use_pca     = self._dqn.get('use_pca', True)
        self.pca_var     = self._dqn.get('pca_variance', 0.95)
        self._pca: Optional[PCA] = None
        self._pca_fitted = False

        self._state_dim: int = self._raw_dim   # updated after PCA fit


    @property
    def state_dim(self) -> int:
        return self._state_dim

    def build(
        self,
        task_queue: list,
        resources: list,
        latencies: np.ndarray,
        bandwidths: np.ndarray,
        current_time: float,
    ) -> np.ndarray:
        """
        Build and return the normalised state vector.
        Implements Algorithm 2 exactly.
        """
        features = []


        n_tasks = min(self.K, len(task_queue))
        for i in range(self.K):
            if i < n_tasks:
                task = task_queue[i]
                # cpu_norm = (CPU - 1000) / 9000
                cpu_norm = (task.cpu_demand - self.cpu_min) / max(
                    self.cpu_max - self.cpu_min, 1e-9
                )
                # mem_norm = (Mem - 100) / 412
                mem_norm = (task.mem_demand - self.mem_min) / max(
                    self.mem_max - self.mem_min, 1e-9
                )
                # deadline_norm = remaining_deadline / 10
                remaining = max(0.0, task.deadline - current_time)
                dl_norm = remaining / max(self.dl_range, 1e-9)
                # priority one-hot
                onehot = task.priority_onehot
            else:
                # Padding with zeros if queue has fewer than K tasks
                cpu_norm = 0.0
                mem_norm = 0.0
                dl_norm  = 0.0
                onehot   = [0, 0, 0]

            features.extend([cpu_norm, mem_norm, dl_norm] + onehot)

        edge_resources  = [r for r in resources if r.resource_type == 'edge']
        cloud_resources = [r for r in resources if r.resource_type == 'cloud']
        ordered = edge_resources + cloud_resources

        for r in ordered:
            cpu_avail  = r.cpu_available / max(r.cpu_capacity, 1e-9)
            mem_avail  = r.mem_available / max(r.mem_capacity, 1e-9)
            queue_norm = r.queue_length   / max(self.q_max, 1e-9)
            if r.resource_type == 'edge':
                energy_norm = r.energy_available / max(r.energy_capacity, 1e-9)
            else:
                energy_norm = 1.0  # cloud = unlimited power
            features.extend([cpu_avail, mem_avail, queue_norm, energy_norm])

        for i in range(self.M):
            for j in range(self.N):
                # latency_norm = (l_ij - 1) / 99
                lat_norm = (latencies[i, j] - self.lat_min) / max(
                    self.lat_max - self.lat_min, 1e-9
                )
                # bw_norm = log(bw) / log(1000)  — log-scale for wide range
                bw_val = max(bandwidths[i, j], 1.0)
                bw_norm = np.log(bw_val) / np.log(max(self.bw_max, 2.0))
                features.extend([
                    float(np.clip(lat_norm, 0.0, 1.0)),
                    float(np.clip(bw_norm,  0.0, 1.0)),
                ])

        state = np.array(features, dtype=np.float32)

        if self.use_pca and self._pca_fitted and self._pca is not None:
            state = self._pca.transform(state.reshape(1, -1)).flatten().astype(np.float32)

        return state

    def fit_pca(self, states: np.ndarray):
        """
        Fit PCA on a batch of collected states.
        Call this after initial random exploration before training.

        Args:
            states: (N_samples, raw_dim) array of raw state vectors.
        """
        if not self.use_pca:
            return
        n_components = min(
            int(0.20 * self._raw_dim),   # cap at 20% of raw dim
            states.shape[0] - 1,
            states.shape[1],
        )
        n_components = max(n_components, 1)
        self._pca = PCA(n_components=n_components, svd_solver='randomized')
        self._pca.fit(states)

        # Find components preserving pca_var variance
        cum_var = np.cumsum(self._pca.explained_variance_ratio_)
        n_keep = int(np.searchsorted(cum_var, self.pca_var) + 1)
        n_keep = min(n_keep, n_components)
        self._pca.components_ = self._pca.components_[:n_keep]
        self._state_dim = n_keep
        self._pca_fitted = True

        print(f"[StateBuilder] PCA fitted: {self._raw_dim} → {n_keep} dims "
              f"(≥{self.pca_var*100:.0f}% variance)")

    def reset_pca(self):
        self._pca         = None
        self._pca_fitted  = False
        self._state_dim   = self._raw_dim