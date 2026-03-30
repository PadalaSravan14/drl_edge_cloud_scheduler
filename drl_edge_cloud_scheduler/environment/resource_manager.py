import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Resource:
    resource_id: int
    resource_type: str         # 'edge' or 'cloud'
    cpu_capacity: float        # MHz
    mem_capacity: float        # MB
    energy_capacity: float     # Wh (inf for cloud)
    power_consumption: float   # Watts (edge only)
    cpu_speed: float           # MIPS

    # Dynamic state — reset each episode
    cpu_available: float = field(default=0.0)
    mem_available: float = field(default=0.0)
    energy_available: float = field(default=0.0)
    queue_length: int = field(default=0)
    failed: bool = field(default=False)

    def __post_init__(self):
        self.cpu_available = self.cpu_capacity
        self.mem_available = self.mem_capacity
        self.energy_available = self.energy_capacity

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def reset(self):
        self.cpu_available = self.cpu_capacity
        self.mem_available = self.mem_capacity
        self.energy_available = self.energy_capacity
        self.queue_length = 0
        self.failed = False

    @property
    def utilization(self) -> float:
        """CPU utilization ratio [0, 1]."""
        return 1.0 - (self.cpu_available / max(self.cpu_capacity, 1e-9))

    # ------------------------------------------------------------------
    # Feasibility — action masking (Section 4.4)
    # ------------------------------------------------------------------
    def can_accept(self, task) -> bool:
        """
        Returns True iff resource can accept the task without violating
        CPU, memory, and energy capacity constraints (Eqs. 15-17).
        """
        if self.failed:
            return False
        if self.cpu_available < task.cpu_demand:
            return False
        if self.mem_available < task.mem_demand:
            return False
        if self.resource_type == 'edge':
            exec_time = task.cpu_demand / max(self.cpu_speed, 1e-9)
            estimated_energy = self.power_consumption * exec_time
            if self.energy_available < estimated_energy:
                return False
        return True

    # ------------------------------------------------------------------
    # Allocation / Deallocation
    # ------------------------------------------------------------------
    def allocate(self, task):
        """Consume resources when task is assigned."""
        self.cpu_available = max(0.0, self.cpu_available - task.cpu_demand)
        self.mem_available = max(0.0, self.mem_available - task.mem_demand)
        self.queue_length += 1
        if self.resource_type == 'edge':
            exec_time = task.cpu_demand / max(self.cpu_speed, 1e-9)
            energy_used = self.power_consumption * exec_time
            self.energy_available = max(0.0, self.energy_available - energy_used)

    def deallocate(self, task):
        """Release resources when task completes."""
        self.cpu_available = min(self.cpu_capacity, self.cpu_available + task.cpu_demand)
        self.mem_available = min(self.mem_capacity, self.mem_available + task.mem_demand)
        self.queue_length = max(0, self.queue_length - 1)


class ResourceManager:


    def __init__(self, config: dict, rng: Optional[np.random.Generator] = None):
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.resources: List[Resource] = []
        self._latencies: Optional[np.ndarray] = None
        self._bandwidths: Optional[np.ndarray] = None
        self._build_resources()
        self._sample_network()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_resources(self):
        """Build M edge devices + N cloud servers."""
        env = self.config['environment']
        self.resources = []

        for i in range(env['num_edge_devices']):
            cpu_cap = float(self.rng.uniform(*env['edge_cpu_range']))
            mem_cap = float(self.rng.uniform(*env['edge_mem_range']))
            energy_cap = float(self.rng.uniform(*env['edge_energy_range']))
            power = float(self.rng.uniform(*env['edge_power_range']))
            r = Resource(
                resource_id=i,
                resource_type='edge',
                cpu_capacity=cpu_cap,
                mem_capacity=mem_cap,
                energy_capacity=energy_cap,
                power_consumption=power,
                cpu_speed=cpu_cap / 10.0,
            )
            self.resources.append(r)

        n_edge = env['num_edge_devices']
        for j in range(env['num_cloud_servers']):
            cpu_cap = float(self.rng.uniform(*env['cloud_cpu_range']))
            mem_cap = float(self.rng.uniform(*env['cloud_mem_range']))
            r = Resource(
                resource_id=n_edge + j,
                resource_type='cloud',
                cpu_capacity=cpu_cap,
                mem_capacity=mem_cap,
                energy_capacity=float('inf'),
                power_consumption=0.0,
                cpu_speed=cpu_cap / 10.0,
            )
            self.resources.append(r)

    def _sample_network(self):
        """Sample latency/bandwidth matrix for edge→cloud pairs."""
        env = self.config['environment']
        M = self.num_edge
        N = self.num_cloud
        self._latencies = self.rng.uniform(
            *env['network_latency_range'], size=(M, N)
        )
        self._bandwidths = self.rng.uniform(
            *env['network_bandwidth_range'], size=(M, N)
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, randomize_capacities: bool = False):

        if randomize_capacities:
            self._build_resources()
        else:
            for r in self.resources:
                r.reset()
        self._sample_network()

    # ------------------------------------------------------------------
    # Failure injection (Section 4.7, Table 12)
    # ------------------------------------------------------------------
    def inject_failures(self, failure_prob: Optional[float] = None):
        """Mark random resources as failed using Bernoulli(p_fail)."""
        p = failure_prob if failure_prob is not None else \
            self.config['environment']['failure_probability']
        for r in self.resources:
            if self.rng.random() < p:
                r.failed = True

    def recover_all(self):
        for r in self.resources:
            r.failed = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def edge_resources(self) -> List[Resource]:
        return [r for r in self.resources if r.resource_type == 'edge']

    @property
    def cloud_resources(self) -> List[Resource]:
        return [r for r in self.resources if r.resource_type == 'cloud']

    @property
    def active_resources(self) -> List[Resource]:
        return [r for r in self.resources if not r.failed]

    @property
    def num_resources(self) -> int:
        return len(self.resources)

    @property
    def num_edge(self) -> int:
        return self.config['environment']['num_edge_devices']

    @property
    def num_cloud(self) -> int:
        return self.config['environment']['num_cloud_servers']

    # ------------------------------------------------------------------
    # Network conditions (Eq. 6)
    # ------------------------------------------------------------------
    def get_network_conditions(self):
        """
        Returns:
            latencies  (M, N) ndarray — ms
            bandwidths (M, N) ndarray — Mbps
        """
        return self._latencies.copy(), self._bandwidths.copy()

    def get_latency(self, edge_idx: int, cloud_idx: int) -> float:
        return float(self._latencies[edge_idx, cloud_idx])

    def get_bandwidth(self, edge_idx: int, cloud_idx: int) -> float:
        return float(self._bandwidths[edge_idx, cloud_idx])

    # ------------------------------------------------------------------
    # System-level stats
    # ------------------------------------------------------------------
    def average_utilization(self) -> float:
        active = [r for r in self.resources if not r.failed]
        if not active:
            return 0.0
        return float(np.mean([r.utilization for r in active]))

    def average_queue_length(self) -> float:
        active = [r for r in self.resources if not r.failed]
        if not active:
            return 0.0
        return float(np.mean([r.queue_length for r in active]))

    def total_energy_consumed(self) -> float:
        """Sum of energy depleted across all edge devices (kWh)."""
        total_wh = sum(
            r.energy_capacity - r.energy_available
            for r in self.edge_resources
        )
        return total_wh / 1000.0  # Wh → kWh

    def get_valid_actions(self, task) -> List[int]:

        return [
            r.resource_id for r in self.resources
            if r.can_accept(task)
        ]
