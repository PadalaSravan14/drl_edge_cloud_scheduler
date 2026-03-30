import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from collections import deque


class EdgeCloudEnv:
    """
    Gym-compatible MDP environment for edge-cloud task scheduling.
    Observation : normalised state vector s_t ∈ ℝ^d_state
    Action      : integer index into resource_manager.resources
    Reward      : scalar from multi-objective reward function
    """

    def __init__(self, config: dict, workload_generator=None, rng=None):
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self._env_cfg = config['environment']
        self._wl_cfg  = config['workload']

        from environment.resource_manager import ResourceManager
        from state_representation.state_builder import StateBuilder
        from reward.reward_function import RewardFunction
        from environment.workload_generator import SyntheticWorkloadGenerator

        self.resource_manager = ResourceManager(config, rng=self.rng)
        self.state_builder    = StateBuilder(config)
        self.reward_fn        = RewardFunction(config)

        self.workload_gen = (
            workload_generator
            if workload_generator is not None
            else SyntheticWorkloadGenerator(config, rng=self.rng)
        )

        # Episode state
        self.task_queue: deque      = deque()
        self.current_time: float    = 0.0
        self.step_count: int        = 0
        self.completed_tasks: list  = []
        self.sla_violations: int    = 0
        self.total_tasks_arrived: int = 0

        self._task_list_mode: bool  = False
        self._remaining_tasks: list = []
        self._episode_batches: list = []
        self._batch_idx: int        = 0

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------
    def reset(
        self,
        task_list=None,
        randomize_resources: bool = False,
        inject_failures: bool = False,
    ) -> np.ndarray:
        """
        Reset environment.
        task_list: pre-built list of Task objects (real-dataset mode).
        """
        self.resource_manager.reset(randomize_capacities=randomize_resources)
        if inject_failures:
            self.resource_manager.inject_failures()

        self.task_queue.clear()
        self.current_time      = 0.0
        self.step_count        = 0
        self.completed_tasks   = []
        self.sla_violations    = 0
        self.total_tasks_arrived = 0

        self.workload_gen.reset_counter()

        if task_list is not None:
            self._task_list_mode  = True
            self._remaining_tasks = list(task_list)
        else:
            self._task_list_mode  = False
            self._episode_batches = self.workload_gen.generate_episode(
                num_tasks=self._wl_cfg['synthetic_num_tasks'],
                burst_at_step=self._wl_cfg.get('burst_step'),
                burst_multiplier=self._wl_cfg.get('burst_multiplier', 2.0),
            )
            self._batch_idx = 0

        self._arrive_tasks()
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one scheduling decision."""
        info: Dict[str, Any] = {}

        if len(self.task_queue) == 0:
            return self._get_state(), 0.0, self._is_done(), info

        task = self.task_queue.popleft()
        resource = self.resource_manager.resources[action]

        # Feasibility guard (action masking should prevent this)
        if not resource.can_accept(task):
            self.task_queue.appendleft(task)
            return self._get_state(), -1.0, self._is_done(), {'invalid_action': True}

        resource.allocate(task)

        # Latency = execution time + communication time (Eq.11)
        exec_time = task.cpu_demand / max(resource.cpu_speed, 1e-9)
        comm_time = 0.0
        if resource.resource_type == 'cloud':
            edge_idx  = 0
            cloud_idx = resource.resource_id - self.resource_manager.num_edge
            latency_ms  = self.resource_manager.get_latency(edge_idx, cloud_idx)
            bandwidth   = self.resource_manager.get_bandwidth(edge_idx, cloud_idx)
            comm_time   = (latency_ms / 1000.0) + (task.data_size / max(bandwidth, 1e-9))

        total_latency   = exec_time + comm_time
        completion_time = self.current_time + total_latency
        sla_violated    = completion_time > task.deadline
        if sla_violated:
            self.sla_violations += 1

        # Energy (Eq.12, edge only)
        energy_consumed = 0.0
        if resource.resource_type == 'edge':
            energy_consumed = resource.power_consumption * exec_time / 1000.0  # → kWh

        # Reward (Algorithm 3)
        reward = self.reward_fn.compute(
            task=task,
            resource=resource,
            latency=total_latency,
            energy=energy_consumed,
            completion_time=completion_time,
            system_utilization=self.resource_manager.average_utilization(),
            avg_queue_length=self.resource_manager.average_queue_length(),
            all_resources=self.resource_manager.resources,
            queue_threshold=self._env_cfg['queue_threshold'],
        )

        self.completed_tasks.append({
            'task_id': task.task_id,
            'resource_id': resource.resource_id,
            'resource_type': resource.resource_type,
            'latency': total_latency,
            'energy_kwh': energy_consumed,
            'sla_violated': sla_violated,
            'completion_time': completion_time,
            'deadline': task.deadline,
            'priority': task.priority,
        })

        resource.deallocate(task)

        self.current_time += 1.0
        self.step_count   += 1
        self._arrive_tasks()

        info = {
            'latency': total_latency,
            'energy': energy_consumed,
            'sla_violated': sla_violated,
            'task_priority': task.priority,
            'resource_type': resource.resource_type,
        }
        return self._get_state(), reward, self._is_done(), info

    # ------------------------------------------------------------------
    # Action space helpers
    # ------------------------------------------------------------------
    def action_mask(self, task=None) -> np.ndarray:
        """
        Boolean mask: True = valid action (resource can accept current task).
        Used for Q-value masking in DQN agent (Section 4.4).
        """
        if task is None:
            if len(self.task_queue) == 0:
                return np.zeros(self.resource_manager.num_resources, dtype=bool)
            task = self.task_queue[0]
        return np.array(
            [r.can_accept(task) for r in self.resource_manager.resources],
            dtype=bool,
        )

    @property
    def n_actions(self) -> int:
        return self.resource_manager.num_resources

    @property
    def state_dim(self) -> int:
        return self.state_builder.state_dim

    # ------------------------------------------------------------------
    # Episode statistics
    # ------------------------------------------------------------------
    def get_episode_metrics(self) -> Dict[str, float]:
        """Return all performance metrics for the completed episode."""
        if not self.completed_tasks:
            return {}

        latencies     = [t['latency']     for t in self.completed_tasks]
        energies      = [t['energy_kwh']  for t in self.completed_tasks]
        sla_flags     = [t['sla_violated'] for t in self.completed_tasks]
        n_completed   = len(self.completed_tasks)
        n_arrived     = max(self.total_tasks_arrived, 1)

        # Task allocation accuracy: fraction assigned to resource with
        # enough capacity headroom (>20% free CPU after allocation)
        def _accurate(rec):
            res = self.resource_manager.resources[rec['resource_id']]
            return (res.cpu_available / max(res.cpu_capacity, 1e-9)) > 0.20

        alloc_acc = sum(1 for t in self.completed_tasks if _accurate(t)) / max(n_completed, 1)

        return {
            'avg_latency_ms':      float(np.mean(latencies)) * 1000,
            'energy_kwh':          float(np.sum(energies)),
            'sla_violation_rate':  float(np.mean(sla_flags)) * 100,
            'task_completion_rate': (n_completed / n_arrived) * 100,
            'task_allocation_accuracy': alloc_acc * 100,
            'n_completed':         n_completed,
            'n_sla_violated':      int(sum(sla_flags)),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _arrive_tasks(self):
        """Bring new tasks into the queue for the current timestep."""
        if self._task_list_mode:
            # Real-dataset mode: tasks arriving at or before current_time
            while self._remaining_tasks:
                task = self._remaining_tasks[0]
                if task.arrival_time <= self.current_time:
                    self.task_queue.append(self._remaining_tasks.pop(0))
                    self.total_tasks_arrived += 1
                else:
                    break
        else:
            # Synthetic mode: use pre-generated batches
            if self._batch_idx < len(self._episode_batches):
                for task in self._episode_batches[self._batch_idx]:
                    self.task_queue.append(task)
                    self.total_tasks_arrived += 1
                self._batch_idx += 1

    def _get_state(self) -> np.ndarray:
        """Build and return normalised state vector (Algorithm 2)."""
        tasks   = list(self.task_queue)
        latencies, bandwidths = self.resource_manager.get_network_conditions()
        return self.state_builder.build(
            task_queue=tasks,
            resources=self.resource_manager.resources,
            latencies=latencies,
            bandwidths=bandwidths,
            current_time=self.current_time,
        )

    def _is_done(self) -> bool:
        max_steps = self._wl_cfg['episode_max_steps']
        no_tasks  = (
            len(self.task_queue) == 0
            and (
                (self._task_list_mode and not self._remaining_tasks)
                or (not self._task_list_mode and self._batch_idx >= len(self._episode_batches))
            )
        )
        return self.step_count >= max_steps or no_tasks