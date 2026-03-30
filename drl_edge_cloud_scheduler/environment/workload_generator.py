import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Iterator


# Priority encoding (Section 4.3, Algorithm 2)
PRIORITY_LABELS = ['low', 'medium', 'high']
PRIORITY_MAP = {label: idx for idx, label in enumerate(PRIORITY_LABELS)}


@dataclass
class Task:

    task_id: int
    cpu_demand: float      # MI — million instructions
    mem_demand: float      # MB
    deadline: float        # absolute time in seconds
    priority: str          # 'low' | 'medium' | 'high'
    arrival_time: float    # seconds
    data_size: float = 0.1 # MB — used for communication time calc

    @property
    def priority_int(self) -> int:
        return PRIORITY_MAP.get(self.priority, 1)

    @property
    def priority_onehot(self) -> List[int]:
        """One-hot encoding for {low, medium, high} — Section 4.3."""
        v = [0, 0, 0]
        v[self.priority_int] = 1
        return v

    def remaining_deadline(self, current_time: float) -> float:
        return max(0.0, self.deadline - current_time)


# ===========================================================================
# 1. Synthetic Workload Generator 
# ===========================================================================
class SyntheticWorkloadGenerator:


    def __init__(self, config: dict, rng: Optional[np.random.Generator] = None):
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self._env = config['environment']
        self._wl = config['workload']
        self._task_counter = 0

    def reset_counter(self):
        self._task_counter = 0

    # ------------------------------------------------------------------
    # Single task sampling
    # ------------------------------------------------------------------
    def _sample_task(self, arrival_time: float) -> Task:
        cpu = float(self.rng.uniform(*self._env['task_cpu_range']))
        mem = float(self.rng.uniform(*self._env['task_mem_range']))
        dl_offset = float(self.rng.uniform(*self._env['task_deadline_range']))
        priority = str(self.rng.choice(PRIORITY_LABELS))
        task = Task(
            task_id=self._task_counter,
            cpu_demand=cpu,
            mem_demand=mem,
            deadline=arrival_time + dl_offset,
            priority=priority,
            arrival_time=arrival_time,
            data_size=cpu * 0.001,
        )
        self._task_counter += 1
        return task

    # ------------------------------------------------------------------
    # Batch generation (one timestep)
    # ------------------------------------------------------------------
    def generate_batch(
        self,
        current_time: float,
        arrival_rate: float,
        num_tasks: Optional[int] = None,
    ) -> List[Task]:
        """
        Generate tasks at a single timestep.
        num_tasks drawn from Poisson(arrival_rate) if not specified.
        """
        if num_tasks is None:
            num_tasks = int(self.rng.poisson(max(arrival_rate, 0.01)))
        return [self._sample_task(current_time) for _ in range(num_tasks)]

    # ------------------------------------------------------------------
    # Episode generation
    # ------------------------------------------------------------------
    def generate_episode(
        self,
        num_tasks: int,
        burst_at_step: Optional[int] = None,
        burst_multiplier: float = 2.0,
    ) -> List[List[Task]]:

        base_rate = float(
            self.rng.uniform(*self._wl['arrival_rate_range'])
        ) / 60.0   # tasks/sec

        batches: List[List[Task]] = []
        total = 0
        step = 0
        t = 0.0

        while total < num_tasks:
            rate = (
                base_rate * burst_multiplier
                if burst_at_step is not None and step >= burst_at_step
                else base_rate
            )
            batch = self.generate_batch(t, rate)
            remaining = num_tasks - total
            batch = batch[:remaining]
            batches.append(batch)
            total += len(batch)
            t += 1.0
            step += 1

        return batches


# ===========================================================================
# 2. Google Cluster Traces Loader 
# ===========================================================================
class GoogleClusterLoader:

    def __init__(
        self,
        filepath: str,
        config: dict,
        rng: Optional[np.random.Generator] = None,
        max_tasks: int = 100_000,
    ):
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.max_tasks = max_tasks
        self.df = self._load(filepath)
        self._task_counter = 0

    def _load(self, filepath: str) -> pd.DataFrame:
        usecols = ['time', 'cpu_request', 'mem_request', 'priority']
        df = pd.read_csv(filepath, nrows=self.max_tasks)

        # Tolerate column-name variations in raw traces
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
        available = [c for c in usecols if c in df.columns]
        df = df[available].dropna().sort_values('time').reset_index(drop=True)

        env = self.config['environment']

        # Scale CPU to paper's MI range [1000, 10000]
        cpu_max = df['cpu_request'].max() if 'cpu_request' in df else 1.0
        df['cpu_mi'] = (df['cpu_request'] / max(cpu_max, 1e-9)) * 9000 + 1000

        # Scale memory to MB range [100, 512]
        mem_max = df['mem_request'].max() if 'mem_request' in df else 1.0
        df['mem_mb'] = (df['mem_request'] / max(mem_max, 1e-9)) * 412 + 100

        # Map Google priority [0-11] → {low, medium, high}
        if 'priority' in df.columns:
            df['pri_label'] = pd.cut(
                df['priority'],
                bins=[-1, 3, 7, 11],
                labels=['low', 'medium', 'high'],
            ).astype(str)
        else:
            df['pri_label'] = 'medium'

        # Convert microseconds → seconds (relative)
        t0 = df['time'].min()
        df['arrival_s'] = (df['time'] - t0) / 1_000_000.0

        return df

    def get_all_tasks(self, max_tasks: Optional[int] = None) -> List[Task]:
        limit = min(max_tasks or len(self.df), len(self.df))
        tasks = []
        for _, row in self.df.head(limit).iterrows():
            dl_offset = float(self.rng.uniform(1, 10))
            t = Task(
                task_id=self._task_counter,
                cpu_demand=float(row['cpu_mi']),
                mem_demand=float(row['mem_mb']),
                deadline=float(row['arrival_s']) + dl_offset,
                priority=str(row['pri_label']),
                arrival_time=float(row['arrival_s']),
                data_size=float(row['cpu_mi']) * 0.001,
            )
            tasks.append(t)
            self._task_counter += 1
        return tasks

    def iter_batches(self, batch_size: int = 10) -> Iterator[List[Task]]:
        """Yield batches in arrival order for step-by-step simulation."""
        for start in range(0, len(self.df), batch_size):
            chunk = self.df.iloc[start : start + batch_size]
            batch = []
            for _, row in chunk.iterrows():
                dl_offset = float(self.rng.uniform(1, 10))
                t = Task(
                    task_id=self._task_counter,
                    cpu_demand=float(row['cpu_mi']),
                    mem_demand=float(row['mem_mb']),
                    deadline=float(row['arrival_s']) + dl_offset,
                    priority=str(row['pri_label']),
                    arrival_time=float(row['arrival_s']),
                    data_size=float(row['cpu_mi']) * 0.001,
                )
                batch.append(t)
                self._task_counter += 1
            yield batch


# ===========================================================================
# 3. Azure Functions Loader 
# ===========================================================================
class AzureFunctionsLoader:

    def __init__(
        self,
        filepath: str,
        config: dict,
        rng: Optional[np.random.Generator] = None,
        max_tasks: int = 50_000,
    ):
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.max_tasks = max_tasks
        self.df = self._load(filepath)
        self._task_counter = 0

    def _load(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath, nrows=self.max_tasks)
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
        df = df.dropna().reset_index(drop=True)

        cols = list(df.columns)
        time_col = cols[0]

        # Normalise arrival time to seconds
        t0 = df[time_col].min()
        df['arrival_s'] = (df[time_col] - t0).astype(float)

        # CPU demand from duration
        if len(cols) > 1:
            dur_max = df[cols[1]].max()
            df['cpu_mi'] = (df[cols[1]] / max(dur_max, 1e-9)) * 9000 + 1000
        else:
            df['cpu_mi'] = self.rng.uniform(1000, 10000, size=len(df))

        # Memory
        if len(cols) > 2:
            mem_max = df[cols[2]].max()
            df['mem_mb'] = (df[cols[2]] / max(mem_max, 1e-9)) * 412 + 100
        else:
            df['mem_mb'] = self.rng.uniform(100, 512, size=len(df))

        # Azure serverless = mostly time-critical → high priority
        df['pri_label'] = self.rng.choice(
            ['medium', 'high', 'high'], size=len(df)
        )

        return df

    def get_all_tasks(self, max_tasks: Optional[int] = None) -> List[Task]:
        limit = min(max_tasks or len(self.df), len(self.df))
        tasks = []
        for _, row in self.df.head(limit).iterrows():
            # Tighter deadlines for serverless (1-5s)
            dl_offset = float(self.rng.uniform(1, 5))
            t = Task(
                task_id=self._task_counter,
                cpu_demand=float(row['cpu_mi']),
                mem_demand=float(row['mem_mb']),
                deadline=float(row['arrival_s']) + dl_offset,
                priority=str(row['pri_label']),
                arrival_time=float(row['arrival_s']),
                data_size=float(row['cpu_mi']) * 0.001,
            )
            tasks.append(t)
            self._task_counter += 1
        return tasks
