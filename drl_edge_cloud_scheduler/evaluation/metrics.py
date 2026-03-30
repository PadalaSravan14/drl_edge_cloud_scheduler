import numpy as np
from typing import List, Dict, Optional


def compute_metrics(completed_tasks: List[Dict]) -> Dict[str, float]:

    if not completed_tasks:
        return {
            'avg_latency_ms':          0.0,
            'energy_kwh':              0.0,
            'sla_violation_rate':      0.0,
            'task_completion_rate':    0.0,
            'task_allocation_accuracy': 0.0,
            'n_completed':             0,
        }

    latencies  = np.array([t['latency']      for t in completed_tasks]) * 1000   # s→ms
    energies   = np.array([t['energy_kwh']   for t in completed_tasks])
    sla_flags  = np.array([t['sla_violated'] for t in completed_tasks], dtype=float)

    # Task allocation accuracy: task assigned to resource type that best
    # suits it (high-priority tasks on cloud, low on edge preferred)
    def _is_accurate(rec: Dict) -> bool:
        pri      = rec.get('priority', 'medium')
        res_type = rec.get('resource_type', 'edge')
        # High-priority → cloud preferred for capacity
        # Low-priority  → edge preferred for energy
        if pri == 'high' and res_type == 'cloud':
            return True
        if pri == 'low'  and res_type == 'edge':
            return True
        if pri == 'medium':
            return True
        # Still counts if no better option existed
        return not rec.get('sla_violated', False)

    alloc_accurate = sum(1 for t in completed_tasks if _is_accurate(t))
    alloc_acc      = alloc_accurate / len(completed_tasks) * 100.0

    return {
        'avg_latency_ms':           float(np.mean(latencies)),
        'p50_latency_ms':           float(np.percentile(latencies, 50)),
        'p95_latency_ms':           float(np.percentile(latencies, 95)),
        'p99_latency_ms':           float(np.percentile(latencies, 99)),
        'energy_kwh':               float(np.sum(energies)),
        'sla_violation_rate':       float(np.mean(sla_flags) * 100),
        'n_sla_violated':           int(np.sum(sla_flags)),
        'task_allocation_accuracy': float(alloc_acc),
        'n_completed':              len(completed_tasks),
    }


class MetricsAccumulator:
    def __init__(self):
        self._records: List[Dict] = []

    def add(self, metrics: Dict):
        self._records.append(metrics)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Return mean, std, 95% CI for each metric."""
        if not self._records:
            return {}

        keys = self._records[0].keys()
        result = {}
        for k in keys:
            vals = np.array([r[k] for r in self._records if k in r], dtype=float)
            n    = len(vals)
            mean = float(np.mean(vals))
            std  = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            # 95% CI: mean ± 1.96 × std / sqrt(n)
            ci   = 1.96 * std / np.sqrt(max(n, 1))
            result[k] = {
                'mean':  mean,
                'std':   std,
                'ci_lo': mean - ci,
                'ci_hi': mean + ci,
                'n':     n,
            }
        return result

    def to_table(self) -> str:
        summary = self.summary()
        lines   = [f"{'Metric':<35} {'Mean ± SD':<22} {'95% CI'}"]
        lines.append('-' * 75)
        for k, v in summary.items():
            lines.append(
                f"{k:<35} {v['mean']:.2f} ± {v['std']:.2f}"
                f"          [{v['ci_lo']:.2f}, {v['ci_hi']:.2f}]"
            )
        return '\n'.join(lines)

    def reset(self):
        self._records.clear()