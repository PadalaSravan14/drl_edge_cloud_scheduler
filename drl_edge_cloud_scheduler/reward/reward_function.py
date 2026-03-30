from typing import List
class RewardFunction:

    def __init__(self, config: dict):
        self.cfg = config['reward']
        bw = self.cfg['base_weights']
        self.base_lambda = [
            bw['latency'],   # λ₁
            bw['energy'],    # λ₂
            bw['sla'],       # λ₃
            bw['overload'],  # λ₄
        ]
    def compute(
        self,
        task,
        resource,
        latency: float,
        energy: float,
        completion_time: float,
        system_utilization: float,
        avg_queue_length: float,
        all_resources: list,
        queue_threshold: float,
    ) -> float:

        latency_cost = latency  # seconds

        energy_cost = energy  # kWh

        if completion_time > task.deadline:
            relative_violation = (completion_time - task.deadline) / max(task.deadline, 1e-9)
            sla_cost = 1.0 + relative_violation
        else:
            sla_cost = 0.0


        overload_cost = 0.0
        for r in all_resources:
            if not r.failed and queue_threshold > 0:
                ratio = r.queue_length / queue_threshold
                if ratio > 1.0:
                    overload_cost += (ratio - 1.0) ** 2


        lam = list(self.base_lambda)  # copy base weights

        peak_load = (
            system_utilization > self.cfg['peak_load_threshold']
            or avg_queue_length > self.cfg['peak_queue_threshold']
        )
        off_peak = system_utilization < self.cfg['off_peak_threshold']


        if peak_load:
            lam[2] *= self.cfg['peak_sla_boost']      # λ₃ × 1.40
            lam[3] *= self.cfg['peak_overload_boost']  # λ₄ × 1.40

  
        if off_peak:
            lam[1] *= self.cfg['offpeak_energy_boost']  # λ₂ × 1.30

        if task.priority == 'high':
            lam[0] *= self.cfg['highpriority_latency_boost']  # λ₁ × 1.35
            lam[2] *= self.cfg['highpriority_sla_boost']      # λ₃ × 1.35


        weight_sum = sum(lam)
        if weight_sum > 0:
            lam = [w / weight_sum for w in lam]


        reward = -(
            lam[0] * latency_cost
            + lam[1] * energy_cost
            + lam[2] * sla_cost
            + lam[3] * overload_cost
        )

        return float(reward)


    def get_weights(
        self,
        system_utilization: float,
        avg_queue_length: float,
        task_priority: str,
        queue_threshold: float,
    ) -> List[float]:
        """Return normalised adaptive weights without computing reward."""
        lam = list(self.base_lambda)

        peak_load = (
            system_utilization > self.cfg['peak_load_threshold']
            or avg_queue_length > self.cfg['peak_queue_threshold']
        )
        off_peak = system_utilization < self.cfg['off_peak_threshold']

        if peak_load:
            lam[2] *= self.cfg['peak_sla_boost']
            lam[3] *= self.cfg['peak_overload_boost']
        if off_peak:
            lam[1] *= self.cfg['offpeak_energy_boost']
        if task_priority == 'high':
            lam[0] *= self.cfg['highpriority_latency_boost']
            lam[2] *= self.cfg['highpriority_sla_boost']

        weight_sum = sum(lam)
        return [w / weight_sum for w in lam] if weight_sum > 0 else lam