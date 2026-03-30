import numpy as np
import os
import json
from typing import Dict, List, Optional

from evaluation.metrics import compute_metrics, MetricsAccumulator


class Evaluator:

    def __init__(self, config: dict, env):
        self.config = config
        self.env    = env
        self.eval_cfg = config['evaluation']

    def evaluate_agent(
        self,
        agent,
        agent_name: str,
        num_episodes: int = 10,
        task_list=None,
        inject_failures: bool = False,
        verbose: bool = True,
    ) -> Dict[str, float]:

        accumulator = MetricsAccumulator()

        for ep in range(num_episodes):
            state = self.env.reset(
                task_list=task_list,
                inject_failures=inject_failures,
            )
            done = False
            while not done:
                mask   = self.env.action_mask()
                action = agent.select_action(state, action_mask=mask, training=False)

                # Inject resource pointers for heuristic baselines
                if hasattr(agent, 'set_resources'):
                    agent.set_resources(self.env.resource_manager.resources)

                # For Min-Min and Greedy, pass current task
                if hasattr(agent, 'resources') and len(self.env.task_queue) > 0:
                    task   = self.env.task_queue[0]
                    action = agent.select_action(
                        state, action_mask=mask, training=False, task=task
                    )

                state, _, done, _ = self.env.step(action)

            metrics = self.env.get_episode_metrics()
            accumulator.add(metrics)

        summary = accumulator.summary()
        result  = {k: v['mean'] for k, v in summary.items()}

        if verbose:
            print(f"\n  [{agent_name}]")
            print(f"    Avg Latency:       {result.get('avg_latency_ms', 0):.1f} ms")
            print(f"    SLA Violation:     {result.get('sla_violation_rate', 0):.1f}%")
            print(f"    Task Completion:   {result.get('task_completion_rate', 0):.1f}%")
            print(f"    Energy:            {result.get('energy_kwh', 0):.1f} kWh")
            print(f"    Alloc Accuracy:    {result.get('task_allocation_accuracy', 0):.1f}%")

        return result

    def evaluate_all(
        self,
        agents: Dict,
        num_episodes: int = 10,
        task_list=None,
        inject_failures: bool = False,
        save_path: Optional[str] = None,
    ) -> Dict[str, Dict]:

        print("\n" + "=" * 60)
        print("  Evaluation Results")
        print("=" * 60)

        all_results = {}
        for name, agent in agents.items():
            result = self.evaluate_agent(
                agent, name, num_episodes, task_list, inject_failures
            )
            all_results[name] = result

        self._print_table(all_results)

        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\n  Results saved → {save_path}")

        return all_results

    def evaluate_statistical(
        self,
        agent,
        agent_name: str,
        seeds: List[int] = None,
        num_episodes: int = 10,
        task_list=None,
    ) -> Dict:

        seeds = seeds or self.eval_cfg.get('random_seeds', [42, 123, 789])
        accumulator = MetricsAccumulator()

        for seed in seeds:
            self.env.rng = np.random.default_rng(seed)
            result = self.evaluate_agent(
                agent, agent_name, num_episodes, task_list, verbose=False
            )
            accumulator.add(result)

        print(f"\n[Statistical Evaluation] {agent_name}:")
        print(accumulator.to_table())

        return accumulator.summary()

    def _print_table(self, results: Dict[str, Dict]):
        keys = [
            'avg_latency_ms', 'sla_violation_rate',
            'task_completion_rate', 'energy_kwh', 'task_allocation_accuracy',
        ]
        headers = ['Method', 'Lat(ms)', 'SLA(%)', 'Compl(%)', 'Energy(kWh)', 'Acc(%)']

        col_w = [20, 10, 10, 10, 13, 8]
        header_str = ''.join(h.ljust(w) for h, w in zip(headers, col_w))
        print('\n  ' + header_str)
        print('  ' + '-' * sum(col_w))
        for name, r in results.items():
            row = [
                name,
                f"{r.get('avg_latency_ms', 0):.1f}",
                f"{r.get('sla_violation_rate', 0):.1f}",
                f"{r.get('task_completion_rate', 0):.1f}",
                f"{r.get('energy_kwh', 0):.1f}",
                f"{r.get('task_allocation_accuracy', 0):.1f}",
            ]
            print('  ' + ''.join(v.ljust(w) for v, w in zip(row, col_w)))