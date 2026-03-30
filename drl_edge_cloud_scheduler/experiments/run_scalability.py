import os
import sys
import yaml
import numpy as np
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_scalability(
    config_path: str = "config/config.yaml",
    model_path: str  = "results/models/proposed_drl_final.pt",
    seed: int = 42,
):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    from environment.edge_cloud_env     import EdgeCloudEnv
    from environment.workload_generator import SyntheticWorkloadGenerator
    from agents.dqn_agent               import DQNAgent
    from agents.ddqn_agent              import DDQNAgent
    from agents.ppo_agent               import PPOAgent
    from baselines.fifo                 import FIFOScheduler
    from evaluation.metrics             import MetricsAccumulator

    rng    = np.random.default_rng(seed)
    wl_gen = SyntheticWorkloadGenerator(config, rng=rng)

    env = EdgeCloudEnv(config, workload_generator=wl_gen, rng=rng)
    raw = []
    for _ in range(80):
        s = env.reset()
        raw.append(s)
        for _ in range(5):
            mask  = env.action_mask()
            valid = np.where(mask)[0]
            a     = int(rng.choice(valid)) if len(valid) > 0 else 0
            s, _, done, _ = env.step(a)
            raw.append(s)
            if done:
                break
    env.state_builder.fit_pca(np.array(raw))
    state_dim = env.state_dim
    n_actions = env.n_actions

    def make_eval_env(s):
        e = EdgeCloudEnv(config, workload_generator=wl_gen, rng=np.random.default_rng(s))
        e.state_builder._pca        = env.state_builder._pca
        e.state_builder._pca_fitted = True
        e.state_builder._state_dim  = state_dim
        return e

    # Load agents
    proposed = DQNAgent(config, state_dim, n_actions, seed=seed)
    if os.path.exists(model_path):
        proposed.load(model_path)
    ddqn = DDQNAgent(config, state_dim, n_actions, seed=seed)
    ppo  = PPOAgent(config,  state_dim, n_actions, seed=seed)
    fifo = FIFOScheduler(config)

    all_agents = {
        "Proposed DRL": proposed,
        "DDQN":         ddqn,
        "PPO":          ppo,
        "FIFO":         fifo,
    }

    # ------------------------------------------------------------------
    # 1. Energy vs workload size ( 200, 500, 1000 tasks)
    # ------------------------------------------------------------------
    print("\n[scalability] Energy vs. workload size (Table 8) ...")
    task_loads = [200, 500, 1000]
    energy_results = {name: {} for name in all_agents}

    for n_tasks in task_loads:
        cfg_t = dict(config)
        cfg_t['workload'] = dict(config['workload'])
        cfg_t['workload']['synthetic_num_tasks'] = n_tasks

        for name, agent in all_agents.items():
            acc = MetricsAccumulator()
            for ep in range(5):
                e = make_eval_env(seed + ep + n_tasks)
                e.config = cfg_t
                e._wl_cfg = cfg_t['workload']
                s    = e.reset()
                done = False
                while not done:
                    mask = e.action_mask()
                    a    = agent.select_action(s, action_mask=mask, training=False)
                    s, _, done, _ = e.step(a)
                acc.add(e.get_episode_metrics())
            summary = acc.summary()
            energy_results[name][n_tasks] = summary.get('energy_kwh', {}).get('mean', 0.0)

    print("\n  Energy Consumption (kWh) — Table 8:")
    header = f"{'Method':<20}" + ''.join(f"{n:>10}" for n in task_loads)
    print("  " + header)
    print("  " + "-" * (20 + 10 * len(task_loads)))
    for name in all_agents:
        row = f"  {name:<20}" + ''.join(
            f"{energy_results[name].get(n, 0):>10.1f}" for n in task_loads
        )
        print(row)

    # ------------------------------------------------------------------
    # 2. Adaptation speed under burst
    # steps to stabilise after burst at step 500
    # ------------------------------------------------------------------
    print("\n[scalability] Adaptation speed under burst ...")
    burst_cfg = dict(config)
    burst_cfg['workload'] = dict(config['workload'])
    burst_cfg['workload']['burst_step']       = 500
    burst_cfg['workload']['burst_multiplier'] = 2.0
    burst_cfg['workload']['synthetic_num_tasks'] = 1200

    adaptation_steps = {}
    for name, agent in all_agents.items():
        step_sla_violations = []
        for ep in range(3):
            e = EdgeCloudEnv(
                burst_cfg,
                workload_generator=SyntheticWorkloadGenerator(burst_cfg, rng=np.random.default_rng(seed + ep)),
                rng=np.random.default_rng(seed + ep)
            )
            e.state_builder._pca        = env.state_builder._pca
            e.state_builder._pca_fitted = True
            e.state_builder._state_dim  = state_dim
            s = e.reset()
            step_sla = []
            while not e._is_done():
                mask = e.action_mask()
                a    = agent.select_action(s, action_mask=mask, training=False)
                s, _, done, info = e.step(a)
                step_sla.append(1 if info.get('sla_violated', False) else 0)
                if done:
                    break
            step_sla_violations.append(step_sla)

        # Adaptation = steps after burst until SLA rate drops to < 15%
        if step_sla_violations:
            avg_sla = np.mean(step_sla_violations, axis=0)
            burst_start = burst_cfg['workload']['burst_step']
            stabilised  = None
            window = 10
            for i in range(burst_start, len(avg_sla) - window):
                if np.mean(avg_sla[i:i+window]) < 0.15:
                    stabilised = i - burst_start
                    break
            adaptation_steps[name] = stabilised if stabilised else len(avg_sla) - burst_start

    print("\n  Steps to stabilise after burst:")
    for name, steps in adaptation_steps.items():
        print(f"    {name:<20}: {steps} steps")
    

    # Save
    scalability_results = {
        'energy_vs_tasks':   energy_results,
        'adaptation_steps':  adaptation_steps,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/scalability_results.json", 'w') as f:
        json.dump(scalability_results, f, indent=2)
    print("\n[scalability] Saved → results/scalability_results.json")
    return scalability_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model",  default="results/models/proposed_drl_final.pt")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()
    run_scalability(args.config, args.model, args.seed)