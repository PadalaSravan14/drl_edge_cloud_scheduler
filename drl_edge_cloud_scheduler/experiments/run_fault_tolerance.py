import os
import sys
import yaml
import numpy as np
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_fault_tolerance(
    config_path: str = "config/config.yaml",
    model_path: str = "results/models/proposed_drl_final.pt",
    seed: int = 42,
):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    from environment.edge_cloud_env   import EdgeCloudEnv
    from environment.workload_generator import SyntheticWorkloadGenerator
    from agents.dqn_agent             import DQNAgent
    from evaluation.metrics           import MetricsAccumulator

    rng    = np.random.default_rng(seed)
    wl_gen = SyntheticWorkloadGenerator(config, rng=rng)

    # Build env + PCA
    env = EdgeCloudEnv(config, workload_generator=wl_gen, rng=rng)
    raw = []
    for _ in range(100):
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

    # Load agent
    agent = DQNAgent(config, state_dim, n_actions, seed=seed)
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"[fault_tolerance] Loaded model: {model_path}")
    else:
        print(f"[fault_tolerance] No pretrained model, using random weights")

    NUM_EPISODES = 10
    results      = {}

    print("\n[fault_tolerance] Scenario 1: No Failure")
    acc = MetricsAccumulator()
    for _ in range(NUM_EPISODES):
        env_i = EdgeCloudEnv(config, workload_generator=wl_gen,
                             rng=np.random.default_rng(seed + _))
        env_i.state_builder._pca        = env.state_builder._pca
        env_i.state_builder._pca_fitted = True
        env_i.state_builder._state_dim  = state_dim
        s = env_i.reset(inject_failures=False)
        done = False
        while not done:
            mask = env_i.action_mask()
            a    = agent.select_action(s, action_mask=mask, training=False)
            s, _, done, _ = env_i.step(a)
        acc.add(env_i.get_episode_metrics())
    summary = acc.summary()
    results['No Failure'] = {k: v['mean'] for k, v in summary.items()}


    print("[fault_tolerance] Scenario 2: Edge Node Failure")
    acc2 = MetricsAccumulator()
    for ep in range(NUM_EPISODES):
        env_i = EdgeCloudEnv(config, workload_generator=wl_gen,
                             rng=np.random.default_rng(seed + ep + 100))
        env_i.state_builder._pca        = env.state_builder._pca
        env_i.state_builder._pca_fitted = True
        env_i.state_builder._state_dim  = state_dim
        s    = env_i.reset(inject_failures=True)   # Bernoulli failure injection
        done = False
        while not done:
            mask = env_i.action_mask()
            a    = agent.select_action(s, action_mask=mask, training=False)
            s, _, done, _ = env_i.step(a)
        acc2.add(env_i.get_episode_metrics())
    summary2 = acc2.summary()
    results['Edge Node Failure'] = {k: v['mean'] for k, v in summary2.items()}

    print("[fault_tolerance] Scenario 3: Network Disruption")
    cfg_net = dict(config)
    cfg_net['environment'] = dict(config['environment'])
    cfg_net['environment']['network_latency_range'] = [10, 1000]  # 10× spike
    acc3 = MetricsAccumulator()
    for ep in range(NUM_EPISODES):
        env_i = EdgeCloudEnv(cfg_net, workload_generator=wl_gen,
                             rng=np.random.default_rng(seed + ep + 200))
        env_i.state_builder._pca        = env.state_builder._pca
        env_i.state_builder._pca_fitted = True
        env_i.state_builder._state_dim  = state_dim
        s    = env_i.reset()
        done = False
        while not done:
            mask = env_i.action_mask()
            a    = agent.select_action(s, action_mask=mask, training=False)
            s, _, done, _ = env_i.step(a)
        acc3.add(env_i.get_episode_metrics())
    summary3 = acc3.summary()
    results['Network Disruption'] = {k: v['mean'] for k, v in summary3.items()}

   
    print("\n Fault Tolerance Results:")
    print(f"{'Scenario':<25} {'Lat(ms)':>9} {'SLA(%)':>8} {'Compl(%)':>10} {'Energy(kWh)':>12}")
    print("-" * 68)
    for scenario, r in results.items():
        print(
            f"{scenario:<25} "
            f"{r.get('avg_latency_ms',0):>9.1f} "
            f"{r.get('sla_violation_rate',0):>8.1f} "
            f"{r.get('task_completion_rate',0):>10.1f} "
            f"{r.get('energy_kwh',0):>12.3f}"
        )

    os.makedirs("results", exist_ok=True)
    with open("results/fault_tolerance_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print("\n[fault_tolerance] Saved → results/fault_tolerance_results.json")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model",  default="results/models/proposed_drl_final.pt")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()
    run_fault_tolerance(args.config, args.model, args.seed)