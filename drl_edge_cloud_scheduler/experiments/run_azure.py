import os
import sys
import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_azure(
    config_path: str = "config/config.yaml",
    data_path: str = "data/processed/azure_processed.csv",
    model_path: str = "results/models/proposed_drl_final.pt",
    seed: int = 42,
):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    from environment.edge_cloud_env   import EdgeCloudEnv
    from environment.workload_generator import AzureFunctionsLoader, SyntheticWorkloadGenerator
    from agents.dqn_agent             import DQNAgent
    from agents.ddqn_agent            import DDQNAgent
    from agents.ppo_agent             import PPOAgent
    from baselines.fifo               import FIFOScheduler
    from baselines.round_robin        import RoundRobinScheduler
    from evaluation.evaluator         import Evaluator

    print(f"[run_azure] Loading dataset: {data_path}")
    loader = AzureFunctionsLoader(data_path, config, max_tasks=50_000)
    tasks  = loader.get_all_tasks(max_tasks=5_000)
    print(f"[run_azure] Loaded {len(tasks):,} tasks")

    wl_gen = SyntheticWorkloadGenerator(config, rng=np.random.default_rng(seed))
    env    = EdgeCloudEnv(config, workload_generator=wl_gen, rng=np.random.default_rng(seed))

    raw_states = []
    for _ in range(100):
        s = env.reset()
        raw_states.append(s)
        for _ in range(5):
            mask  = env.action_mask()
            valid = np.where(mask)[0]
            a     = int(np.random.default_rng(seed).choice(valid)) if len(valid) > 0 else 0
            s, _, done, _ = env.step(a)
            raw_states.append(s)
            if done:
                break
    env.state_builder.fit_pca(np.array(raw_states))
    state_dim = env.state_dim
    n_actions = env.n_actions

    proposed = DQNAgent(config, state_dim, n_actions, seed=seed)
    if os.path.exists(model_path):
        proposed.load(model_path)
    ddqn = DDQNAgent(config, state_dim, n_actions, seed=seed)
    ppo  = PPOAgent(config, state_dim, n_actions, seed=seed)

    agents = {
        "Proposed DRL": proposed,
        "DDQN":         ddqn,
        "PPO":          ppo,
        "FIFO":         FIFOScheduler(config),
        "Round-Robin":  RoundRobinScheduler(config),
    }

    eval_env = EdgeCloudEnv(config, workload_generator=wl_gen,
                            rng=np.random.default_rng(seed + 200))
    eval_env.state_builder._pca        = env.state_builder._pca
    eval_env.state_builder._pca_fitted = env.state_builder._pca_fitted
    eval_env.state_builder._state_dim  = state_dim

    evaluator = Evaluator(config, eval_env)
    results = evaluator.evaluate_all(
        agents,
        num_episodes=5,
        task_list=tasks,
        save_path="results/azure_results.json",
    )
    print("\n[run_azure] Done.")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data",   default="data/processed/azure_processed.csv")
    parser.add_argument("--model",  default="results/models/proposed_drl_final.pt")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()
    run_azure(args.config, args.data, args.model, args.seed)