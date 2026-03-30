import os
import sys
import json
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_synthetic(config_path: str = "config/config.yaml", seed: int = 42):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    rng = np.random.default_rng(seed)

    from environment.edge_cloud_env import EdgeCloudEnv
    from environment.workload_generator import SyntheticWorkloadGenerator
    from agents.dqn_agent   import DQNAgent
    from agents.ddqn_agent  import DDQNAgent
    from agents.ppo_agent   import PPOAgent
    from baselines.fifo         import FIFOScheduler
    from baselines.round_robin  import RoundRobinScheduler
    from baselines.min_min      import MinMinScheduler
    from baselines.greedy_energy import GreedyEnergyScheduler
    from state_representation.state_builder import StateBuilder
    from training.trainer   import Trainer
    from evaluation.evaluator import Evaluator
    from evaluation.statistical_tests import compare_all_baselines


    wl_gen = SyntheticWorkloadGenerator(config, rng=rng)
    env    = EdgeCloudEnv(config, workload_generator=wl_gen, rng=rng)

    # Warm-up to fit PCA
    print("[run_synthetic] Warming up environment for PCA fitting ...")
    raw_states = []
    for _ in range(200):
        s = env.reset()
        raw_states.append(s)
        for _ in range(10):
            mask = env.action_mask()
            valid = np.where(mask)[0]
            a = int(rng.choice(valid)) if len(valid) > 0 else 0
            s, _, done, _ = env.step(a)
            raw_states.append(s)
            if done:
                break

    env.state_builder.fit_pca(np.array(raw_states))
    state_dim = env.state_dim
    n_actions = env.n_actions
    print(f"[run_synthetic] state_dim={state_dim}, n_actions={n_actions}")

    proposed_drl = DQNAgent(config, state_dim, n_actions, seed=seed)
    ddqn_agent   = DDQNAgent(config, state_dim, n_actions, seed=seed)
    ppo_agent    = PPOAgent(config, state_dim, n_actions, seed=seed)


    print("\n[run_synthetic] Training Proposed AW-DQN ...")
    trainer = Trainer(config, proposed_drl, env, agent_name="proposed_drl")
    history = trainer.train(num_episodes=config['training']['num_episodes'])


    print("\n[run_synthetic] Training DDQN baseline ...")
    env2     = EdgeCloudEnv(config, workload_generator=wl_gen, rng=np.random.default_rng(seed+1))
    env2.state_builder._pca         = env.state_builder._pca
    env2.state_builder._pca_fitted  = env.state_builder._pca_fitted
    env2.state_builder._state_dim   = state_dim
    trainer2 = Trainer(config, ddqn_agent, env2, agent_name="ddqn")
    trainer2.train(num_episodes=min(1000, config['training']['num_episodes']))

    print("\n[run_synthetic] Training PPO baseline ...")
    env3 = EdgeCloudEnv(config, workload_generator=wl_gen, rng=np.random.default_rng(seed+2))
    env3.state_builder._pca         = env.state_builder._pca
    env3.state_builder._pca_fitted  = env.state_builder._pca_fitted
    env3.state_builder._state_dim   = state_dim
    trainer3 = Trainer(config, ppo_agent, env3, agent_name="ppo")
    trainer3.train(num_episodes=min(1000, config['training']['num_episodes']))

    fifo_sched    = FIFOScheduler(config)
    rr_sched      = RoundRobinScheduler(config)
    minmin_sched  = MinMinScheduler(config)
    greedy_sched  = GreedyEnergyScheduler(config)

    agents = {
        "Proposed DRL": proposed_drl,
        "DDQN":         ddqn_agent,
        "PPO":          ppo_agent,
        "FIFO":         fifo_sched,
        "Round-Robin":  rr_sched,
        "Min-Min":      minmin_sched,
        "Greedy-Energy":greedy_sched,
    }

    eval_env = EdgeCloudEnv(config, workload_generator=wl_gen, rng=np.random.default_rng(999))
    eval_env.state_builder._pca        = env.state_builder._pca
    eval_env.state_builder._pca_fitted = env.state_builder._pca_fitted
    eval_env.state_builder._state_dim  = state_dim

    evaluator  = Evaluator(config, eval_env)
    all_results = evaluator.evaluate_all(
        agents,
        num_episodes=config['evaluation']['num_test_episodes'],
        save_path="results/synthetic_results.json",
    )

    print("\n[run_synthetic] Statistical validation ...")
    seeds = config['evaluation']['random_seeds']
    stat_results = evaluator.evaluate_statistical(
        proposed_drl, "Proposed DRL", seeds=seeds,
        num_episodes=config['evaluation']['num_test_episodes'],
    )

    print("\n[run_synthetic] Done. Results in results/")
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()
    run_synthetic(args.config, args.seed)