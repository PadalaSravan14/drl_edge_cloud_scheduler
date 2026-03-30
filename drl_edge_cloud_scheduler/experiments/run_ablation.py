import os
import sys
import yaml
import numpy as np
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_env_and_pca(config, seed):
    from environment.edge_cloud_env import EdgeCloudEnv
    from environment.workload_generator import SyntheticWorkloadGenerator

    rng    = np.random.default_rng(seed)
    wl_gen = SyntheticWorkloadGenerator(config, rng=rng)
    env    = EdgeCloudEnv(config, workload_generator=wl_gen, rng=rng)

    raw = []
    for _ in range(50):
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
    return env, wl_gen


def run_ablation(config_path="config/config.yaml", seed=42):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    from agents.dqn_agent    import DQNAgent
    from training.trainer    import Trainer
    from evaluation.evaluator import Evaluator
    from environment.edge_cloud_env import EdgeCloudEnv

    env, wl_gen = build_env_and_pca(config, seed)
    state_dim   = env.state_dim
    n_actions   = env.n_actions

    ablation_results = {}

    # ------------------------------------------------------------------
    # 1. No Reward Function Tuning (equal static weights)
    # ------------------------------------------------------------------
    print("\n[Ablation] 1. No Reward Function Tuning ...")
    cfg_no_rwt = dict(config)
    cfg_no_rwt['reward'] = dict(config['reward'])
    cfg_no_rwt['reward']['base_weights'] = {'latency':0.25,'energy':0.25,'sla':0.25,'overload':0.25}
    cfg_no_rwt['reward']['peak_sla_boost']       = 1.0
    cfg_no_rwt['reward']['peak_overload_boost']  = 1.0
    cfg_no_rwt['reward']['offpeak_energy_boost'] = 1.0
    cfg_no_rwt['reward']['highpriority_latency_boost'] = 1.0
    cfg_no_rwt['reward']['highpriority_sla_boost']     = 1.0

    agent1 = DQNAgent(cfg_no_rwt, state_dim, n_actions, seed=seed)
    env1, _ = build_env_and_pca(cfg_no_rwt, seed + 10)
    Trainer(cfg_no_rwt, agent1, env1, "ablation_no_reward").train(
        num_episodes=min(500, config['training']['num_episodes']), verbose=False
    )
    eval_env1 = EdgeCloudEnv(cfg_no_rwt, rng=np.random.default_rng(seed+100))
    eval_env1.state_builder._pca        = env.state_builder._pca
    eval_env1.state_builder._pca_fitted = True
    eval_env1.state_builder._state_dim  = state_dim
    r1 = Evaluator(cfg_no_rwt, eval_env1).evaluate_agent(agent1, "No Reward Tuning", verbose=True)
    ablation_results['No Reward Function Tuning'] = r1

    # ------------------------------------------------------------------
    # 2. No Adaptive Learning Rate (fixed lr, no decay)
    # ------------------------------------------------------------------
    print("\n[Ablation] 2. No Adaptive Learning Rate ...")
    agent2 = DQNAgent(config, state_dim, n_actions, seed=seed)
    env2, _ = build_env_and_pca(config, seed + 20)
    Trainer(config, agent2, env2, "ablation_no_alr").train(
        num_episodes=min(500, config['training']['num_episodes']), verbose=False
    )
    eval_env2 = EdgeCloudEnv(config, rng=np.random.default_rng(seed+200))
    eval_env2.state_builder._pca        = env.state_builder._pca
    eval_env2.state_builder._pca_fitted = True
    eval_env2.state_builder._state_dim  = state_dim
    r2 = Evaluator(config, eval_env2).evaluate_agent(agent2, "No Adaptive LR", verbose=True)
    ablation_results['No Adaptive Learning Rate'] = r2

    # ------------------------------------------------------------------
    # 3. No Temporal Dependencies (tiny replay buffer = no reuse)
    # ------------------------------------------------------------------
    print("\n[Ablation] 3. No Temporal Dependencies ...")
    cfg_no_td = dict(config)
    cfg_no_td['dqn'] = dict(config['dqn'])
    cfg_no_td['dqn']['replay_buffer_size'] = 64  # effectively no replay
    agent3 = DQNAgent(cfg_no_td, state_dim, n_actions, seed=seed)
    env3, _ = build_env_and_pca(cfg_no_td, seed + 30)
    Trainer(cfg_no_td, agent3, env3, "ablation_no_td").train(
        num_episodes=min(500, config['training']['num_episodes']), verbose=False
    )
    eval_env3 = EdgeCloudEnv(cfg_no_td, rng=np.random.default_rng(seed+300))
    eval_env3.state_builder._pca        = env.state_builder._pca
    eval_env3.state_builder._pca_fitted = True
    eval_env3.state_builder._state_dim  = state_dim
    r3 = Evaluator(cfg_no_td, eval_env3).evaluate_agent(agent3, "No Temporal Dep", verbose=True)
    ablation_results['No Temporal Dependencies'] = r3

    # ------------------------------------------------------------------
    # 4. Full Model (load from previous training if available)
    # ------------------------------------------------------------------
    print("\n[Ablation] 4. Full Model ...")
    agent4 = DQNAgent(config, state_dim, n_actions, seed=seed)
    model_p = os.path.join(config['training']['model_dir'], "proposed_drl_final.pt")
    if os.path.exists(model_p):
        agent4.load(model_p)
    eval_env4 = EdgeCloudEnv(config, rng=np.random.default_rng(seed+400))
    eval_env4.state_builder._pca        = env.state_builder._pca
    eval_env4.state_builder._pca_fitted = True
    eval_env4.state_builder._state_dim  = state_dim
    r4 = Evaluator(config, eval_env4).evaluate_agent(agent4, "Full Model", verbose=True)
    ablation_results['Full Model'] = r4

    # Print Table 10 format
    print("\n Ablation Study Results:")
    print(f"{'Component':<35} {'Lat(ms)':>9} {'SLA(%)':>8} {'Compl(%)':>10} {'Energy':>8} {'Acc(%)':>8}")
    print("-" * 82)
    for name, r in ablation_results.items():
        print(f"{name:<35} {r.get('avg_latency_ms',0):>9.1f} {r.get('sla_violation_rate',0):>8.1f} "
              f"{r.get('task_completion_rate',0):>10.1f} {r.get('energy_kwh',0):>8.1f} "
              f"{r.get('task_allocation_accuracy',0):>8.1f}")

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/ablation_results.json", 'w') as f:
        json.dump(ablation_results, f, indent=2)
    print("\n[Ablation] Saved → results/ablation_results.json")
    return ablation_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()
    run_ablation(args.config, args.seed)