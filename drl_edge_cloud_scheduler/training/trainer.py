import os
import json
import numpy as np
from typing import Dict, List, Optional
import yaml


class Trainer:

    def __init__(self, config: dict, agent, env, agent_name: str = "dqn"):
        self.config     = config
        self.agent      = agent
        self.env        = env
        self.agent_name = agent_name
        self.train_cfg  = config['training']

        os.makedirs(self.train_cfg['results_dir'], exist_ok=True)
        os.makedirs(self.train_cfg['model_dir'],   exist_ok=True)

        # History
        self.episode_rewards:     List[float] = []
        self.episode_losses:      List[float] = []
        self.episode_metrics:     List[Dict]  = []
        self.convergence_episode: Optional[int] = None

    def train(
        self,
        num_episodes: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Run the full training loop.

        Returns:
            Dictionary of training history.
        """
        n_episodes  = num_episodes or self.train_cfg['num_episodes']
        log_freq    = self.train_cfg['log_freq']
        save_freq   = self.train_cfg['save_freq']
        min_buffer  = self.train_cfg['min_buffer_size']
        conv_window = self.train_cfg['convergence_window']
        conv_thresh = self.train_cfg['convergence_threshold']

        print(f"[Trainer] Starting training: {n_episodes} episodes | "
              f"agent={self.agent_name}")

        for episode in range(1, n_episodes + 1):

            state = self.env.reset()
            episode_reward = 0.0
            episode_loss   = 0.0
            loss_count     = 0
            done           = False

            while not done:
                # Action mask (Section 4.4)
                mask = self.env.action_mask()

                # Algorithm 1, lines 7-11: ε-greedy action selection
                action = self.agent.select_action(
                    state, action_mask=mask, training=True
                )

                # Algorithm 1, line 12: Execute action
                next_state, reward, done, info = self.env.step(action)

                # Algorithm 1, line 13: Store transition
                self.agent.store_transition(state, action, reward, next_state, done)

                episode_reward += reward
                state = next_state

                # Algorithm 1, lines 14-25: Training update
                if hasattr(self.agent, 'replay_buffer'):
                    if len(self.agent.replay_buffer) >= min_buffer:
                        loss = self.agent.update()
                        if loss is not None:
                            episode_loss += loss
                            loss_count   += 1
                elif hasattr(self.agent, 'rewards'):
                    # PPO: store reward in rollout buffer
                    self.agent.store_reward(reward, done)

            # PPO update after rollout
            if hasattr(self.agent, 'rewards') and not hasattr(self.agent, 'replay_buffer'):
                loss = self.agent.update()
                if loss is not None:
                    episode_loss = loss
                    loss_count   = 1

            # Algorithm 1, line 29: ε decay
            if hasattr(self.agent, 'decay_epsilon'):
                self.agent.decay_epsilon()

            # Record metrics
            avg_loss = episode_loss / max(loss_count, 1)
            self.episode_rewards.append(episode_reward)
            self.episode_losses.append(avg_loss)
            metrics = self.env.get_episode_metrics()
            self.episode_metrics.append(metrics)

            if verbose and episode % log_freq == 0:
                recent_r = np.mean(self.episode_rewards[-log_freq:])
                eps = getattr(self.agent, 'epsilon', 0.0)
                buf = len(getattr(self.agent, 'replay_buffer', [])) \
                      if hasattr(self.agent, 'replay_buffer') else 0
                print(
                    f"  Ep {episode:4d}/{n_episodes} | "
                    f"Reward: {recent_r:7.3f} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"ε: {eps:.4f} | "
                    f"Buffer: {buf} | "
                    f"Latency: {metrics.get('avg_latency_ms', 0):.1f}ms | "
                    f"SLA: {metrics.get('sla_violation_rate', 0):.1f}%"
                )

            if episode % save_freq == 0:
                ckpt_path = os.path.join(
                    self.train_cfg['model_dir'],
                    f"{self.agent_name}_ep{episode}.pt",
                )
                if hasattr(self.agent, 'save'):
                    self.agent.save(ckpt_path)

            if (self.convergence_episode is None
                    and len(self.episode_rewards) >= conv_window):
                recent = self.episode_rewards[-conv_window:]
                rel_var = np.std(recent) / (abs(np.mean(recent)) + 1e-9)
                if rel_var < conv_thresh:
                    self.convergence_episode = episode
                    if verbose:
                        print(f"\n[Trainer] Converged at episode {episode} "
                              f"(rel_var={rel_var:.4f} < {conv_thresh})\n")

        # Save final model
        final_path = os.path.join(
            self.train_cfg['model_dir'], f"{self.agent_name}_final.pt"
        )
        if hasattr(self.agent, 'save'):
            self.agent.save(final_path)

        history = self._compile_history()
        self._save_history(history)
        return history
    def _compile_history(self) -> Dict:
        rewards = np.array(self.episode_rewards)
        losses  = np.array(self.episode_losses)
        return {
            'agent':                self.agent_name,
            'episode_rewards':      rewards.tolist(),
            'episode_losses':       losses.tolist(),
            'episode_metrics':      self.episode_metrics,
            'convergence_episode':  self.convergence_episode,
            'final_avg_reward':     float(np.mean(rewards[-100:])) if len(rewards) >= 100 else float(np.mean(rewards)),
            'final_avg_latency_ms': float(np.mean([
                m.get('avg_latency_ms', 0) for m in self.episode_metrics[-100:]
            ])) if self.episode_metrics else 0.0,
            'final_sla_rate':       float(np.mean([
                m.get('sla_violation_rate', 0) for m in self.episode_metrics[-100:]
            ])) if self.episode_metrics else 0.0,
        }

    def _save_history(self, history: Dict):
        path = os.path.join(
            self.train_cfg['results_dir'],
            f"training_history_{self.agent_name}.json",
        )
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"[Trainer] History saved → {path}")