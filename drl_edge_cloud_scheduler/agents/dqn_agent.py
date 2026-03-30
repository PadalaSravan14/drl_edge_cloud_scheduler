import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
import os

from models.dqn_network import DQNNetwork, LSTMDQNNetwork
from agents.replay_buffer import ReplayBuffer


class DQNAgent:


    def __init__(self, config: dict, state_dim: int, n_actions: int, seed: int = 42):
        self.config    = config
        self.cfg       = config['dqn']
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.rng       = np.random.default_rng(seed)

        # Device
        hw_device = config.get('hardware', {}).get('device', 'cpu')
        self.device = torch.device(
            hw_device if hw_device == 'cuda' and torch.cuda.is_available() else 'cpu'
        )

        NetworkClass = LSTMDQNNetwork if self.cfg.get('use_lstm', False) else DQNNetwork
        self.q_net = NetworkClass(
            state_dim=state_dim,
            n_actions=n_actions,
            hidden_sizes=self.cfg['hidden_sizes'],
        ).to(self.device)

        self.target_net = NetworkClass(
            state_dim=state_dim,
            n_actions=n_actions,
            hidden_sizes=self.cfg['hidden_sizes'],
        ).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.q_net.parameters(),
            lr=self.cfg['learning_rate'],
        )

        self.replay_buffer = ReplayBuffer(
            capacity=self.cfg['replay_buffer_size'],
            state_dim=state_dim,
            seed=seed,
        )

        self.epsilon     = self.cfg['epsilon_start']
        self.epsilon_min = self.cfg['epsilon_min']
        self.epsilon_decay = self.cfg['epsilon_decay']

        # Training counters
        self.total_steps   = 0
        self.update_count  = 0
        self.gamma         = self.cfg['gamma']
        self.batch_size    = self.cfg['batch_size']
        self.target_update = self.cfg['target_update_freq']

        # Loss tracking
        self.losses = []

    def select_action(
        self,
        state: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        training: bool = True,
    ) -> int:
        """
        ε-greedy action selection with action masking (Section 4.4).

        With prob ε  → random valid action (exploration)
        With prob 1-ε → argmax Q(s,a;θ) over valid actions (exploitation)
        """
        # Determine valid actions
        if action_mask is not None:
            valid_actions = np.where(action_mask)[0]
        else:
            valid_actions = np.arange(self.n_actions)

        if len(valid_actions) == 0:
            return int(self.rng.integers(0, self.n_actions))

        if training and self.rng.random() < self.epsilon:
            # Exploration: random valid action
            return int(self.rng.choice(valid_actions))

        # Exploitation: argmax Q with masking
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.q_net.eval()
        with torch.no_grad():
            if self.cfg.get('use_lstm', False):
                q_values, _ = self.q_net(state_t)
            else:
                q_values = self.q_net(state_t)
        self.q_net.train()

        q_values = q_values.squeeze(0).cpu().numpy()

        # Apply action masking: set invalid Q-values to -inf
        masked_q = np.full(self.n_actions, -np.inf)
        masked_q[valid_actions] = q_values[valid_actions]

        return int(np.argmax(masked_q))

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.replay_buffer.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    # Training update  (Algorithm 1, lines 14-24)
    # ------------------------------------------------------------------
    def update(self) -> Optional[float]:
        """
        Sample mini-batch from replay buffer and perform one gradient step.
        Returns TD loss value, or None if buffer not ready.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample mini-batch B of size 64  (Algorithm 1, line 15)
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        self.q_net.eval()
        if self.cfg.get('use_lstm', False):
            current_q, _ = self.q_net(states_t)
            with torch.no_grad():
                next_q, _ = self.target_net(next_states_t)
        else:
            current_q = self.q_net(states_t)
            with torch.no_grad():
                next_q = self.target_net(next_states_t)
        self.q_net.train()

        # Gather Q-values for taken actions
        current_q = current_q.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values
        max_next_q = next_q.max(dim=1)[0]
        target_q   = rewards_t + self.gamma * max_next_q * (1.0 - dones_t)
        target_q   = target_q.detach()


        loss = nn.MSELoss()(current_q, target_q)

        # Gradient descent step  (Algorithm 1, line 24)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.update_count += 1
        loss_val = float(loss.item())
        self.losses.append(loss_val)

        if self.update_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss_val


    def decay_epsilon(self):
        """ε ← max(ε_min, ε × λ)"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.total_steps += 1

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'q_net_state':      self.q_net.state_dict(),
            'target_net_state': self.target_net.state_dict(),
            'optimizer_state':  self.optimizer.state_dict(),
            'epsilon':          self.epsilon,
            'total_steps':      self.total_steps,
            'update_count':     self.update_count,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt['q_net_state'])
        self.target_net.load_state_dict(ckpt['target_net_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.epsilon      = ckpt.get('epsilon', self.epsilon_min)
        self.total_steps  = ckpt.get('total_steps', 0)
        self.update_count = ckpt.get('update_count', 0)
        print(f"[DQNAgent] Loaded checkpoint from {path}")