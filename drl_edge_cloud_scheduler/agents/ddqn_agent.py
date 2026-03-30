import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
import os

from models.dqn_network import DQNNetwork
from agents.replay_buffer import ReplayBuffer


class DDQNAgent:
    """Double DQN baseline agent (Section 5.3, Table 6-9)."""

    def __init__(self, config: dict, state_dim: int, n_actions: int, seed: int = 42):
        self.config    = config
        self.cfg       = config['ddqn']
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.rng       = np.random.default_rng(seed)

        hw = config.get('hardware', {}).get('device', 'cpu')
        self.device = torch.device(
            hw if hw == 'cuda' and torch.cuda.is_available() else 'cpu'
        )

        self.q_net = DQNNetwork(
            state_dim, n_actions, self.cfg['hidden_sizes']
        ).to(self.device)
        self.target_net = DQNNetwork(
            state_dim, n_actions, self.cfg['hidden_sizes']
        ).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.q_net.parameters(), lr=self.cfg['learning_rate']
        )
        self.replay_buffer = ReplayBuffer(
            self.cfg['replay_buffer_size'], state_dim, seed=seed
        )

        self.epsilon       = self.cfg['epsilon_start']
        self.epsilon_min   = self.cfg['epsilon_min']
        self.epsilon_decay = self.cfg['epsilon_decay']
        self.gamma         = self.cfg['gamma']
        self.batch_size    = self.cfg['batch_size']
        self.target_update = self.cfg['target_update_freq']

        self.total_steps  = 0
        self.update_count = 0
        self.losses       = []

    def select_action(
        self,
        state: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        training: bool = True,
    ) -> int:
        valid = np.where(action_mask)[0] if action_mask is not None else np.arange(self.n_actions)
        if len(valid) == 0:
            return int(self.rng.integers(0, self.n_actions))

        if training and self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid))

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t).squeeze(0).cpu().numpy()

        masked_q = np.full(self.n_actions, -np.inf)
        masked_q[valid] = q_values[valid]
        return int(np.argmax(masked_q))

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        s  = torch.FloatTensor(states).to(self.device)
        a  = torch.LongTensor(actions).to(self.device)
        r  = torch.FloatTensor(rewards).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        d  = torch.FloatTensor(dones).to(self.device)

        current_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # DDQN: main net selects action, target net evaluates value
        with torch.no_grad():
            next_actions = self.q_net(ns).argmax(dim=1, keepdim=True)
            next_q       = self.target_net(ns).gather(1, next_actions).squeeze(1)
        target_q = (r + self.gamma * next_q * (1.0 - d)).detach()

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.update_count += 1
        self.losses.append(float(loss.item()))

        if self.update_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.total_steps += 1

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({'q_net': self.q_net.state_dict(), 'epsilon': self.epsilon}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt['q_net'])
        self.target_net.load_state_dict(ckpt['q_net'])
        self.epsilon = ckpt.get('epsilon', self.epsilon_min)