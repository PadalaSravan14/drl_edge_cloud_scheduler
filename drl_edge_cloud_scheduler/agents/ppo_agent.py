import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Tuple, Optional
import os


class PPOActorCritic(nn.Module):
    """Shared-backbone Actor-Critic network for PPO."""

    def __init__(self, state_dim: int, n_actions: int, hidden_sizes: List[int]):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.actor  = nn.Linear(in_dim, n_actions)   # policy logits
        self.critic = nn.Linear(in_dim, 1)            # value estimate

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor):
        features = self.backbone(state)
        logits   = self.actor(features)
        value    = self.critic(features).squeeze(-1)
        return logits, value

    def get_action(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None):
        logits, value = self(state)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


class PPOAgent:

    def __init__(self, config: dict, state_dim: int, n_actions: int, seed: int = 42):
        self.config    = config
        self.cfg       = config['ppo']
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.rng       = np.random.default_rng(seed)
        torch.manual_seed(seed)

        hw = config.get('hardware', {}).get('device', 'cpu')
        self.device = torch.device(
            hw if hw == 'cuda' and torch.cuda.is_available() else 'cpu'
        )

        self.ac_net = PPOActorCritic(
            state_dim, n_actions, self.cfg['hidden_sizes']
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.ac_net.parameters(), lr=self.cfg['learning_rate']
        )

        # Rollout buffer
        self.states:      List[np.ndarray] = []
        self.actions:     List[int]        = []
        self.log_probs:   List[float]      = []
        self.rewards:     List[float]      = []
        self.values:      List[float]      = []
        self.dones:       List[bool]       = []

        self.clip_epsilon  = self.cfg['clip_epsilon']
        self.gamma         = self.cfg['gamma']
        self.epochs        = self.cfg['epochs']
        self.batch_size    = self.cfg['batch_size']
        self.ent_coeff     = self.cfg.get('entropy_coeff', 0.01)
        self.val_coeff     = self.cfg.get('value_loss_coeff', 0.5)
        self.max_grad_norm = self.cfg.get('max_grad_norm', 0.5)

        self.update_count = 0
        self.losses       = []

    def select_action(
        self,
        state: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        training: bool = True,
    ) -> int:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mask_t  = None
        if action_mask is not None:
            mask_t = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, _, value = self.ac_net.get_action(state_t, mask_t)

        if training:
            self.states.append(state)
            self.actions.append(int(action.item()))
            self.log_probs.append(float(log_prob.item()))
            self.values.append(float(value.item()))

        return int(action.item())

    def store_reward(self, reward: float, done: bool):
        self.rewards.append(reward)
        self.dones.append(done)

    def update(self) -> Optional[float]:
        """PPO update over collected rollout."""
        n = min(len(self.rewards), len(self.states))
        if n < 2:
            return None

        # Compute discounted returns
        returns = []
        G = 0.0
        for r, d in zip(reversed(self.rewards[:n]), reversed(self.dones[:n])):
            G = r + self.gamma * G * (1.0 - float(d))
            returns.insert(0, G)

        states_t    = torch.FloatTensor(np.array(self.states[:n])).to(self.device)
        actions_t   = torch.LongTensor(self.actions[:n]).to(self.device)
        old_lp_t    = torch.FloatTensor(self.log_probs[:n]).to(self.device)
        returns_t   = torch.FloatTensor(returns).to(self.device)
        values_t    = torch.FloatTensor(self.values[:n]).to(self.device)

        advantages  = (returns_t - values_t).detach()
        advantages  = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        for _ in range(self.epochs):
            idx = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                b_idx = idx[start:start + self.batch_size]
                logits, values = self.ac_net(states_t[b_idx])
                dist     = Categorical(logits=logits)
                new_lp   = dist.log_prob(actions_t[b_idx])
                entropy  = dist.entropy().mean()

                ratio    = (new_lp - old_lp_t[b_idx]).exp()
                adv_b    = advantages[b_idx]
                surr1    = ratio * adv_b
                surr2    = torch.clamp(ratio, 1 - self.clip_epsilon,
                                       1 + self.clip_epsilon) * adv_b
                actor_loss  = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values.squeeze(), returns_t[b_idx])
                loss        = (actor_loss
                               + self.val_coeff * critic_loss
                               - self.ent_coeff * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.ac_net.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                total_loss += float(loss.item())

        # Clear rollout buffer
        self.states.clear(); self.actions.clear(); self.log_probs.clear()
        self.rewards.clear(); self.values.clear(); self.dones.clear()

        self.update_count += 1
        avg_loss = total_loss / max(self.epochs, 1)
        self.losses.append(avg_loss)
        return avg_loss

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(self.ac_net.state_dict(), path)

    def load(self, path: str):
        self.ac_net.load_state_dict(torch.load(path, map_location=self.device))