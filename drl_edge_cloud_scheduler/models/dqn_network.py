import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ---------------------------------------------------------------------------
# Standard DNN Q-Network
# ---------------------------------------------------------------------------
class DQNNetwork(nn.Module):
    """
    Multi-layer fully-connected Q-network.
    Q(s, a; θ) mapping state → Q-value per action.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_sizes: List[int] = None,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 256]

        self.state_dim  = state_dim
        self.n_actions  = n_actions

        # Build FC layers
        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        """He (Kaiming) initialisation for stable gradient flow (Section 4.6)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, state_dim) float tensor
        Returns:
            q_values: (batch, n_actions)
        """
        return self.net(state)



# ---------------------------------------------------------------------------
class LSTMDQNNetwork(nn.Module):

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_size: int = 256,
        lstm_hidden: int = 256,
    ):
        super().__init__()
        self.state_dim   = state_dim
        self.n_actions   = n_actions
        self.lstm_hidden = lstm_hidden

        self.fc1   = nn.Linear(state_dim, hidden_size)
        self.lstm  = nn.LSTM(hidden_size, lstm_hidden, batch_first=True)
        self.fc2   = nn.Linear(lstm_hidden, hidden_size)
        self.out   = nn.Linear(hidden_size, n_actions)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(
        self,
        state: torch.Tensor,
        hidden=None,
    ):

        # Handle both 2D (single step) and 3D (sequence) input
        if state.dim() == 2:
            state = state.unsqueeze(1)   # (batch, 1, state_dim)

        x = F.relu(self.fc1(state))      # (batch, seq, 256)
        lstm_out, hidden = self.lstm(x, hidden)  # (batch, seq, 256)
        x = lstm_out[:, -1, :]            # take last timestep
        x = F.relu(self.fc2(x))
        q_values = self.out(x)
        return q_values, hidden