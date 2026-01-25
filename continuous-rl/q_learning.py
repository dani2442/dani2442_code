"""Continuous-time Deep Q-Learning using torchsde.

Implements: ρQ(x,a) = r(x,a) + L^a(max_a' Q(x,a'))
"""

import torch
import torch.nn as nn
from model import QNetwork


class ContinuousTimeDQN:
    """Continuous-time Deep Q-Learning agent."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_bounds: tuple,
        rho: float = 0.1,
        lr: float = 1e-3,
        n_action_samples: int = 32,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low, self.action_high = action_bounds
        self.rho = rho
        self.device = device
        self.n_action_samples = n_action_samples
        
        self.q_net = QNetwork(state_dim, action_dim).to(device)
        self.target_q_net = QNetwork(state_dim, action_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.epsilon = 1.0
    
    def sample_actions(self, batch_size: int) -> torch.Tensor:
        """Sample random actions."""
        u = torch.rand(batch_size, self.n_action_samples, self.action_dim, device=self.device)
        return self.action_low + (self.action_high - self.action_low) * u
    
    def max_q(self, states: torch.Tensor, q_net: QNetwork) -> torch.Tensor:
        """Compute max_a Q(s,a) over sampled actions."""
        batch_size = states.shape[0]
        actions = self.sample_actions(batch_size)  # (B, N, A)
        states_exp = states.unsqueeze(1).expand(-1, self.n_action_samples, -1)  # (B, N, S)
        
        q_vals = q_net(states_exp.reshape(-1, self.state_dim), 
                       actions.reshape(-1, self.action_dim))  # (B*N, 1)
        q_vals = q_vals.reshape(batch_size, self.n_action_samples)
        return q_vals.max(dim=1, keepdim=True)[0]
    
    def select_action(self, state: torch.Tensor, explore: bool = True) -> torch.Tensor:
        """Epsilon-greedy action selection. Supports batched states."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        batch_size = state.shape[0]
        
        if explore and torch.rand(1).item() < self.epsilon:
            u = torch.rand(batch_size, self.action_dim, device=self.device)
            action = self.action_low + (self.action_high - self.action_low) * u
        else:
            with torch.no_grad():
                actions = self.sample_actions(batch_size)  # (B, N, A)
                states_exp = state.unsqueeze(1).expand(-1, self.n_action_samples, -1)
                q_vals = self.q_net(states_exp.reshape(-1, self.state_dim),
                                    actions.reshape(-1, self.action_dim))
                q_vals = q_vals.reshape(batch_size, self.n_action_samples)
                best_idx = q_vals.argmax(dim=1)
                action = actions[torch.arange(batch_size, device=self.device), best_idx]
        
        return action.squeeze(0) if squeeze else action
    
    def compute_L_V(self, states: torch.Tensor, drift: torch.Tensor) -> torch.Tensor:
        """Compute L^a V(x) ≈ ∇V(x)·f(x,a) (ignoring diffusion term for simplicity)."""
        states = states.requires_grad_(True)
        V = self.max_q(states, self.target_q_net)
        grad_V = torch.autograd.grad(V.sum(), states, create_graph=True)[0]
        return (grad_V * drift).sum(dim=-1, keepdim=True)
    
    def update(self, states, actions, rewards, drifts, dt: float) -> float:
        """Update using: ρQ = r + L^a V, so Q = (r + L^a V) / ρ."""
        drift_rate = drifts / dt
        
        current_q = self.q_net(states, actions)
        L_V = self.compute_L_V(states, drift_rate)
        # Correct HJB: ρV = r + L^a V  =>  V = (r + L^a V) / ρ
        target_q = (rewards + L_V) / self.rho
        
        loss = nn.functional.mse_loss(current_q, target_q.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target(self, tau: float = 0.01):
        for tp, p in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            tp.data.lerp_(p.data, tau)
