"""Example: Continuous-time Deep Q-Learning for stabilization.

Double integrator: dx = v dt, dv = a dt + σ dW
Goal: Stabilize at origin with r(x,a) = -||x||² - ||v||² - α||a||²
"""

import torch
import numpy as np
import torchsde
from q_learning import ContinuousTimeDQN
from visualization import plot_training, plot_value_function, plot_trajectory


class DoubleIntegratorSDE(torchsde.SDEIto):
    """Double integrator: [x, v] with control a."""
    noise_type = 'diagonal'
    sde_type = 'ito'
    
    def __init__(self, policy_fn, sigma=0.1):
        super().__init__(noise_type='diagonal')
        self.sigma = sigma
        self.policy_fn = policy_fn  # policy_fn(state) -> action
    
    def f(self, t, y):  # Drift: [v, a]
        a = self.policy_fn(y)  # (batch, 1)
        return torch.cat([y[:, 1:2], a], dim=1)
    
    def g(self, t, y):  # Diffusion: only on velocity
        return torch.tensor([[0.0, self.sigma]], device=y.device).expand(y.shape[0], 2)


def reward_fn(state, action, alpha=0.1):
    """r(x,a) = -x² - v² - α*a²"""
    return -(state[:, 0:1]**2 + state[:, 1:2]**2 + alpha * action**2)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Agent
    agent = ContinuousTimeDQN(
        state_dim=2,
        action_dim=1,
        action_bounds=(torch.tensor([-2.0], device=device), 
                       torch.tensor([2.0], device=device)),
        rho=0.1,
        lr=1e-3,
        device=device
    )
    
    # Time grid for simulation
    dt = 0.1
    T = 4.0  # Episode duration
    n_steps = int(T / dt)
    ts = torch.linspace(0, T, n_steps + 1, device=device)
    
    n_episodes = 500
    batch_size = 256
    
    rewards_hist, losses_hist = [], []
    buffer_size = 10000
    buffer_states = torch.empty((buffer_size, 2), device=device)
    buffer_actions = torch.empty((buffer_size, 1), device=device)
    buffer_rewards = torch.empty((buffer_size, 1), device=device)
    buffer_drifts = torch.empty((buffer_size, 2), device=device)
    buffer_ptr = 0
    buffer_filled = 0
    
    print("Training...")
    for ep in range(n_episodes):
        # Create SDE with current policy (vectorized)
        policy_fn = lambda y: agent.select_action(y, explore=True)
        sde = DoubleIntegratorSDE(policy_fn, sigma=0.05)
        
        # Simulate full trajectory
        state0 = torch.randn(1, 2, device=device) * 2
        with torch.no_grad():
            traj = torchsde.sdeint(sde, state0, ts, method='euler', dt_min=0.1)  # (n_steps+1, 1, 2)
        
        traj = traj.squeeze(1)  # (n_steps+1, 2)
        
        # Collect transitions (vectorized)
        states = traj[:-1]  # (n_steps, 2)
        next_states = traj[1:]  # (n_steps, 2)
        with torch.no_grad():
            actions = policy_fn(states)  # (n_steps, 1)
        rewards = reward_fn(states, actions)  # (n_steps, 1)
        drifts = next_states - states  # (n_steps, 2)
        
        # Add to buffer (circular buffer, vectorized)
        n_new = n_steps
        end_ptr = buffer_ptr + n_new
        if end_ptr <= buffer_size:
            buffer_states[buffer_ptr:end_ptr] = states
            buffer_actions[buffer_ptr:end_ptr] = actions
            buffer_rewards[buffer_ptr:end_ptr] = rewards
            buffer_drifts[buffer_ptr:end_ptr] = drifts
        else:
            # Wrap around
            first_part = buffer_size - buffer_ptr
            buffer_states[buffer_ptr:] = states[:first_part]
            buffer_actions[buffer_ptr:] = actions[:first_part]
            buffer_rewards[buffer_ptr:] = rewards[:first_part]
            buffer_drifts[buffer_ptr:] = drifts[:first_part]
            buffer_states[:n_new - first_part] = states[first_part:]
            buffer_actions[:n_new - first_part] = actions[first_part:]
            buffer_rewards[:n_new - first_part] = rewards[first_part:]
            buffer_drifts[:n_new - first_part] = drifts[first_part:]
        buffer_ptr = end_ptr % buffer_size
        buffer_filled = min(buffer_filled + n_new, buffer_size)
        
        ep_reward = rewards.sum().item() * dt
        
        # Update from buffer
        if buffer_filled >= batch_size:
            idx = torch.randint(0, buffer_filled, (batch_size,), device=device)
            states_b = buffer_states[idx]
            actions_b = buffer_actions[idx]
            rewards_b = buffer_rewards[idx]
            drifts_b = buffer_drifts[idx]
            
            loss = agent.update(states_b, actions_b, rewards_b, drifts_b, dt)
            agent.update_target()
        else:
            loss = 0
        
        # Decay exploration
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
        
        rewards_hist.append(ep_reward)
        losses_hist.append(loss)
        
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{n_episodes} | Reward: {np.mean(rewards_hist[-50:]):.2f} | ε: {agent.epsilon:.3f}")
    
    print("Training complete!")
    
    # Visualizations
    plot_training(rewards_hist, losses_hist)
    plot_value_function(agent, bounds=(-3, 3))
    
    # Evaluate policy
    print("Evaluating learned policy...")
    greedy_policy = lambda y: agent.select_action(y, explore=False)
    sde_eval = DoubleIntegratorSDE(greedy_policy, sigma=0.05)
    state0 = torch.tensor([[2.0, 0.0]], device=device)
    with torch.no_grad():
        traj = torchsde.sdeint(sde_eval, state0, ts, method='euler', dt_min=0.1)
    
    states_eval = traj.squeeze(1).cpu().numpy()
    plot_trajectory(states_eval, target=np.array([0, 0]))
    
    return agent


if __name__ == "__main__":
    train()
