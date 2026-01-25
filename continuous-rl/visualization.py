"""Visualization utilities for continuous-time RL."""

import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_training(rewards: list, losses: list, save_path: str = None):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(rewards, 'b-', alpha=0.6)
    if len(rewards) > 20:
        w = len(rewards) // 10
        smoothed = np.convolve(rewards, np.ones(w)/w, mode='valid')
        ax1.plot(range(w-1, len(rewards)), smoothed, 'r-', lw=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(losses, 'g-', alpha=0.6)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_value_function(agent, bounds: tuple, resolution: int = 50, save_path: str = None):
    """Plot V(x) = max_a Q(x,a) for 2D state space."""
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    states = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), 
                          dtype=torch.float32, device=agent.device)
    
    with torch.no_grad():
        V = agent.max_q(states, agent.q_net).cpu().numpy()
    
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, V.reshape(resolution, resolution), levels=50, cmap='viridis')
    plt.colorbar(label='V(x)')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('Learned Value Function')
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_trajectory(states: np.ndarray, target: np.ndarray = None, save_path: str = None):
    """Plot trajectory in state space."""
    plt.figure(figsize=(8, 6))
    plt.plot(states[:, 0], states[:, 1], 'b-', alpha=0.7, lw=1)
    plt.scatter(*states[0], c='green', s=100, marker='o', label='Start', zorder=5)
    plt.scatter(*states[-1], c='red', s=100, marker='x', label='End', zorder=5)
    if target is not None:
        plt.scatter(*target, c='gold', s=150, marker='*', label='Target', zorder=5)
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('State Trajectory')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
