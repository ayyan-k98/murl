"""
Utilities for Visualization and Analysis

Helper functions for plotting and analysis.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import torch

from config import config
from data_structures import CoverageMetrics, WorldState, RobotState


def plot_training_curves(metrics: CoverageMetrics,
                         save_path: str = None,
                         show: bool = True):
    """
    Plot training curves.

    Args:
        metrics: Training metrics
        save_path: Path to save figure
        show: Whether to display figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Progress', fontsize=16)

    # 1. Episode Rewards
    ax = axes[0, 0]
    ax.plot(metrics.episode_rewards, alpha=0.3, label='Episode')
    if len(metrics.episode_rewards) > 100:
        window = min(100, len(metrics.episode_rewards) // 10)
        smoothed = np.convolve(metrics.episode_rewards,
                              np.ones(window)/window,
                              mode='valid')
        ax.plot(smoothed, label=f'Smoothed ({window})', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Coverage Percentage
    ax = axes[0, 1]
    coverages_pct = [c * 100 for c in metrics.episode_coverages]
    ax.plot(coverages_pct, alpha=0.3, label='Episode')
    if len(coverages_pct) > 100:
        window = min(100, len(coverages_pct) // 10)
        smoothed = np.convolve(coverages_pct,
                              np.ones(window)/window,
                              mode='valid')
        ax.plot(smoothed, label=f'Smoothed ({window})', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Coverage Percentage')
    ax.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Target (70%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Episode Length
    ax = axes[0, 2]
    ax.plot(metrics.episode_lengths, alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Episode Length')
    ax.grid(True, alpha=0.3)

    # 4. DQN Loss
    ax = axes[1, 0]
    if len(metrics.dqn_loss) > 0:
        ax.plot(metrics.dqn_loss, alpha=0.3, label='Step')
        if len(metrics.dqn_loss) > 100:
            window = min(100, len(metrics.dqn_loss) // 10)
            smoothed = np.convolve(metrics.dqn_loss,
                                  np.ones(window)/window,
                                  mode='valid')
            ax.plot(smoothed, label=f'Smoothed ({window})', linewidth=2)
        ax.set_xlabel('Optimization Step')
        ax.set_ylabel('Loss')
        ax.set_title('DQN Loss')
        ax.legend()
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 5. Epsilon Decay
    ax = axes[1, 1]
    ax.plot(metrics.epsilon_values)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.set_title('Exploration Rate (Epsilon)')
    ax.grid(True, alpha=0.3)

    # 6. Gradient Norms
    ax = axes[1, 2]
    if len(metrics.grad_norms) > 0:
        ax.plot(metrics.grad_norms, alpha=0.3, label='Step')
        if len(metrics.grad_norms) > 100:
            window = min(100, len(metrics.grad_norms) // 10)
            smoothed = np.convolve(metrics.grad_norms,
                                  np.ones(window)/window,
                                  mode='valid')
            ax.plot(smoothed, label=f'Smoothed ({window})', linewidth=2)
        ax.axhline(y=config.EXPLOSION_THRESHOLD, color='r',
                  linestyle='--', alpha=0.5, label='Explosion Threshold')
        ax.set_xlabel('Optimization Step')
        ax.set_ylabel('Gradient Norm')
        ax.set_title(f'Gradient Norms (Explosions: {metrics.grad_explosions})')
        ax.legend()
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved training curves: {save_path}")

    if show:
        plt.show()


def plot_validation_results(metrics: CoverageMetrics,
                            save_path: str = None,
                            show: bool = True):
    """
    Plot validation results over training.

    Args:
        metrics: Training metrics with validation scores
        save_path: Path to save figure
        show: Whether to display figure
    """
    if len(metrics.validation_scores) == 0:
        print("No validation scores to plot")
        return

    episodes = sorted(metrics.validation_scores.keys())
    map_types = list(metrics.validation_scores[episodes[0]].keys())

    fig, ax = plt.subplots(figsize=(12, 6))

    for map_type in map_types:
        coverages = [metrics.validation_scores[ep][map_type] * 100
                    for ep in episodes]
        ax.plot(episodes, coverages, marker='o', label=map_type, linewidth=2)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Coverage (%)', fontsize=12)
    ax.set_title('Validation Coverage by Map Type', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved validation results: {save_path}")

    if show:
        plt.show()


def visualize_episode(world_state: WorldState,
                     robot_state: RobotState,
                     trajectory: list = None,
                     save_path: str = None,
                     show: bool = True,
                     title: str = "Coverage Visualization"):
    """
    Visualize a single episode.

    Args:
        world_state: World state
        robot_state: Robot state
        trajectory: List of positions (optional)
        save_path: Path to save figure
        show: Whether to display figure
        title: Figure title
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=16)

    grid_size = world_state.grid_size

    # 1. Ground Truth Map
    ax = axes[0]
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect('equal')
    ax.set_title('Ground Truth Map')
    ax.invert_yaxis()

    # Draw obstacles
    for (x, y) in world_state.obstacles:
        rect = Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='black', edgecolor='gray')
        ax.add_patch(rect)

    # Draw coverage
    for x in range(grid_size):
        for y in range(grid_size):
            if (x, y) not in world_state.obstacles:
                coverage = world_state.coverage_map[x, y]
                if coverage > 0:
                    color = plt.cm.Greens(coverage)
                    rect = Rectangle((x - 0.5, y - 0.5), 1, 1,
                                   facecolor=color, edgecolor='gray', alpha=0.7)
                    ax.add_patch(rect)

    # Draw robot
    rx, ry = robot_state.position
    circle = Circle((rx, ry), 0.3, facecolor='blue', edgecolor='darkblue', linewidth=2)
    ax.add_patch(circle)

    # Draw trajectory
    if trajectory:
        traj_x = [p[0] for p in trajectory]
        traj_y = [p[1] for p in trajectory]
        ax.plot(traj_x, traj_y, 'b--', alpha=0.5, linewidth=1)

    ax.grid(True, alpha=0.2)

    # 2. Agent's Local Map (POMDP)
    ax = axes[1]
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect('equal')
    ax.set_title(f"Agent's Local Map ({len(robot_state.local_map)} cells sensed)")
    ax.invert_yaxis()

    # Draw sensed cells
    for (x, y), (coverage, cell_type) in robot_state.local_map.items():
        if cell_type == "obstacle":
            color = 'black'
        elif coverage > 0:
            color = plt.cm.Greens(coverage)
        else:
            color = 'lightgray'

        rect = Rectangle((x - 0.5, y - 0.5), 1, 1,
                        facecolor=color, edgecolor='gray', alpha=0.7)
        ax.add_patch(rect)

    # Draw robot
    circle = Circle((rx, ry), 0.3, facecolor='blue', edgecolor='darkblue', linewidth=2)
    ax.add_patch(circle)

    # Draw sensor range
    sensor_circle = Circle((rx, ry), config.SENSOR_RANGE,
                          facecolor='none', edgecolor='blue',
                          linestyle='--', linewidth=2, alpha=0.5)
    ax.add_patch(sensor_circle)

    ax.grid(True, alpha=0.2)

    # 3. Visit Heatmap
    ax = axes[2]
    im = ax.imshow(robot_state.visit_heat.T, cmap='hot', origin='lower',
                   vmin=0, vmax=max(5, robot_state.visit_heat.max()))
    ax.set_title('Visit Heatmap')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, label='Visit Count')

    # Mark robot position
    ax.plot(rx, ry, 'bo', markersize=10)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization: {save_path}")

    if show:
        plt.show()


def save_metrics(metrics: CoverageMetrics, path: str):
    """Save metrics to file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"✓ Saved metrics: {path}")


def load_metrics(path: str) -> CoverageMetrics:
    """Load metrics from file."""
    with open(path, 'rb') as f:
        metrics = pickle.load(f)
    print(f"✓ Loaded metrics: {path}")
    return metrics


def print_statistics(metrics: CoverageMetrics, window: int = 100):
    """
    Print training statistics.

    Args:
        metrics: Training metrics
        window: Window size for averaging
    """
    print("\n" + "=" * 80)
    print("TRAINING STATISTICS")
    print("=" * 80)

    print(f"Total Episodes: {len(metrics.episode_rewards)}")

    if len(metrics.episode_rewards) > 0:
        print(f"\nRewards:")
        print(f"  Mean (last {window}): {metrics.get_recent_avg('reward', window):.2f}")
        print(f"  Max: {max(metrics.episode_rewards):.2f}")
        print(f"  Min: {min(metrics.episode_rewards):.2f}")

    if len(metrics.episode_coverages) > 0:
        print(f"\nCoverage:")
        print(f"  Mean (last {window}): {metrics.get_recent_avg('coverage', window)*100:.1f}%")
        print(f"  Max: {max(metrics.episode_coverages)*100:.1f}%")
        print(f"  Min: {min(metrics.episode_coverages)*100:.1f}%")

    if len(metrics.episode_lengths) > 0:
        print(f"\nEpisode Length:")
        print(f"  Mean (last {window}): {metrics.get_recent_avg('length', window):.1f}")

    if len(metrics.dqn_loss) > 0:
        print(f"\nDQN Loss:")
        print(f"  Mean (last {window}): {metrics.get_recent_avg('loss', window):.4f}")

    if len(metrics.grad_norms) > 0:
        recent_norms = metrics.grad_norms[-min(1000, len(metrics.grad_norms)):]
        print(f"\nGradient Norms:")
        print(f"  Mean: {np.mean(recent_norms):.2f}")
        print(f"  Max: {np.max(recent_norms):.2f}")
        print(f"  Explosions (>{config.EXPLOSION_THRESHOLD}): {metrics.grad_explosions}")

    if len(metrics.validation_scores) > 0:
        print(f"\nValidation Results:")
        for ep, scores in sorted(metrics.validation_scores.items())[-3:]:
            avg = np.mean(list(scores.values()))
            print(f"  Episode {ep}: {avg*100:.1f}% average")

    print("=" * 80)


if __name__ == "__main__":
    # Test visualization with dummy data
    from environment import CoverageEnvironment
    from agent import CoverageAgent

    print("Testing visualization utilities...")

    # Create dummy environment
    env = CoverageEnvironment(grid_size=20, map_type="room")
    state = env.reset()

    # Run a few steps
    agent = CoverageAgent(grid_size=20)
    trajectory = [state.position]

    for _ in range(20):
        action = agent.select_action(state, env.world_state, epsilon=0.5)
        next_state, reward, done, info = env.step(action)
        trajectory.append(next_state.position)
        state = next_state

        if done:
            break

    # Visualize
    visualize_episode(
        env.world_state,
        env.robot_state,
        trajectory=trajectory,
        save_path=None,
        show=True,
        title="Test Visualization"
    )

    print("✓ Visualization test complete")
