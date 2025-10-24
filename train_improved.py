"""
Improved Training Script

Combines all improvements:
1. Improved environment (better rewards)
2. Improved config (better hyperparameters)
3. Improved curriculum (easier start)
4. Baseline architecture (simpler, faster)
5. Better logging and diagnostics

Expected: 60-70% coverage in 400 episodes (not 30% in 800!)
"""

import numpy as np
import torch
import time
from collections import deque
from typing import List, Tuple

# Use BASELINE architecture (proven, fast)
from agent import CoverageAgent
from environment_improved import ImprovedCoverageEnvironment
from config_improved import config_improved, get_adaptive_epsilon_decay
from curriculum_improved import ImprovedCurriculumScheduler
from map_generator import MapGenerator
from utils import plot_training_curves, save_checkpoint


class ImprovedTrainer:
    """Trainer with all improvements."""

    def __init__(self,
                 grid_size: int = 20,
                 total_episodes: int = 2000,
                 device: str = None):

        self.grid_size = grid_size
        self.total_episodes = total_episodes
        self.device = device or config_improved.DEVICE

        # Create improved components
        self.curriculum = ImprovedCurriculumScheduler(total_episodes)
        self.map_generator = MapGenerator(grid_size)

        # Create BASELINE agent (not enhanced!)
        print("Creating BASELINE agent (simple, fast, proven)...")
        self.agent = CoverageAgent(
            grid_size=grid_size,
            learning_rate=config_improved.LEARNING_RATE,
            gamma=config_improved.GAMMA,
            device=self.device
        )

        # Override agent's epsilon settings
        self.agent.epsilon = config_improved.EPSILON_START
        self.agent.epsilon_min = config_improved.EPSILON_MIN

        # Tracking
        self.episode_rewards = []
        self.episode_coverages = []
        self.episode_losses = []
        self.phase_changes = []

        # Rolling average for adaptive epsilon
        self.recent_coverages = deque(maxlen=50)  # Last 50 episodes

    def train(self):
        """Run improved training loop."""
        print("=" * 80)
        print("IMPROVED TRAINING START")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Total Episodes: {self.total_episodes}")
        print(f"Grid Size: {self.grid_size}")
        print(f"Architecture: BASELINE (simple, fast)")
        print(f"Improvements: Rewards + Hyperparams + Curriculum")
        print("=" * 80)

        start_time = time.time()

        for episode in range(self.total_episodes):
            # Get current curriculum phase
            phase = self.curriculum.get_current_phase(episode)
            phase_info = self.curriculum.get_phase_info(episode)

            # Log phase changes
            if episode == phase.episode_start:
                print(f"\n{'='*80}")
                print(f"PHASE {phase_info['phase_number']}: {phase.name}")
                print(f"Episodes: {phase.episode_start}-{phase.episode_end}")
                print(f"Target Coverage: {phase.expected_coverage:.1%}")
                print(f"Map Distribution: {phase.map_distribution}")
                print(f"{'='*80}\n")
                self.phase_changes.append((episode, phase.name))

            # Sample map type from curriculum
            map_type = self._sample_map_type(phase.map_distribution)

            # Create environment with improved rewards
            env = ImprovedCoverageEnvironment(
                grid_size=self.grid_size,
                map_type=map_type
            )

            # Run episode
            episode_reward, episode_coverage, episode_loss = self._run_episode(
                env, episode, phase
            )

            # Track metrics
            self.episode_rewards.append(episode_reward)
            self.episode_coverages.append(episode_coverage)
            self.episode_losses.append(episode_loss)
            self.recent_coverages.append(episode_coverage)

            # Update epsilon with ADAPTIVE decay
            self._update_epsilon(episode_coverage)

            # Update target network
            if episode % config_improved.TARGET_UPDATE_FREQ == 0:
                self.agent.update_target_network()

            # Logging
            if episode % config_improved.LOG_INTERVAL == 0:
                self._log_progress(episode, phase_info)

            # Validation
            if episode % config_improved.VALIDATION_INTERVAL == 0 and episode > 0:
                self._validate(episode)

            # Checkpoint
            if episode % config_improved.CHECKPOINT_INTERVAL == 0 and episode > 0:
                self._save_checkpoint(episode)

        # Training complete
        total_time = time.time() - start_time
        self._print_summary(total_time)

    def _run_episode(self, env, episode: int, phase) -> Tuple[float, float, float]:
        """Run single episode."""
        state = env.reset()
        episode_reward = 0.0
        episode_losses = []

        for step in range(config_improved.MAX_EPISODE_STEPS):
            # Select action
            action = self.agent.select_action(state, env.world_state)

            # Execute action
            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            # Store transition
            graph_data = self.agent.graph_encoder.encode(state, env.world_state, 0)
            next_graph_data = self.agent.graph_encoder.encode(next_state, env.world_state, 0)
            self.agent.store_transition(graph_data, action, reward, next_graph_data, done, info)

            # Optimize
            loss = self.agent.optimize()
            if loss is not None:
                episode_losses.append(loss)

            state = next_state

            if done:
                break

        episode_coverage = info['coverage_pct']
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0

        return episode_reward, episode_coverage, avg_loss

    def _update_epsilon(self, current_coverage: float):
        """Update epsilon with ADAPTIVE decay."""
        recent_avg_coverage = np.mean(self.recent_coverages) if self.recent_coverages else 0.0

        # Get adaptive decay rate
        decay_rate = get_adaptive_epsilon_decay(
            self.agent.epsilon,
            recent_avg_coverage,
            config_improved
        )

        # Decay epsilon
        self.agent.epsilon = max(
            self.agent.epsilon_min,
            self.agent.epsilon * decay_rate
        )

    def _sample_map_type(self, distribution: dict) -> str:
        """Sample map type from distribution."""
        map_types = list(distribution.keys())
        probabilities = list(distribution.values())
        return np.random.choice(map_types, p=probabilities)

    def _log_progress(self, episode: int, phase_info: dict):
        """Log training progress."""
        recent_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0.0
        recent_coverage = np.mean(self.episode_coverages[-10:]) if self.episode_coverages else 0.0
        recent_loss = np.mean(self.episode_losses[-10:]) if self.episode_losses else 0.0

        print(f"Ep {episode:4d} | "
              f"Phase {phase_info['phase_number']:2d}/{phase_info['total_phases']} | "
              f"Cov: {recent_coverage:.2%} (target: {phase_info['target_coverage']:.2%}) | "
              f"Reward: {recent_reward:7.2f} | "
              f"Loss: {recent_loss:.4f} | "
              f"ε: {self.agent.epsilon:.3f} | "
              f"Buf: {len(self.agent.memory)}")

    def _validate(self, episode: int):
        """Run validation on multiple map types."""
        print(f"\n{'='*60}")
        print(f"VALIDATION at Episode {episode}")
        print(f"{'='*60}")

        map_types = ["empty", "random", "room", "corridor", "cave", "lshape"]
        results = {}

        for map_type in map_types:
            coverages = []
            for _ in range(config_improved.VALIDATION_EPISODES):
                env = ImprovedCoverageEnvironment(self.grid_size, map_type)
                coverage = self._run_validation_episode(env)
                coverages.append(coverage)

            avg_coverage = np.mean(coverages)
            std_coverage = np.std(coverages)
            results[map_type] = (avg_coverage, std_coverage)

            print(f"  {map_type:10s}: {avg_coverage:.2%} ± {std_coverage:.2%}")

        print(f"{'='*60}\n")

        return results

    def _run_validation_episode(self, env) -> float:
        """Run single validation episode (greedy)."""
        state = env.reset()

        for step in range(config_improved.MAX_EPISODE_STEPS):
            # Greedy action (no exploration)
            action = self.agent.select_action(state, env.world_state, epsilon=0.0)
            next_state, reward, done, info = env.step(action)
            state = next_state

            if done:
                break

        return info['coverage_pct']

    def _save_checkpoint(self, episode: int):
        """Save checkpoint."""
        checkpoint_path = f"{config_improved.CHECKPOINT_DIR}/checkpoint_ep{episode}.pt"
        save_checkpoint(
            checkpoint_path,
            self.agent.policy_net,
            self.agent.optimizer,
            episode,
            {
                'episode_rewards': self.episode_rewards,
                'episode_coverages': self.episode_coverages,
                'episode_losses': self.episode_losses,
                'phase_changes': self.phase_changes
            }
        )
        print(f"✓ Checkpoint saved: {checkpoint_path}")

    def _print_summary(self, total_time: float):
        """Print training summary."""
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE!")
        print("=" * 80)

        final_coverage = np.mean(self.episode_coverages[-100:]) if len(self.episode_coverages) >= 100 else 0.0
        final_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else 0.0

        print(f"\nTime: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        print(f"Episodes: {self.total_episodes}")
        print(f"Final Coverage (last 100 eps): {final_coverage:.2%}")
        print(f"Final Reward (last 100 eps): {final_reward:.2f}")
        print(f"Best Coverage: {max(self.episode_coverages):.2%}")

        print(f"\nPhase Progression:")
        for ep, phase_name in self.phase_changes:
            print(f"  Episode {ep:4d}: {phase_name}")

        print("\n" + "=" * 80)


def main():
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(description="Improved GAT-MARL Coverage Training")
    parser.add_argument('--episodes', type=int, default=2000, help='Total episodes')
    parser.add_argument('--grid_size', type=int, default=20, help='Grid size')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Create trainer
    trainer = ImprovedTrainer(
        grid_size=args.grid_size,
        total_episodes=args.episodes,
        device=args.device
    )

    # Train!
    trainer.train()

    # Plot results
    print("\nGenerating plots...")
    plot_training_curves(
        trainer.episode_rewards,
        trainer.episode_coverages,
        trainer.episode_losses,
        save_path=f"{config_improved.RESULTS_DIR}/training_curves_improved.png"
    )
    print(f"✓ Plots saved to {config_improved.RESULTS_DIR}/")


if __name__ == "__main__":
    main()