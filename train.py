"""
Training Loop for Stage 1

Single-agent training with curriculum learning.
"""

import os
import time
from typing import Optional
import numpy as np

from config import config
from data_structures import CoverageMetrics
from environment import CoverageEnvironment
from environment_probabilistic import ProbabilisticCoverageEnvironment
from agent import CoverageAgent
from curriculum import CurriculumManager


def train_stage1(num_episodes: int = 1600,
                 grid_size: int = 20,
                 validate_interval: int = 50,
                 checkpoint_interval: int = 100,
                 resume_from: Optional[str] = None,
                 verbose: bool = True) -> tuple:
    """
    Train Stage 1: Single-agent mastery with curriculum learning.

    Args:
        num_episodes: Number of episodes to train
        grid_size: Map size
        validate_interval: Validate every N episodes
        checkpoint_interval: Save checkpoint every N episodes
        resume_from: Path to checkpoint to resume from
        verbose: Print training progress

    Returns:
        agent: Trained agent
        metrics: Training metrics
    """
    if verbose:
        print("=" * 80)
        print("STAGE 1: SINGLE-AGENT MASTERY TRAINING")
        print("=" * 80)
        print(f"Episodes: {num_episodes}")
        print(f"Grid size: {grid_size}")
        print(f"Device: {config.DEVICE}")
        if config.USE_PROBABILISTIC_ENV:
            print(f"Environment: PROBABILISTIC (sigmoid coverage)")
        else:
            print(f"Environment: BINARY (instant coverage)")
        print("=" * 80)

    # Initialize components
    agent = CoverageAgent(grid_size=grid_size)
    curriculum = CurriculumManager()
    metrics = CoverageMetrics()

    # Create checkpoint directory
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Resume from checkpoint if specified
    start_episode = 0
    if resume_from is not None:
        start_episode, loaded_metrics = agent.load_checkpoint(resume_from)
        metrics = loaded_metrics
        if verbose:
            print(f"✓ Resumed from episode {start_episode}")

    # Print curriculum overview
    if verbose and start_episode == 0:
        print("\n" + curriculum.get_summary())
        print()

    # Training loop
    for episode in range(start_episode, num_episodes):
        episode_start_time = time.time()

        # Get map type from curriculum
        map_type = curriculum.get_map_type(episode)

        # Get phase-specific epsilon parameters from curriculum
        epsilon_floor = curriculum.get_epsilon_floor(episode)
        epsilon_decay = curriculum.get_epsilon_decay(episode)
        
        # Enforce epsilon floor (minimum exploration)
        if agent.epsilon < epsilon_floor:
            agent.set_epsilon(epsilon_floor)

        # Create environment (probabilistic or binary based on config)
        if config.USE_PROBABILISTIC_ENV:
            env = ProbabilisticCoverageEnvironment(grid_size=grid_size, map_type=map_type)
        else:
            env = CoverageEnvironment(grid_size=grid_size, map_type=map_type)
        state = env.reset()

        # Episode loop
        episode_reward = 0
        episode_loss = []
        
        # Timing breakdown (only for first few episodes to diagnose bottlenecks)
        enable_timing = config.ENABLE_TIMING_BREAKDOWN and episode < 5
        time_encoding = 0 if enable_timing else None
        time_action = 0 if enable_timing else None
        time_env = 0 if enable_timing else None
        time_train = 0 if enable_timing else None

        for step in range(config.MAX_EPISODE_STEPS):
            # Encode state to graph
            if enable_timing:
                t0 = time.time()
            graph_data = agent.graph_encoder.encode(state, env.world_state, agent_idx=0)
            if enable_timing:
                time_encoding += time.time() - t0

            # Select action
            if enable_timing:
                t0 = time.time()
            action = agent.select_action(state, env.world_state)
            if enable_timing:
                time_action += time.time() - t0

            # Step environment
            if enable_timing:
                t0 = time.time()
            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            # Encode next state
            next_graph_data = agent.graph_encoder.encode(next_state, env.world_state, agent_idx=0)
            if enable_timing:
                time_env += time.time() - t0

            # Store transition
            agent.store_transition(graph_data, action, reward, next_graph_data, done, info)

            # Optimize every TRAIN_FREQ steps (not every step for efficiency)
            if step % config.TRAIN_FREQ == 0:
                if enable_timing:
                    t0 = time.time()
                loss = agent.optimize()
                if enable_timing:
                    time_train += time.time() - t0
                if loss is not None:
                    episode_loss.append(loss)
                    metrics.add_loss(loss)

            state = next_state

            if done:
                break

        # Episode complete
        coverage_pct = env.get_coverage_percentage()
        episode_length = step + 1

        # Update metrics
        metrics.add_episode(episode_reward, coverage_pct, episode_length, agent.epsilon)

        # Update target network
        if (episode + 1) % config.TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        # Update epsilon with phase-specific decay rate
        agent.update_epsilon(decay_rate=epsilon_decay, min_epsilon=epsilon_floor)
        agent.update_learning_rate()

        # Get gradient stats (only when needed for logging)
        if verbose and (episode + 1) % config.LOG_INTERVAL == 0:
            grad_stats = agent.get_grad_stats()
            if len(agent.grad_norm_history) > 0:
                metrics.add_grad_norm(agent.grad_norm_history[-1])
        else:
            grad_stats = None

        # Logging
        if verbose and (episode + 1) % config.LOG_INTERVAL == 0:
            phase = curriculum.get_current_phase(episode)
            avg_reward = metrics.get_recent_avg('reward', window=100)
            avg_coverage = metrics.get_recent_avg('coverage', window=100)
            avg_loss = metrics.get_recent_avg('loss', window=100)
            current_lr = agent.get_learning_rate()

            episode_time = time.time() - episode_start_time

            print(f"Ep {episode+1:4d}/{num_episodes} | "
                  f"{phase.name:30s} | "
                  f"Map: {map_type:8s} | "
                  f"Cov: {coverage_pct*100:5.1f}% (μ={avg_coverage*100:5.1f}%) | "
                  f"R: {episode_reward:7.2f} (μ={avg_reward:7.2f}) | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"LR: {current_lr:.1e} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Grad: {grad_stats['mean']:.2f} | "
                  f"Time: {episode_time:.2f}s")
            
            # Print timing breakdown for first 5 episodes to diagnose bottlenecks
            if config.ENABLE_TIMING_BREAKDOWN and episode < 5:
                total_time = time_encoding + time_action + time_env + time_train
                per_step_time = total_time / episode_length if episode_length > 0 else 0
                print(f"  └─ Timing breakdown: Total {total_time:.1f}s ({episode_length} steps, {per_step_time:.3f}s/step)")
                if total_time > 0:
                    print(f"     Encoding: {time_encoding:.2f}s ({time_encoding/total_time*100:.0f}%), "
                          f"Action: {time_action:.2f}s ({time_action/total_time*100:.0f}%), "
                          f"Env: {time_env:.2f}s ({time_env/total_time*100:.0f}%), "
                          f"Training: {time_train:.2f}s ({time_train/total_time*100:.0f}%)")


        # Validation
        if (episode + 1) % validate_interval == 0:
            val_results = validate(agent, grid_size, num_val_episodes=config.VALIDATION_EPISODES)
            metrics.validation_scores[episode + 1] = val_results

            if verbose:
                print(f"\n{'='*80}")
                print(f"VALIDATION @ Episode {episode+1}")
                print(f"{'='*80}")
                for map_type, coverage in val_results.items():
                    print(f"  {map_type:10s}: {coverage*100:5.1f}%")
                print(f"  {'Average':10s}: {np.mean(list(val_results.values()))*100:5.1f}%")
                print(f"{'='*80}\n")

            # Check if should advance curriculum phase
            avg_coverage = metrics.get_recent_avg('coverage', window=50)
            if curriculum.should_advance(episode, avg_coverage):
                next_phase = curriculum.get_current_phase(episode + 1)
                if verbose:
                    print(f"✓ Advancing to {next_phase.name}")

        # Checkpoint
        if (episode + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_ep{episode+1}.pt")
            agent.save_checkpoint(checkpoint_path, episode + 1, metrics)
            if verbose:
                print(f"✓ Checkpoint saved: {checkpoint_path}")

    # Training complete
    if verbose:
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Total episodes: {num_episodes}")
        print(f"Final epsilon: {agent.epsilon:.3f}")
        print(f"Final avg coverage: {metrics.get_recent_avg('coverage', window=100)*100:.1f}%")
        print(f"Gradient explosions: {metrics.grad_explosions}")
        print("=" * 80)

    # Save final model
    final_path = os.path.join(config.CHECKPOINT_DIR, "final_model.pt")
    agent.save_checkpoint(final_path, num_episodes, metrics)

    return agent, metrics


def validate(agent: CoverageAgent,
            grid_size: int = 20,
            num_val_episodes: int = 10) -> dict:
    """
    Validate agent on all map types.

    Args:
        agent: Agent to validate
        grid_size: Map size
        num_val_episodes: Number of episodes per map type

    Returns:
        results: Dict of {map_type: avg_coverage}
    """
    map_types = ["empty", "random", "room", "corridor", "cave", "lshape"]
    results = {}

    # Save current epsilon
    original_epsilon = agent.epsilon

    # Evaluate with low epsilon (mostly greedy)
    agent.set_epsilon(0.05)

    for map_type in map_types:
        coverages = []

        for _ in range(num_val_episodes):
            # Use same environment type as training
            if config.USE_PROBABILISTIC_ENV:
                env = ProbabilisticCoverageEnvironment(grid_size=grid_size, map_type=map_type)
            else:
                env = CoverageEnvironment(grid_size=grid_size, map_type=map_type)
            state = env.reset()

            # Use reduced steps for faster validation if configured
            max_steps = config.VALIDATION_MAX_STEPS if config.FAST_VALIDATION else config.MAX_EPISODE_STEPS
            
            for _ in range(max_steps):
                # Select action (mostly greedy)
                action = agent.select_action(state, env.world_state, epsilon=0.05)

                # Step
                next_state, reward, done, info = env.step(action)
                state = next_state

                if done:
                    break

            # Record coverage
            coverages.append(env.get_coverage_percentage())

        # Average coverage for this map type
        results[map_type] = np.mean(coverages)

    # Restore epsilon
    agent.set_epsilon(original_epsilon)

    return results


def test_grid_size_generalization(agent: CoverageAgent,
                                  test_sizes: list = [15, 20, 25, 30, 40],
                                  num_episodes: int = 10) -> dict:
    """
    Test agent generalization across different grid sizes.

    Args:
        agent: Trained agent
        test_sizes: List of grid sizes to test
        num_episodes: Episodes per size

    Returns:
        results: Dict of {grid_size: avg_coverage}
    """
    results = {}

    # Save original epsilon
    original_epsilon = agent.epsilon
    agent.set_epsilon(0.05)  # Mostly greedy for evaluation

    for size in test_sizes:
        coverages = []

        for _ in range(num_episodes):
            # Create environment with different grid size (use same type as training)
            if config.USE_PROBABILISTIC_ENV:
                env = ProbabilisticCoverageEnvironment(grid_size=size, map_type="room")
            else:
                env = CoverageEnvironment(grid_size=size, map_type="room")
            state = env.reset()

            # Update agent's graph encoder for new size
            agent.graph_encoder.grid_size = size

            for _ in range(config.MAX_EPISODE_STEPS):
                action = agent.select_action(state, env.world_state, epsilon=0.05)
                next_state, reward, done, info = env.step(action)
                state = next_state

                if done:
                    break

            coverages.append(env.get_coverage_percentage())

        results[size] = np.mean(coverages)

        print(f"Grid size {size:2d}x{size:2d}: {results[size]*100:.1f}% coverage")

    # Restore
    agent.set_epsilon(original_epsilon)
    agent.graph_encoder.grid_size = config.GRID_SIZE

    return results


if __name__ == "__main__":
    # Quick test run (10 episodes)
    print("Testing training loop (10 episodes)...")

    agent, metrics = train_stage1(
        num_episodes=10,
        grid_size=20,
        validate_interval=5,
        checkpoint_interval=5,
        verbose=True
    )

    print("\n✓ Training test complete")
    print(f"  Episodes trained: {len(metrics.episode_rewards)}")
    print(f"  Final reward: {metrics.episode_rewards[-1]:.2f}")
    print(f"  Final coverage: {metrics.episode_coverages[-1]*100:.1f}%")
