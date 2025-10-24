"""
Enhanced Training Loop for Stage 1

Single-agent training with curriculum learning using Enhanced Architecture.
Uses EnhancedCoverageAgent with all Phase 1 improvements.
"""

import os
import time
from typing import Optional
import numpy as np

from config import config
from data_structures import CoverageMetrics
from environment import CoverageEnvironment
from environment_probabilistic import ProbabilisticCoverageEnvironment
from agent_enhanced import EnhancedCoverageAgent
from curriculum import CurriculumManager


def train_stage1_enhanced(num_episodes: int = 1600,
                         grid_size: int = 20,
                         validate_interval: int = 50,
                         checkpoint_interval: int = 100,
                         resume_from: Optional[str] = None,
                         verbose: bool = True) -> tuple:
    """
    Train Stage 1: Single-agent mastery with curriculum learning (Enhanced Architecture).

    Uses Enhanced Architecture:
    - EnhancedCoverageAgent with recurrent memory
    - EnhancedGATCoverageDQN with adaptive virtual node
    - 10D node features + 3D edge features
    - RecurrentStateEncoder for POMDP memory

    Args:
        num_episodes: Number of episodes to train
        grid_size: Map size
        validate_interval: Validate every N episodes
        checkpoint_interval: Save checkpoint every N episodes
        resume_from: Path to checkpoint to resume from
        verbose: Print training progress

    Returns:
        agent: Trained enhanced agent
        metrics: Training metrics
    """
    if verbose:
        print("=" * 80)
        print("STAGE 1: ENHANCED SINGLE-AGENT MASTERY TRAINING")
        print("=" * 80)
        print(f"Episodes: {num_episodes}")
        print(f"Grid size: {grid_size}")
        print(f"Device: {config.DEVICE}")
        print(f"Architecture: Enhanced (Phase 1 improvements)")
        print(f"Features: 10D nodes + 3D edges + recurrent memory")
        if config.USE_PROBABILISTIC_ENV:
            print(f"Environment: PROBABILISTIC (sigmoid coverage)")
        else:
            print(f"Environment: BINARY (instant coverage)")
        print("=" * 80)

    # Initialize components with enhanced agent
    agent = EnhancedCoverageAgent(grid_size=grid_size)
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

        # Get epsilon floor from curriculum (minimum exploration)
        epsilon_floor = curriculum.get_epsilon_floor(episode)
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
        last_print_step = 0  # Track for progress printing

        # Timing diagnostics (optional - remove for production)
        time_encoding = 0
        time_action = 0
        time_env = 0
        time_train = 0

        # OPTIMIZATION: Cache graph encoding to avoid redundant encoding
        # Encode initial state once
        cached_graph_data = agent.graph_encoder.encode(state, env.world_state, agent_idx=0)

        for step in range(config.MAX_EPISODE_STEPS):
            # Show progress for slow episodes (every 50 steps)
            if verbose and step > 0 and step % 50 == 0 and step != last_print_step:
                print(f"  Ep {episode+1} Step {step}/{config.MAX_EPISODE_STEPS} | "
                      f"Cov: {env.get_coverage_percentage()*100:.1f}% | "
                      f"Mem: {len(agent.memory)} | "
                      f"Time/step: {(time.time()-episode_start_time)/step:.3f}s", end='\r')
                last_print_step = step

            # OPTIMIZATION: Reuse cached encoding from previous step
            t0 = time.time()
            graph_data = cached_graph_data  # No encoding needed!
            time_encoding += time.time() - t0

            # Select action (OPTIMIZED: pass pre-encoded graph to avoid redundant encoding)
            t0 = time.time()
            if step == 0:
                action = agent.select_action_from_graph(graph_data, reset_memory=True)
            else:
                action = agent.select_action_from_graph(graph_data, reset_memory=False)
            time_action += time.time() - t0

            # Step environment
            t0 = time.time()
            next_state, reward, done, info = env.step(action)
            time_env += time.time() - t0
            episode_reward += reward

            # Encode next state (this becomes cached for next iteration)
            next_graph_data = agent.graph_encoder.encode(next_state, env.world_state, agent_idx=0)
            agent.store_transition(graph_data, action, reward, next_graph_data, done, info)

            # Train every TRAIN_FREQ steps (not every step for efficiency)
            if len(agent.memory) >= config.MIN_REPLAY_SIZE and step % config.TRAIN_FREQ == 0:
                t0 = time.time()
                loss = agent.optimize()
                time_train += time.time() - t0
                if loss is not None:
                    episode_loss.append(loss)

            # OPTIMIZATION: Cache next encoding for reuse
            state = next_state
            cached_graph_data = next_graph_data

            if done:
                break

        # Clear progress line
        if verbose and last_print_step > 0:
            print(" " * 100, end='\r')  # Clear the progress line

        # Episode complete
        episode_time = time.time() - episode_start_time
        coverage_pct = env.get_coverage_percentage()
        
        # Print timing breakdown for first few episodes to diagnose bottlenecks
        if verbose and episode < 5:
            total_steps = step + 1
            print(f"\n  ⏱️  Episode {episode+1} timing breakdown:")
            print(f"     Total: {episode_time:.2f}s ({total_steps} steps, {episode_time/total_steps:.3f}s/step)")
            print(f"     Encoding: {time_encoding:.2f}s ({time_encoding/episode_time*100:.1f}%)")
            print(f"     Action:   {time_action:.2f}s ({time_action/episode_time*100:.1f}%)")
            print(f"     Env Step: {time_env:.2f}s ({time_env/episode_time*100:.1f}%)")
            print(f"     Training: {time_train:.2f}s ({time_train/episode_time*100:.1f}%)")

        # Update metrics
        metrics.add_episode(episode_reward, coverage_pct, step + 1, agent.epsilon)
        if episode_loss:
            avg_loss = sum(episode_loss) / len(episode_loss)
            metrics.add_loss(avg_loss)

        # Decay epsilon and learning rate
        agent.update_epsilon()
        agent.update_learning_rate()

        # Update target network
        if (episode + 1) % config.TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        # Get gradient stats for enhanced logging
        grad_stats = agent.get_grad_stats()
        if len(agent.grad_norm_history) > 0:
            metrics.add_grad_norm(agent.grad_norm_history[-1])

        # Enhanced progress logging (every 10 episodes)
        if verbose and (episode + 1) % config.LOG_INTERVAL == 0:
            phase = curriculum.get_current_phase(episode)
            recent_reward = metrics.get_recent_avg('reward', window=10)
            recent_coverage = metrics.get_recent_avg('coverage', window=10)
            recent_length = metrics.get_recent_avg('length', window=10)
            avg_reward = metrics.get_recent_avg('reward', window=100)
            avg_coverage = metrics.get_recent_avg('coverage', window=100)
            avg_loss = metrics.get_recent_avg('loss', window=100)
            current_lr = agent.get_learning_rate()
            
            print(f"Ep {episode+1:4d}/{num_episodes} | "
                  f"{phase.name[:25]:25s} | "
                  f"Map: {map_type:8s} | "
                  f"Cov: {coverage_pct*100:5.1f}% (μ={avg_coverage*100:5.1f}%) | "
                  f"R: {episode_reward:7.2f} (μ={avg_reward:7.2f}) | "
                  f"L: {step+1:3d} (μ={recent_length:5.1f}) | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"LR: {current_lr:.1e} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Grad: {grad_stats['mean']:.2f} | "
                  f"Mem: {len(agent.memory):,} | "
                  f"T: {episode_time:.1f}s")
            
            # Enhanced architecture specific info (every 50 episodes)
            if (episode + 1) % 50 == 0:
                print(f"     🚀 Enhanced Features: "
                      f"Recurrent Memory ✓ | "
                      f"Edge Features ✓ | "
                      f"Adaptive VN ✓ | "
                      f"Params: {sum(p.numel() for p in agent.policy_net.parameters()):,}")

        # Validation
        if (episode + 1) % validate_interval == 0:
            if verbose:
                print(f"\n{'='*60}")
                print(f"VALIDATION AT EPISODE {episode + 1}")
                print(f"{'='*60}")
            
            val_results = validate_enhanced(agent, grid_size, num_val_episodes=10)
            metrics.validation_scores[episode + 1] = val_results
            
            if verbose:
                avg_val_coverage = sum(val_results.values()) / len(val_results)
                print(f"Average validation coverage: {avg_val_coverage*100:.1f}%")
                print("-" * 40)
                for map_type, coverage in val_results.items():
                    status = "✓" if coverage > 0.7 else "○"
                    print(f"  {status} {map_type:12s}: {coverage*100:5.1f}%")
                print("-" * 40)
                
                # Training progress summary
                total_episodes = len(metrics.episode_rewards)
                if total_episodes >= 10:
                    recent_improvement = (metrics.get_recent_avg('coverage', 10) - 
                                        metrics.get_recent_avg('coverage', min(50, total_episodes))) * 100
                    print(f"Recent improvement: {recent_improvement:+.1f}% coverage")
                
                print(f"Memory usage: {len(agent.memory):,}/{agent.memory.capacity:,}")
                memory_stats = agent.memory.get_stats()
                print(f"Memory distribution: C:{memory_stats['coverage']} "
                      f"E:{memory_stats['exploration']} F:{memory_stats['failure']} "
                      f"N:{memory_stats['neutral']}")
                print(f"{'='*60}")
                print()

            # Check curriculum advancement
            avg_coverage = metrics.get_recent_avg('coverage', window=50)
            if curriculum.should_advance(episode, avg_coverage):
                next_phase = curriculum.get_current_phase(episode + 1)
                if verbose:
                    print(f"✓ Advancing to {next_phase.name}")

        # Checkpoint
        if (episode + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"enhanced_checkpoint_ep{episode+1}.pt")
            agent.save_checkpoint(checkpoint_path, episode + 1, metrics)
            if verbose:
                print(f"✓ Checkpoint saved: {checkpoint_path}")

    # Training complete
    if verbose:
        print("\n" + "=" * 80)
        print("ENHANCED TRAINING COMPLETE")
        print("=" * 80)
        print(f"Total episodes: {num_episodes}")
        print(f"Final coverage: {metrics.episode_coverages[-1]*100:.1f}%")
        print(f"Final reward: {metrics.episode_rewards[-1]:.1f}")
        print(f"Memory size: {len(agent.memory):,}")
        print("=" * 80)

    # Save final model
    final_path = os.path.join(config.CHECKPOINT_DIR, "enhanced_final_model.pt")
    agent.save_checkpoint(final_path, num_episodes, metrics)

    return agent, metrics


def validate_enhanced(agent: EnhancedCoverageAgent,
                     grid_size: int = 20,
                     num_val_episodes: int = 10) -> dict:
    """
    Validate enhanced agent on all map types.

    Args:
        agent: Enhanced agent to validate
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
        
        for ep in range(num_val_episodes):
            # Use same environment type as training
            if config.USE_PROBABILISTIC_ENV:
                env = ProbabilisticCoverageEnvironment(grid_size=grid_size, map_type=map_type)
            else:
                env = CoverageEnvironment(grid_size=grid_size, map_type=map_type)
            state = env.reset()
            
            # Reset memory for validation episode
            agent.reset_memory()
            
            for step in range(config.MAX_EPISODE_STEPS):
                # Reset memory on first step
                reset_mem = (step == 0)
                action = agent.select_action(state, env.world_state, reset_memory=reset_mem)
                next_state, reward, done, info = env.step(action)
                state = next_state
                
                if done:
                    break
            
            coverages.append(env.get_coverage_percentage())
        
        results[map_type] = sum(coverages) / len(coverages)

    # Restore epsilon
    agent.set_epsilon(original_epsilon)

    return results


def test_grid_size_generalization_enhanced(agent: EnhancedCoverageAgent,
                                          test_sizes: list = [15, 20, 25, 30, 40],
                                          num_episodes: int = 10) -> dict:
    """
    Test enhanced agent generalization across different grid sizes.

    Args:
        agent: Trained enhanced agent
        test_sizes: List of grid sizes to test
        num_episodes: Episodes per size

    Returns:
        results: Dict of {grid_size: avg_coverage}
    """
    results = {}

    # Save original epsilon and grid size
    original_epsilon = agent.epsilon
    original_grid_size = agent.grid_size
    agent.set_epsilon(0.05)  # Mostly greedy for evaluation

    for size in test_sizes:
        print(f"Testing grid size {size}x{size}...")
        
        # Update agent's grid size
        agent.grid_size = size
        agent.graph_encoder.grid_size = size
        
        coverages = []
        
        for ep in range(num_episodes):
            # Use same environment type as training
            if config.USE_PROBABILISTIC_ENV:
                env = ProbabilisticCoverageEnvironment(grid_size=size, map_type="random")
            else:
                env = CoverageEnvironment(grid_size=size, map_type="random")
            state = env.reset()
            
            # Reset memory for test episode
            agent.reset_memory()
            
            for step in range(config.MAX_EPISODE_STEPS):
                reset_mem = (step == 0)
                action = agent.select_action(state, env.world_state, reset_memory=reset_mem)
                next_state, reward, done, info = env.step(action)
                state = next_state
                
                if done:
                    break
            
            coverages.append(env.get_coverage_percentage())
        
        results[size] = sum(coverages) / len(coverages)

    # Restore
    agent.set_epsilon(original_epsilon)
    agent.grid_size = original_grid_size
    agent.graph_encoder.grid_size = original_grid_size

    return results


if __name__ == "__main__":
    # Test enhanced training loop (10 episodes)
    print("Testing enhanced training loop (10 episodes)...")
    agent, metrics = train_stage1_enhanced(num_episodes=10, verbose=True)
    
    print(f"\n✓ Enhanced training test complete")
    print(f"  Final agent type: {type(agent).__name__}")
    print(f"  Network type: {type(agent.policy_net).__name__}")
    print(f"  Parameters: {sum(p.numel() for p in agent.policy_net.parameters()):,}")
    print(f"  Has recurrent memory: {hasattr(agent, 'reset_memory')}")