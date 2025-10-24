"""
Enhanced Main Entry Point for GAT-MARL Coverage

Command-line interface for training and evaluation using Enhanced Architecture.
Uses EnhancedCoverageAgent with all Phase 1 improvements.
"""

import argparse
import os
import sys

from config import config, print_config
from train_enhanced import train_stage1_enhanced, validate_enhanced, test_grid_size_generalization_enhanced
from agent_enhanced import EnhancedCoverageAgent
from utils import (plot_training_curves, plot_validation_results,
                  save_metrics, load_metrics, print_statistics)


def main():
    parser = argparse.ArgumentParser(
        description='GAT-MARL for Multi-Robot Coverage (Enhanced Architecture)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Enhanced Architecture Examples:
  # Train Stage 1 with Enhanced Architecture
  python main_enhanced.py --mode train --episodes 1600

  # Resume enhanced training
  python main_enhanced.py --mode train --resume checkpoints/enhanced_checkpoint_ep500.pt

  # Validate trained enhanced model
  python main_enhanced.py --mode validate --checkpoint checkpoints/enhanced_final_model.pt

  # Test grid-size generalization (enhanced)
  python main_enhanced.py --mode test_generalization --checkpoint checkpoints/enhanced_final_model.pt

  # Plot enhanced results
  python main_enhanced.py --mode plot --metrics results/enhanced_metrics.pkl

Enhanced Features:
  ✅ 10D node features + 3D edge features
  ✅ Adaptive virtual node (context-dependent)
  ✅ Recurrent state encoder (POMDP memory)
  ✅ Enhanced GAT with edge attention
  ✅ AGC gradient clipping
        """
    )

    # Mode
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'validate', 'test_generalization', 'plot'],
                       help='Operating mode')

    # Training args
    parser.add_argument('--episodes', type=int, default=1600,
                       help='Number of episodes to train')
    parser.add_argument('--grid_size', type=int, default=20,
                       help='Grid size')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to enhanced checkpoint to resume from')

    # Validation args
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to enhanced checkpoint for validation/testing')
    parser.add_argument('--val_episodes', type=int, default=10,
                       help='Episodes per map type for validation')

    # Generalization test args
    parser.add_argument('--test_sizes', type=str, default='15,20,25,30,40',
                       help='Comma-separated grid sizes to test')

    # Plotting args
    parser.add_argument('--metrics', type=str, default=None,
                       help='Path to enhanced metrics file for plotting')

    # General args
    parser.add_argument('--output_dir', type=str, default='results_enhanced',
                       help='Output directory for enhanced results')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--probabilistic', action='store_true',
                       help='Use probabilistic coverage environment (sigmoid function)')

    args = parser.parse_args()
    
    # Set probabilistic environment if specified
    if args.probabilistic:
        config.USE_PROBABILISTIC_ENV = True

    # Set random seed
    import random
    import numpy as np
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Print config
    if args.verbose:
        print("=" * 80)
        print("ENHANCED GAT-MARL COVERAGE SYSTEM")
        print("=" * 80)
        print_config()
        print("\n🚀 ENHANCED ARCHITECTURE FEATURES:")
        print("  ✅ EnhancedCoverageAgent with recurrent memory")
        print("  ✅ 10D node features + 3D edge features")
        print("  ✅ Adaptive virtual node (context-dependent)")
        print("  ✅ RecurrentStateEncoder for POMDP memory")
        print("  ✅ Enhanced GAT with edge attention")
        print("  ✅ AGC gradient clipping")
        print("=" * 80)
        print()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Execute mode
    if args.mode == 'train':
        run_enhanced_training(args)

    elif args.mode == 'validate':
        run_enhanced_validation(args)

    elif args.mode == 'test_generalization':
        run_enhanced_generalization_test(args)

    elif args.mode == 'plot':
        run_enhanced_plotting(args)


def run_enhanced_training(args):
    """Run enhanced training."""
    print("=" * 80)
    print("STARTING ENHANCED TRAINING")
    print("=" * 80)
    print(f"Episodes: {args.episodes}")
    print(f"Grid size: {args.grid_size}")
    print(f"Architecture: Enhanced (Phase 1 improvements)")
    if args.resume:
        print(f"Resuming from: {args.resume}")
    print("=" * 80)
    print()

    # Train with enhanced architecture
    agent, metrics = train_stage1_enhanced(
        num_episodes=args.episodes,
        grid_size=args.grid_size,
        validate_interval=config.VALIDATION_INTERVAL,
        checkpoint_interval=config.CHECKPOINT_INTERVAL,
        resume_from=args.resume,
        verbose=args.verbose
    )

    # Save enhanced metrics
    metrics_path = os.path.join(args.output_dir, 'enhanced_training_metrics.pkl')
    save_metrics(metrics, metrics_path)

    # Plot training curves
    curves_path = os.path.join(args.output_dir, 'enhanced_training_curves.png')
    plot_training_curves(metrics, save_path=curves_path, show=False)

    # Plot validation results
    if len(metrics.validation_scores) > 0:
        val_path = os.path.join(args.output_dir, 'enhanced_validation_results.png')
        plot_validation_results(metrics, save_path=val_path, show=False)

    # Print statistics
    print_statistics(metrics)

    # Print enhanced architecture info
    num_params = sum(p.numel() for p in agent.policy_net.parameters())
    print(f"\n🚀 ENHANCED ARCHITECTURE SUMMARY:")
    print(f"  Agent type: {type(agent).__name__}")
    print(f"  Network type: {type(agent.policy_net).__name__}")
    print(f"  Total parameters: {num_params:,}")
    print(f"  Has recurrent memory: {hasattr(agent, 'reset_memory')}")
    print(f"  Graph encoder: {type(agent.graph_encoder).__name__}")

    print("\n✓ Enhanced training complete!")
    print(f"  Metrics saved: {metrics_path}")
    print(f"  Plots saved: {args.output_dir}/")


def run_enhanced_validation(args):
    """Run enhanced validation."""
    if args.checkpoint is None:
        print("Error: --checkpoint required for validation")
        sys.exit(1)

    print("=" * 80)
    print("ENHANCED VALIDATION")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Grid size: {args.grid_size}")
    print(f"Episodes per map: {args.val_episodes}")
    print(f"Architecture: Enhanced")
    print("=" * 80)
    print()

    # Load enhanced agent
    agent = EnhancedCoverageAgent(grid_size=args.grid_size)
    episode, metrics = agent.load_checkpoint(args.checkpoint)

    print(f"✓ Loaded enhanced checkpoint from episode {episode}")
    print(f"  Agent type: {type(agent).__name__}")
    print(f"  Parameters: {sum(p.numel() for p in agent.policy_net.parameters()):,}")
    print()

    # Validate
    results = validate_enhanced(agent, args.grid_size, args.val_episodes)

    # Print results
    print("\nEnhanced Validation Results:")
    print("-" * 40)
    for map_type, coverage in results.items():
        print(f"  {map_type:12s}: {coverage*100:5.1f}%")
    print("-" * 40)
    avg_coverage = sum(results.values()) / len(results)
    print(f"  {'Average':12s}: {avg_coverage*100:5.1f}%")
    print("-" * 40)


def run_enhanced_generalization_test(args):
    """Run enhanced grid-size generalization test."""
    if args.checkpoint is None:
        print("Error: --checkpoint required for generalization test")
        sys.exit(1)

    test_sizes = [int(s.strip()) for s in args.test_sizes.split(',')]

    print("=" * 80)
    print("ENHANCED GRID-SIZE GENERALIZATION TEST")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test sizes: {test_sizes}")
    print(f"Architecture: Enhanced")
    print("=" * 80)
    print()

    # Load enhanced agent
    agent = EnhancedCoverageAgent(grid_size=args.grid_size)
    episode, metrics = agent.load_checkpoint(args.checkpoint)

    print(f"✓ Loaded enhanced checkpoint from episode {episode}")
    print(f"  Agent type: {type(agent).__name__}")
    print()

    # Test generalization
    print("Testing enhanced generalization...")
    results = test_grid_size_generalization_enhanced(agent, test_sizes, num_episodes=args.val_episodes)

    # Print summary
    print("\n" + "=" * 80)
    print("ENHANCED GENERALIZATION RESULTS")
    print("=" * 80)
    for size, coverage in results.items():
        trained_marker = " (trained)" if size == args.grid_size else ""
        print(f"  {size:2d}x{size:2d}: {coverage*100:5.1f}%{trained_marker}")
    print("=" * 80)


def run_enhanced_plotting(args):
    """Plot enhanced results from metrics file."""
    if args.metrics is None:
        print("Error: --metrics required for plotting")
        sys.exit(1)

    print("=" * 80)
    print("PLOTTING ENHANCED RESULTS")
    print("=" * 80)
    print(f"Metrics file: {args.metrics}")
    print(f"Architecture: Enhanced")
    print("=" * 80)
    print()

    # Load enhanced metrics
    metrics = load_metrics(args.metrics)

    # Plot training curves
    curves_path = os.path.join(args.output_dir, 'enhanced_training_curves.png')
    plot_training_curves(metrics, save_path=curves_path, show=True)

    # Plot validation results
    if len(metrics.validation_scores) > 0:
        val_path = os.path.join(args.output_dir, 'enhanced_validation_results.png')
        plot_validation_results(metrics, save_path=val_path, show=True)

    # Print statistics
    print_statistics(metrics)

    print(f"\n✓ Enhanced plots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()