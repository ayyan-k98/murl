"""
Main Entry Point for GAT-MARL Coverage

Command-line interface for training and evaluation.
"""

import argparse
import os
import sys

from config import config, print_config
from train import train_stage1, validate, test_grid_size_generalization
from agent import CoverageAgent
from utils import (plot_training_curves, plot_validation_results,
                  save_metrics, load_metrics, print_statistics)


def main():
    parser = argparse.ArgumentParser(
        description='GAT-MARL for Multi-Robot Coverage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train Stage 1
  python main.py --mode train --episodes 1600

  # Resume training
  python main.py --mode train --resume checkpoints/checkpoint_ep500.pt

  # Validate trained model
  python main.py --mode validate --checkpoint checkpoints/final_model.pt

  # Test grid-size generalization
  python main.py --mode test_generalization --checkpoint checkpoints/final_model.pt

  # Plot results
  python main.py --mode plot --metrics results/metrics.pkl
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
                       help='Path to checkpoint to resume from')

    # Validation args
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for validation/testing')
    parser.add_argument('--val_episodes', type=int, default=10,
                       help='Episodes per map type for validation')

    # Generalization test args
    parser.add_argument('--test_sizes', type=str, default='15,20,25,30,40',
                       help='Comma-separated grid sizes to test')

    # Plotting args
    parser.add_argument('--metrics', type=str, default=None,
                       help='Path to metrics file for plotting')

    # General args
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
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
        print_config()
        print()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Execute mode
    if args.mode == 'train':
        run_training(args)

    elif args.mode == 'validate':
        run_validation(args)

    elif args.mode == 'test_generalization':
        run_generalization_test(args)

    elif args.mode == 'plot':
        run_plotting(args)


def run_training(args):
    """Run training."""
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print(f"Episodes: {args.episodes}")
    print(f"Grid size: {args.grid_size}")
    if args.resume:
        print(f"Resuming from: {args.resume}")
    print("=" * 80)
    print()

    # Train
    agent, metrics = train_stage1(
        num_episodes=args.episodes,
        grid_size=args.grid_size,
        validate_interval=config.VALIDATION_INTERVAL,
        checkpoint_interval=config.CHECKPOINT_INTERVAL,
        resume_from=args.resume,
        verbose=args.verbose
    )

    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'training_metrics.pkl')
    save_metrics(metrics, metrics_path)

    # Plot training curves
    curves_path = os.path.join(args.output_dir, 'training_curves.png')
    plot_training_curves(metrics, save_path=curves_path, show=False)

    # Plot validation results
    if len(metrics.validation_scores) > 0:
        val_path = os.path.join(args.output_dir, 'validation_results.png')
        plot_validation_results(metrics, save_path=val_path, show=False)

    # Print statistics
    print_statistics(metrics)

    print("\n✓ Training complete!")
    print(f"  Metrics saved: {metrics_path}")
    print(f"  Plots saved: {args.output_dir}/")


def run_validation(args):
    """Run validation."""
    if args.checkpoint is None:
        print("Error: --checkpoint required for validation")
        sys.exit(1)

    print("=" * 80)
    print("VALIDATION")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Grid size: {args.grid_size}")
    print(f"Episodes per map: {args.val_episodes}")
    print("=" * 80)
    print()

    # Load agent
    agent = CoverageAgent(grid_size=args.grid_size)
    episode, metrics = agent.load_checkpoint(args.checkpoint)

    print(f"✓ Loaded checkpoint from episode {episode}")
    print()

    # Validate
    results = validate(agent, args.grid_size, args.val_episodes)

    # Print results
    print("\nValidation Results:")
    print("-" * 40)
    for map_type, coverage in results.items():
        print(f"  {map_type:12s}: {coverage*100:5.1f}%")
    print("-" * 40)
    avg_coverage = sum(results.values()) / len(results)
    print(f"  {'Average':12s}: {avg_coverage*100:5.1f}%")
    print("-" * 40)


def run_generalization_test(args):
    """Run grid-size generalization test."""
    if args.checkpoint is None:
        print("Error: --checkpoint required for generalization test")
        sys.exit(1)

    test_sizes = [int(s.strip()) for s in args.test_sizes.split(',')]

    print("=" * 80)
    print("GRID-SIZE GENERALIZATION TEST")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test sizes: {test_sizes}")
    print("=" * 80)
    print()

    # Load agent
    agent = CoverageAgent(grid_size=args.grid_size)
    episode, metrics = agent.load_checkpoint(args.checkpoint)

    print(f"✓ Loaded checkpoint from episode {episode}")
    print()

    # Test generalization
    print("Testing generalization...")
    results = test_grid_size_generalization(agent, test_sizes, num_episodes=args.val_episodes)

    # Print summary
    print("\n" + "=" * 80)
    print("GENERALIZATION RESULTS")
    print("=" * 80)
    for size, coverage in results.items():
        trained_marker = " (trained)" if size == args.grid_size else ""
        print(f"  {size:2d}x{size:2d}: {coverage*100:5.1f}%{trained_marker}")
    print("=" * 80)


def run_plotting(args):
    """Plot results from metrics file."""
    if args.metrics is None:
        print("Error: --metrics required for plotting")
        sys.exit(1)

    print("=" * 80)
    print("PLOTTING RESULTS")
    print("=" * 80)
    print(f"Metrics file: {args.metrics}")
    print("=" * 80)
    print()

    # Load metrics
    metrics = load_metrics(args.metrics)

    # Plot training curves
    curves_path = os.path.join(args.output_dir, 'training_curves.png')
    plot_training_curves(metrics, save_path=curves_path, show=True)

    # Plot validation results
    if len(metrics.validation_scores) > 0:
        val_path = os.path.join(args.output_dir, 'validation_results.png')
        plot_validation_results(metrics, save_path=val_path, show=True)

    # Print statistics
    print_statistics(metrics)

    print(f"\n✓ Plots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
