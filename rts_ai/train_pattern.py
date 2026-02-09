"""
Training script for the pattern-based AI agent.

Uses the four pillars of learning (feedback, approximability, composability,
exploration) to learn RTS strategies through pattern matching — no neural
network involved.

Usage:
    python -m rts_ai.train_pattern                          # Default: 500 episodes vs Rush
    python -m rts_ai.train_pattern --episodes 1000          # More training
    python -m rts_ai.train_pattern --opponent economy       # Train vs Economy AI
    python -m rts_ai.train_pattern --resume checkpoints_pattern/  # Resume training
    python -m rts_ai.train_pattern --save-path my_agent/    # Custom save path

    # Parallel training (uses all CPU cores):
    python -m rts_ai.train_pattern --episodes 50000 --parallel
    python -m rts_ai.train_pattern --episodes 50000 --parallel --workers 8
    python -m rts_ai.train_pattern --episodes 50000 --parallel --batch-size 64
"""

import argparse
import os
import sys

from rts_ai.pattern_agent import PatternAgent


def main():
    parser = argparse.ArgumentParser(
        description='Train the pattern-based RTS AI agent'
    )
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of training episodes (default: 500)')
    parser.add_argument('--opponent', type=str, default='rush',
                        choices=['rush', 'economy', 'defensive', 'random'],
                        help='Opponent AI (default: rush)')
    parser.add_argument('--map-size', type=int, default=8,
                        help='Map size (default: 8)')
    parser.add_argument('--max-ticks', type=int, default=2000,
                        help='Max ticks per game (default: 2000)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log every N episodes (default: 10)')
    parser.add_argument('--save-path', type=str, default='checkpoints_pattern',
                        help='Save path (default: checkpoints_pattern)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')

    # Parallel training options
    parser.add_argument('--parallel', '-p', action='store_true',
                        help='Enable parallel training across CPU cores')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='Number of worker processes (default: cpu_count)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Episodes per parallel batch (default: workers*2)')
    parser.add_argument('--warmup', type=int, default=None,
                        help='Sequential warmup episodes before parallel '
                             '(default: 10%% of total, capped at 200)')
    args = parser.parse_args()

    print("=" * 70)
    print("RECURSIVE LEARNING AI - Pattern-Based Training")
    print("Four Pillars: Feedback | Approximability | Composability | Exploration")
    print("=" * 70)
    print()

    agent = PatternAgent(map_height=args.map_size, map_width=args.map_size)

    # Resume from checkpoint
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            agent.load(args.resume)
            print(f"  Episodes so far: {agent.episodes_completed}")
            print(f"  Patterns loaded: "
                  f"{agent.policy.get_stats()['patterns_discovered']}")
            print()
        else:
            print(f"WARNING: Checkpoint {args.resume} not found, "
                  f"starting fresh.")
            print()

    if args.parallel:
        result = agent.train_parallel(
            total_episodes=args.episodes,
            opponent=args.opponent,
            workers=args.workers,
            batch_size=args.batch_size,
            log_interval=args.log_interval,
            save_path=args.save_path,
            max_ticks=args.max_ticks,
            warmup=args.warmup,
        )
    else:
        result = agent.train(
            total_episodes=args.episodes,
            opponent=args.opponent,
            log_interval=args.log_interval,
            save_path=args.save_path,
            max_ticks=args.max_ticks,
        )

    # Show how to visualize
    print(f"\n  Visualize learned patterns:")
    print(f"    python -m chess_ai.visualize_chess "
          f"--checkpoint {args.save_path}")


if __name__ == '__main__':
    main()
