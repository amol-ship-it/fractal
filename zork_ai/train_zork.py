"""
Training script for the pattern-based Zork AI agent.

Uses the four pillars of learning (feedback, approximability, composability,
exploration) to learn Zork I strategies through pattern matching — no neural
network involved.

Usage:
    python -m zork_ai.train_zork --game-file zork1.z5
    python -m zork_ai.train_zork --game-file zork1.z5 --episodes 1000
    python -m zork_ai.train_zork --game-file zork1.z5 --parallel --workers 4
    python -m zork_ai.train_zork --game-file zork1.z5 --resume checkpoints_zork/
    python -m zork_ai.train_zork --game-file zork1.z5 --play
    python -m zork_ai.train_zork --game-file zork1.z5 --play --resume checkpoints_zork/
"""

import argparse
import os
import sys

from zork_ai.zork_agent import ZorkPatternAgent


def main():
    parser = argparse.ArgumentParser(
        description='Train the pattern-based Zork AI agent'
    )
    parser.add_argument('--game-file', type=str, required=True,
                        help='Path to Zork I .z5 game file')
    parser.add_argument('--frotz-path', type=str, default='dfrotz',
                        help='Path to dfrotz binary (default: dfrotz)')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of training episodes (default: 500)')
    parser.add_argument('--max-moves', type=int, default=400,
                        help='Max moves per episode (default: 400)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log every N episodes (default: 10)')
    parser.add_argument('--save-path', type=str, default='checkpoints_zork',
                        help='Save path (default: checkpoints_zork)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed game output for first few episodes')
    parser.add_argument('--play', action='store_true',
                        help='Watch the AI play one game with visual output '
                             '(no training, just observation)')

    # Parallel training options
    parser.add_argument('--parallel', '-p', action='store_true',
                        help='Enable parallel training across CPU cores')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='Number of worker processes (default: cpu_count)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Episodes per parallel batch (default: workers)')
    parser.add_argument('--warmup', type=int, default=None,
                        help='Sequential warmup episodes before parallel '
                             '(default: 0)')

    args = parser.parse_args()

    # Validate game file
    if not os.path.exists(args.game_file):
        print(f"ERROR: Game file not found: {args.game_file}")
        print(f"  Download Zork I: curl -L -o zork1.z5 "
              f"https://github.com/danielricks/textplayer/raw/master/"
              f"games/zork1.z5")
        sys.exit(1)

    agent = ZorkPatternAgent(
        game_file=args.game_file,
        frotz_path=args.frotz_path,
    )

    # Resume from checkpoint
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            agent.load(args.resume)
            print(f"  Episodes trained: {agent.episodes_completed}")
            print(f"  Best score: {agent.best_score}")
            print(f"  Patterns loaded: "
                  f"{agent.policy.get_stats()['patterns_discovered']}")
            print(f"  Exploration rate: "
                  f"{agent.policy.explorer.exploration_rate:.1%}")
            print()
        else:
            print(f"WARNING: Checkpoint {args.resume} not found, "
                  f"starting fresh.")
            print()

    # ── Play mode: watch the AI play one game ──
    if args.play:
        agent.play_visual(max_moves=args.max_moves)
        return

    # ── Training mode ──
    print("=" * 70)
    print("RECURSIVE LEARNING AI - Zork I Pattern-Based Training")
    print("Four Pillars: Feedback | Approximability | "
          "Composability | Exploration")
    print("=" * 70)
    print()

    if args.parallel:
        result = agent.train_parallel(
            total_episodes=args.episodes,
            workers=args.workers,
            batch_size=args.batch_size,
            log_interval=args.log_interval,
            save_path=args.save_path,
            max_moves=args.max_moves,
            warmup=args.warmup,
        )
    else:
        result = agent.train(
            total_episodes=args.episodes,
            log_interval=args.log_interval,
            save_path=args.save_path,
            max_moves=args.max_moves,
            verbose=args.verbose,
        )

    print(f"\n  Visualize learned patterns (opens browser dashboard):")
    print(f"    python -m zork_ai.visualize_zork "
          f"--checkpoint {args.save_path}")
    print(f"\n  Watch the trained AI play:")
    print(f"    python -m zork_ai.train_zork --game-file {args.game_file} "
          f"--play --resume {args.save_path}")


if __name__ == '__main__':
    main()
