"""
Training script for the pattern-based Chess AI.

Uses the four pillars of learning (Feedback, Approximability, Composability,
Exploration) to learn chess strategies through game experience.

Usage:
    python -m chess_ai.train_chess                              # 500 episodes vs random
    python -m chess_ai.train_chess --episodes 1000              # More training
    python -m chess_ai.train_chess --opponent greedy             # Train vs greedy
    python -m chess_ai.train_chess --opponent minimax1           # Train vs minimax depth 1
    python -m chess_ai.train_chess --resume checkpoints_chess/   # Resume training
    python -m chess_ai.train_chess --color black                 # Play as black
"""

import argparse
import chess

from chess_ai.chess_agent import ChessPatternAgent
from chess_ai.opponents import CHESS_OPPONENTS


def main():
    parser = argparse.ArgumentParser(
        description='Train the pattern-based Chess AI agent'
    )
    parser.add_argument(
        '--episodes', type=int, default=500,
        help='Number of training games (default: 500)')
    parser.add_argument(
        '--opponent', type=str, default='random',
        choices=list(CHESS_OPPONENTS.keys()),
        help='Opponent to train against (default: random)')
    parser.add_argument(
        '--color', type=str, default='white',
        choices=['white', 'black'],
        help='Color to play as (default: white)')
    parser.add_argument(
        '--max-moves', type=int, default=200,
        help='Max half-moves per game (default: 200)')
    parser.add_argument(
        '--log-interval', type=int, default=10,
        help='Print stats every N episodes (default: 10)')
    parser.add_argument(
        '--save-path', type=str, default='checkpoints_chess',
        help='Where to save checkpoints (default: checkpoints_chess)')
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Resume from checkpoint path')
    args = parser.parse_args()

    print("=" * 70)
    print("RECURSIVE LEARNING AI - Chess Pattern Training")
    print("Four Pillars: Feedback | Approximability | Composability "
          "| Exploration")
    print("=" * 70)
    print()

    color = chess.WHITE if args.color == 'white' else chess.BLACK

    agent = ChessPatternAgent()

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        agent.load(args.resume)
        print(f"  Episodes so far: {agent.episodes_completed}")
        print(f"  Patterns loaded: "
              f"{agent.policy.get_stats()['patterns_discovered']}")
        print()

    agent.train(
        total_episodes=args.episodes,
        opponent_name=args.opponent,
        color=color,
        log_interval=args.log_interval,
        save_path=args.save_path,
        max_moves=args.max_moves,
    )


if __name__ == '__main__':
    main()
