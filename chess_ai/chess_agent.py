"""
Chess Pattern Agent — Training loop for the pattern-based chess AI.

Runs games of chess, using the ChessPatternPolicy to make decisions
each turn, then feeds win/loss/draw outcomes back to refine patterns.

No neural network. No gradient descent. Learning happens through the
four pillars: feedback, approximability, composability, exploration.

Usage:
    python -m chess_ai.train_chess --episodes 500
"""

import json
import os
import time
import chess
import numpy as np
from typing import Dict, Optional, List

from chess_ai.chess_policy import ChessPatternPolicy, ChessStrategy
from chess_ai.opponents import CHESS_OPPONENTS


class ChessPatternAgent:
    """
    Training coordinator for the pattern-based chess AI.

    Runs full chess games, feeds outcomes back to the ChessPatternPolicy,
    and tracks learning progress.
    """

    def __init__(self):
        self.policy = ChessPatternPolicy()

        # Training stats
        self.episodes_completed = 0
        self.win_history: List[float] = []  # 1.0=win, 0.5=draw, 0.0=loss
        self.game_lengths: List[int] = []

    def play_episode(self, opponent, color: chess.Color = chess.WHITE,
                     max_moves: int = 200) -> Dict:
        """
        Play one full chess game.

        Args:
            opponent: object with get_move(board, color) -> chess.Move
            color: which color the pattern agent plays
            max_moves: max half-moves before declaring draw

        Returns:
            Dict with game outcome info.
        """
        board = chess.Board()
        self.policy.begin_episode()
        move_count = 0

        while not board.is_game_over() and move_count < max_moves:
            if board.turn == color:
                move = self.policy.get_move(board, color)
            else:
                move = opponent.get_move(board, not color)

            if move is None:
                break
            board.push(move)
            move_count += 1

        # Determine outcome
        result = board.result()
        if result == "*":
            # Max moves reached or no legal moves → draw
            draw = True
            won = False
        else:
            draw = result == "1/2-1/2"
            if color == chess.WHITE:
                won = result == "1-0"
            else:
                won = result == "0-1"

        self.episodes_completed += 1
        self.win_history.append(1.0 if won else (0.5 if draw else 0.0))
        self.game_lengths.append(move_count)

        # Compute final material balance
        features = self.policy.encoder.encode_board(board)
        material_balance = features[37]  # index 37 = material balance

        game_info = {
            'num_moves': move_count,
            'result': result,
            'final_material_balance': material_balance,
        }

        # Feed outcome back to policy (this is where learning happens)
        self.policy.record_outcome(won, draw, game_info)

        return {
            'won': won,
            'draw': draw,
            'num_moves': move_count,
            'result': result,
            'final_material_balance': material_balance,
        }

    def train(self, total_episodes: int = 500,
              opponent_name: str = 'random',
              color: chess.Color = chess.WHITE,
              log_interval: int = 10,
              save_path: str = None,
              max_moves: int = 200) -> Dict:
        """
        Main training loop. Plays chess games and learns from outcomes.
        """
        start_time = time.time()

        # Create opponent
        opp_factory = CHESS_OPPONENTS.get(opponent_name)
        if opp_factory is None:
            raise ValueError(
                f"Unknown opponent: {opponent_name}. "
                f"Options: {list(CHESS_OPPONENTS.keys())}")
        opponent = opp_factory() if callable(opp_factory) else opp_factory

        color_name = "White" if color == chess.WHITE else "Black"
        print(f"Starting chess pattern training: {total_episodes} episodes")
        print(f"  Playing as: {color_name}")
        print(f"  Opponent: {opponent_name}")
        print(f"  Initial exploration rate: "
              f"{self.policy.explorer.exploration_rate:.1%}")
        print()

        for ep in range(1, total_episodes + 1):
            result = self.play_episode(opponent, color, max_moves)

            if ep % log_interval == 0:
                recent = self.win_history[-100:]
                win_rate = np.mean([1.0 if x == 1.0 else 0.0
                                    for x in recent])
                draw_rate = np.mean([1.0 if x == 0.5 else 0.0
                                     for x in recent])
                avg_moves = np.mean(self.game_lengths[-100:])
                stats = self.policy.get_stats()

                print(
                    f"Episode {ep}/{total_episodes} | "
                    f"Win: {win_rate:.1%} | "
                    f"Draw: {draw_rate:.1%} | "
                    f"Avg Moves: {avg_moves:.0f} | "
                    f"Patterns: {stats['patterns_discovered']} | "
                    f"Bindings: {stats['strategy_bindings']} | "
                    f"Explore: {stats['exploration_rate']:.1%}")

            # Save periodically
            if save_path and ep % (log_interval * 10) == 0:
                self.save(save_path)

        elapsed = time.time() - start_time

        # Final save
        if save_path:
            self.save(save_path)

        final_stats = self.policy.get_stats()
        recent = self.win_history[-100:]
        final_win_rate = np.mean([1.0 if x == 1.0 else 0.0
                                  for x in recent]) if recent else 0
        final_draw_rate = np.mean([1.0 if x == 0.5 else 0.0
                                   for x in recent]) if recent else 0

        print(f"\n{'=' * 60}")
        print(f"TRAINING COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Episodes:         {self.episodes_completed}")
        print(f"  Final win rate:   {final_win_rate:.1%}")
        print(f"  Final draw rate:  {final_draw_rate:.1%}")
        print(f"  Patterns learned: {final_stats['patterns_discovered']}")
        print(f"  Strategy bindings:{final_stats['strategy_bindings']}")
        print(f"  Exploration rate: {final_stats['exploration_rate']:.1%}")
        print(f"  Elapsed time:     {elapsed:.1f}s")
        print(f"\n  Strategy usage:")
        for name, count in final_stats['strategy_usage'].items():
            print(f"    {name}: {count}")

        if save_path:
            print(f"\n  Checkpoint saved to: {save_path}")

        return {
            'episodes': self.episodes_completed,
            'final_win_rate': final_win_rate,
            'final_draw_rate': final_draw_rate,
            'patterns_learned': final_stats['patterns_discovered'],
            'elapsed_time': elapsed,
        }

    def save(self, path: str):
        """Save agent state."""
        os.makedirs(path, exist_ok=True)
        self.policy.save(path)

        def to_native(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: to_native(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [to_native(v) for v in obj]
            return obj

        stats = to_native({
            'episodes_completed': self.episodes_completed,
            'win_history': self.win_history[-1000:],
            'game_lengths': self.game_lengths[-1000:],
        })
        with open(os.path.join(path, 'training_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)

    def load(self, path: str):
        """Load agent state."""
        self.policy.load(path)

        stats_path = os.path.join(path, 'training_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            self.episodes_completed = stats.get('episodes_completed', 0)
            self.win_history = stats.get('win_history', [])
            self.game_lengths = stats.get('game_lengths', [])
