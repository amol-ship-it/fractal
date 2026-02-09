"""
Chess Opponents — Built-in players for the pattern AI to train against.

All implement: get_move(board, color) → chess.Move

- RandomPlayer: uniformly random legal move
- GreedyPlayer: captures highest-value piece if possible, else random
- MinimaxPlayer: negamax with alpha-beta pruning, material evaluation
"""

import random
import chess
from typing import Optional


PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


class RandomPlayer:
    """Picks a uniformly random legal move."""

    def get_move(self, board: chess.Board, color: chess.Color) -> chess.Move:
        moves = list(board.legal_moves)
        return random.choice(moves)


class GreedyPlayer:
    """Captures highest-value piece if possible, else random."""

    def get_move(self, board: chess.Board, color: chess.Color) -> chess.Move:
        best_move = None
        best_value = -1

        for move in board.legal_moves:
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                if captured is None:
                    # En passant
                    value = PIECE_VALUES[chess.PAWN]
                else:
                    value = PIECE_VALUES.get(captured.piece_type, 0)
                if value > best_value:
                    best_value = value
                    best_move = move

        if best_move is not None:
            return best_move
        return random.choice(list(board.legal_moves))


class MinimaxPlayer:
    """
    Minimax player with alpha-beta pruning.

    Uses simple material evaluation. Configurable search depth.
    """

    def __init__(self, depth: int = 2):
        self.depth = depth

    def get_move(self, board: chess.Board, color: chess.Color) -> chess.Move:
        moves = list(board.legal_moves)
        if not moves:
            return None

        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for move in moves:
            board.push(move)
            score = -self._negamax(board, self.depth - 1, -beta, -alpha)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)

        return best_move if best_move else random.choice(moves)

    def _negamax(self, board: chess.Board, depth: int,
                 alpha: float, beta: float) -> float:
        """Negamax with alpha-beta pruning."""
        if depth == 0 or board.is_game_over():
            return self._evaluate(board)

        best = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best

    def _evaluate(self, board: chess.Board) -> float:
        """Material evaluation from side-to-move perspective."""
        if board.is_checkmate():
            return -99999
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        score = 0
        for pt, val in PIECE_VALUES.items():
            score += len(board.pieces(pt, board.turn)) * val
            score -= len(board.pieces(pt, not board.turn)) * val
        return score


# Lookup for training script
CHESS_OPPONENTS = {
    'random': RandomPlayer,
    'greedy': GreedyPlayer,
    'minimax1': lambda: MinimaxPlayer(depth=1),
    'minimax2': lambda: MinimaxPlayer(depth=2),
}
