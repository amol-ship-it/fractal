"""
Chess AI — Pattern-based chess agent using the four pillars of learning.

Uses the same core AI components (PatternEngine, DualMemory, FeedbackLoop,
ExplorationStrategy) that power the RTS AI, applied to the game of chess.

No neural network. No gradient descent. Learning happens through:
- Feedback Loops: Win/loss outcomes refine which situation→strategy pairs work
- Approximability: Pattern signatures improve with each game played
- Composability: Simple patterns compose into higher-level strategic concepts
- Exploration: Agent occasionally tries novel strategies to discover better ones
"""

from chess_ai.board_encoder import ChessBoardEncoder
from chess_ai.chess_policy import ChessStrategy, StrategyBinding, ChessPatternPolicy
from chess_ai.chess_agent import ChessPatternAgent
from chess_ai.opponents import RandomPlayer, GreedyPlayer, MinimaxPlayer

__all__ = [
    "ChessBoardEncoder",
    "ChessStrategy",
    "StrategyBinding",
    "ChessPatternPolicy",
    "ChessPatternAgent",
    "RandomPlayer",
    "GreedyPlayer",
    "MinimaxPlayer",
]
