"""
Zork AI — Pattern-based Zork I agent using the four pillars of learning.

No neural network. Learning happens through pattern recognition, comparison,
composition, and feedback — the same architecture as the RTS and Chess agents.

Usage:
    python -m zork_ai.train_zork --game-file zork1.z5 --episodes 500
    python -m zork_ai.train_zork --game-file zork1.z5 --parallel --workers 4
"""

from zork_ai.zork_policy import ZorkStrategy, ZorkPatternPolicy
from zork_ai.zork_agent import ZorkPatternAgent

__all__ = [
    'ZorkStrategy',
    'ZorkPatternPolicy',
    'ZorkPatternAgent',
]
