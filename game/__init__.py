"""
MicroRTS-style Game Environment for Recursive Learning AI

A pure-Python real-time strategy game environment modeled after MicroRTS,
designed for training reinforcement learning agents. Features:

- Grid-based maps with resources, bases, barracks, and units
- Multiple unit types: Worker, Light, Heavy, Ranged
- Resource harvesting, building construction, combat
- Vectorized observation space with feature planes
- Multi-discrete action space with invalid action masking
- Deterministic game mechanics
"""

from game.units import UnitType, Unit, UNIT_STATS
from game.game_map import GameMap
from game.game_state import GameState
from game.actions import ActionType, Action
from game.engine import GameEngine
from game.ai_opponents import RandomAI, RushAI, EconomyAI, DefensiveAI
from game.renderer import GameRenderer

__all__ = [
    "UnitType", "Unit", "UNIT_STATS",
    "GameMap", "GameState",
    "ActionType", "Action",
    "GameEngine",
    "RandomAI", "RushAI", "EconomyAI", "DefensiveAI",
    "GameRenderer",
]
