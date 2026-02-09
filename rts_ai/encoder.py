"""
Game State Encoder - Bridges game observations to the Recursive Learning AI.

Converts game state observations (spatial feature planes) into:
1. Tensor observations for the neural network policy
2. Flattened feature vectors for the pattern recognition engine
3. Strategic feature summaries for knowledge extraction
"""

import numpy as np
from typing import Dict, List, Tuple

from game.game_state import GameState, NUM_FEATURE_PLANES
from game.units import UnitType


class GameStateEncoder:
    """
    Encodes game state into multiple representations for different subsystems.
    """

    def __init__(self, map_height: int, map_width: int):
        self.map_height = map_height
        self.map_width = map_width
        self.num_feature_planes = NUM_FEATURE_PLANES

    def encode_observation(self, state: GameState, player: int) -> np.ndarray:
        """
        Get the raw spatial observation tensor.
        Shape: (H, W, NUM_FEATURE_PLANES)
        """
        return state.get_observation(player)

    def encode_for_pattern_engine(self, state: GameState,
                                  player: int) -> List[float]:
        """
        Encode game state as a flat feature vector for the pattern recognition engine.

        Extracts high-level strategic features:
        - Resource counts
        - Unit composition
        - Territory control
        - Economic indicators
        - Military strength ratios
        """
        features = []

        my_units = state.game_map.get_player_units(player)
        enemy_units = state.game_map.get_player_units(1 - player)

        # Resource features
        my_resources = state.player_resources[player]
        enemy_resources = state.player_resources[1 - player]
        features.append(min(my_resources / 20.0, 1.0))
        features.append(min(enemy_resources / 20.0, 1.0))

        # Unit count features (normalized)
        unit_types = [UnitType.WORKER, UnitType.LIGHT, UnitType.HEAVY,
                      UnitType.RANGED, UnitType.BASE, UnitType.BARRACKS]
        for ut in unit_types:
            my_count = sum(1 for u in my_units if u.unit_type == ut)
            enemy_count = sum(1 for u in enemy_units if u.unit_type == ut)
            features.append(min(my_count / 10.0, 1.0))
            features.append(min(enemy_count / 10.0, 1.0))

        # Total military power (HP * damage)
        my_power = sum(u.hp * u.stats.damage for u in my_units
                       if u.stats.damage > 0)
        enemy_power = sum(u.hp * u.stats.damage for u in enemy_units
                          if u.stats.damage > 0)
        total_power = max(my_power + enemy_power, 1)
        features.append(my_power / total_power)
        features.append(enemy_power / total_power)

        # Territory control (average position of units)
        h, w = self.map_height, self.map_width
        if my_units:
            my_cx = sum(u.x for u in my_units) / len(my_units) / w
            my_cy = sum(u.y for u in my_units) / len(my_units) / h
        else:
            my_cx, my_cy = 0.0, 0.0
        if enemy_units:
            en_cx = sum(u.x for u in enemy_units) / len(enemy_units) / w
            en_cy = sum(u.y for u in enemy_units) / len(enemy_units) / h
        else:
            en_cx, en_cy = 1.0, 1.0
        features.extend([my_cx, my_cy, en_cx, en_cy])

        # Game phase (early/mid/late based on tick)
        phase = min(state.tick / state.max_ticks, 1.0)
        features.append(phase)

        # Economic rate (resources per worker)
        num_workers = sum(1 for u in my_units if u.unit_type == UnitType.WORKER)
        features.append(min(num_workers / 4.0, 1.0))

        # Available resources on map
        resource_count = len(state.game_map.get_resources())
        features.append(min(resource_count / 20.0, 1.0))

        return features

    def extract_strategic_summary(self, state: GameState,
                                  player: int) -> Dict[str, float]:
        """
        Extract a human-readable strategic summary for knowledge storage.
        These become the transferable concepts for future games.
        """
        my_units = state.game_map.get_player_units(player)
        enemy_units = state.game_map.get_player_units(1 - player)

        my_workers = sum(1 for u in my_units if u.unit_type == UnitType.WORKER)
        my_combat = sum(1 for u in my_units
                        if u.unit_type in (UnitType.LIGHT, UnitType.HEAVY, UnitType.RANGED))
        my_buildings = sum(1 for u in my_units if u.is_structure)
        enemy_combat = sum(1 for u in enemy_units
                           if u.unit_type in (UnitType.LIGHT, UnitType.HEAVY, UnitType.RANGED))

        total_units = len(my_units) + len(enemy_units)

        return {
            # Economy metrics
            'economy_ratio': my_workers / max(len(my_units), 1),
            'resource_advantage': (state.player_resources[player]
                                   - state.player_resources[1 - player]) / 20.0,
            # Military metrics
            'military_ratio': my_combat / max(my_combat + enemy_combat, 1),
            'army_size': my_combat / 10.0,
            # Infrastructure
            'infrastructure': my_buildings / max(total_units, 1),
            # Map control
            'unit_spread': self._compute_unit_spread(my_units),
            # Aggression
            'aggression': self._compute_aggression(my_units, enemy_units),
            # Game phase
            'game_phase': min(state.tick / state.max_ticks, 1.0),
        }

    def _compute_unit_spread(self, units: List) -> float:
        """Compute how spread out units are (0=clustered, 1=spread)."""
        if len(units) <= 1:
            return 0.0
        cx = sum(u.x for u in units) / len(units)
        cy = sum(u.y for u in units) / len(units)
        avg_dist = sum(abs(u.x - cx) + abs(u.y - cy) for u in units) / len(units)
        max_dist = (self.map_width + self.map_height) / 2
        return min(avg_dist / max_dist, 1.0)

    def _compute_aggression(self, my_units: List, enemy_units: List) -> float:
        """
        Compute aggression level (how close our units are to enemy).
        0 = defensive, 1 = very aggressive
        """
        if not my_units or not enemy_units:
            return 0.0

        # Average distance from my combat units to nearest enemy
        combat_units = [u for u in my_units
                        if u.unit_type in (UnitType.LIGHT, UnitType.HEAVY, UnitType.RANGED)]
        if not combat_units:
            return 0.0

        total_dist = 0
        for cu in combat_units:
            min_dist = min(cu.distance_to(eu.x, eu.y) for eu in enemy_units)
            total_dist += min_dist
        avg_dist = total_dist / len(combat_units)
        max_dist = self.map_width + self.map_height
        return 1.0 - min(avg_dist / max_dist, 1.0)
