"""
Game State - Complete game state with observation encoding.

Provides the bridge between the game engine and RL agents by encoding
the game state as a tensor of feature planes (matching MicroRTS format).

Observation space: (height, width, num_feature_planes)
Feature planes encode HP, resources carried, ownership, unit type,
current action, and terrain for each cell.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from game.game_map import GameMap, Terrain
from game.units import Unit, UnitType, ActionState, UNIT_STATS
from game.actions import (
    get_action_space_dims, get_valid_actions_mask, DIMS_PER_CELL
)


# Feature plane definitions (27 planes total)
# HP: 0, 1, 2, 3, >=4  (5 planes)
# Resources carried: 0, 1, 2, 3, >=4  (5 planes)
# Owner: none, player 0, player 1  (3 planes)
# Unit type: none, resource, base, barracks, worker, light, heavy, ranged (8 planes)
# Current action: idle, move, harvest, return, produce, attack (6 planes)

NUM_HP_PLANES = 5
NUM_RESOURCE_PLANES = 5
NUM_OWNER_PLANES = 3
NUM_TYPE_PLANES = 8
NUM_ACTION_PLANES = 6
NUM_FEATURE_PLANES = (NUM_HP_PLANES + NUM_RESOURCE_PLANES + NUM_OWNER_PLANES
                      + NUM_TYPE_PLANES + NUM_ACTION_PLANES)
# = 5 + 5 + 3 + 8 + 6 = 27


class GameState:
    """
    Complete game state wrapper providing observation encoding and action masking.
    """

    def __init__(self, game_map: GameMap):
        self.game_map = game_map
        self.player_resources = {0: 0, 1: 0}  # Starting resources
        self.tick = 0
        self.max_ticks = 2000
        self.done = False
        self.winner = -1  # -1 = ongoing, 0 = player 0, 1 = player 1, 2 = draw

    def get_observation(self, player: int) -> np.ndarray:
        """
        Encode game state as a (H, W, NUM_FEATURE_PLANES) tensor.
        Observation is from the perspective of the given player.
        """
        h, w = self.game_map.height, self.game_map.width
        obs = np.zeros((h, w, NUM_FEATURE_PLANES), dtype=np.float32)

        for unit in self.game_map.units.values():
            if not unit.is_alive:
                continue

            x, y = unit.x, unit.y
            offset = 0

            # HP planes (one-hot for 0,1,2,3,>=4)
            hp_idx = min(unit.hp, 4)
            obs[y, x, offset + hp_idx] = 1.0
            offset += NUM_HP_PLANES

            # Resources carried planes
            rc_idx = min(unit.resources_carried, 4)
            obs[y, x, offset + rc_idx] = 1.0
            offset += NUM_RESOURCE_PLANES

            # Owner planes (0=none/resource, 1=player 0, 2=player 1)
            if unit.player == -1:
                obs[y, x, offset + 0] = 1.0
            elif unit.player == player:
                obs[y, x, offset + 1] = 1.0  # "my" units
            else:
                obs[y, x, offset + 2] = 1.0  # enemy units
            offset += NUM_OWNER_PLANES

            # Unit type planes (one-hot)
            # index 0 = none (empty cell), 1-7 = unit types
            type_idx = int(unit.unit_type) + 1  # +1 because 0 is "none"
            if type_idx < NUM_TYPE_PLANES:
                obs[y, x, offset + type_idx] = 1.0
            offset += NUM_TYPE_PLANES

            # Current action planes
            action_idx = int(unit.action_state)
            if action_idx < NUM_ACTION_PLANES:
                obs[y, x, offset + action_idx] = 1.0

        # Mark empty cells with "none" ownership and "none" unit type
        for y in range(h):
            for x in range(w):
                unit = self.game_map.get_unit_at(x, y)
                if unit is None:
                    offset = NUM_HP_PLANES + NUM_RESOURCE_PLANES
                    obs[y, x, offset + 0] = 1.0  # owner = none
                    offset += NUM_OWNER_PLANES
                    obs[y, x, offset + 0] = 1.0  # type = none

        return obs

    def get_action_mask(self, player: int) -> np.ndarray:
        """
        Get invalid action mask for all cells.

        Returns: (H * W, sum(action_dims)) binary mask where 1 = valid.
        """
        h, w = self.game_map.height, self.game_map.width
        dims = get_action_space_dims()
        total_dims = sum(dims)

        mask = np.zeros((h * w, total_dims), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                cell_idx = y * w + x
                unit = self.game_map.get_unit_at(x, y)

                if unit is None or unit.player != player:
                    # No unit or enemy unit: only NOOP is valid
                    mask[cell_idx, 0] = 1.0  # NOOP
                    continue

                unit_mask = get_valid_actions_mask(
                    unit, self.game_map, self.player_resources[player]
                )
                offset = 0
                for dim_idx, dim_mask in enumerate(unit_mask):
                    for j, val in enumerate(dim_mask):
                        mask[cell_idx, offset + j] = float(val)
                    offset += dims[dim_idx]

        return mask

    def get_flat_action_mask(self, player: int) -> np.ndarray:
        """Get a fully flattened action mask: (H * W * sum(action_dims),)."""
        return self.get_action_mask(player).flatten()

    def check_game_over(self) -> Tuple[bool, int]:
        """
        Check if game is over.

        Game ends when:
        - A player has no units left
        - Max ticks reached (draw or score-based)
        """
        p0_units = self.game_map.get_player_units(0)
        p1_units = self.game_map.get_player_units(1)

        if not p0_units and not p1_units:
            return True, 2  # Draw
        if not p0_units:
            return True, 1  # Player 1 wins
        if not p1_units:
            return True, 0  # Player 0 wins
        if self.tick >= self.max_ticks:
            # Score-based: count total unit HP + resources
            s0 = sum(u.hp for u in p0_units) + self.player_resources[0]
            s1 = sum(u.hp for u in p1_units) + self.player_resources[1]
            if s0 > s1:
                return True, 0
            elif s1 > s0:
                return True, 1
            return True, 2

        return False, -1

    def compute_reward(self, player: int, prev_state_info: Dict) -> float:
        """
        Compute shaped reward for a player. Components:
        - Win/loss: +10/-10
        - Resource delta: +1 per resource gained
        - Unit production: +1 per unit produced
        - Damage dealt: +0.2 per HP of damage
        - Combat unit production: +4 per combat unit
        """
        reward = 0.0

        # Win/loss
        done, winner = self.check_game_over()
        if done:
            if winner == player:
                reward += 10.0
            elif winner == 1 - player:
                reward -= 10.0
            # Draw = 0

        # Resource delta
        curr_resources = self.player_resources[player]
        prev_resources = prev_state_info.get('resources', 0)
        reward += max(0, curr_resources - prev_resources) * 1.0

        # Unit count delta (weighted by type)
        curr_units = self.game_map.get_player_units(player)
        prev_unit_count = prev_state_info.get('unit_count', 0)
        curr_combat = sum(1 for u in curr_units
                          if u.unit_type in (UnitType.LIGHT, UnitType.HEAVY, UnitType.RANGED))
        prev_combat = prev_state_info.get('combat_count', 0)
        reward += max(0, curr_combat - prev_combat) * 4.0
        reward += max(0, len(curr_units) - prev_unit_count) * 1.0

        # Damage dealt
        enemy_units = self.game_map.get_player_units(1 - player)
        curr_enemy_hp = sum(u.hp for u in enemy_units)
        prev_enemy_hp = prev_state_info.get('enemy_hp', 0)
        damage_dealt = max(0, prev_enemy_hp - curr_enemy_hp)
        reward += damage_dealt * 0.2

        return reward

    def get_state_info(self, player: int) -> Dict:
        """Capture current state info for reward computation."""
        units = self.game_map.get_player_units(player)
        enemy_units = self.game_map.get_player_units(1 - player)
        return {
            'resources': self.player_resources[player],
            'unit_count': len(units),
            'combat_count': sum(1 for u in units
                                if u.unit_type in (UnitType.LIGHT, UnitType.HEAVY, UnitType.RANGED)),
            'enemy_hp': sum(u.hp for u in enemy_units),
        }
