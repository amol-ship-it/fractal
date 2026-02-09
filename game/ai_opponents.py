"""
Scripted AI Opponents - Bot opponents for training the RL agent.

Provides several difficulty levels:
- RandomAI: Random valid actions (weakest)
- RushAI: Aggressive rush strategy (produce combat units, attack)
- EconomyAI: Focus on economy, build up before attacking
- DefensiveAI: Balanced economy and defense
"""

import random
import numpy as np
from typing import List, Optional

from game.game_state import GameState
from game.game_map import GameMap
from game.units import Unit, UnitType, ActionState, UNIT_STATS, PRODUCTION_TABLE
from game.actions import (
    ActionType, Direction, DIR_OFFSETS, DIMS_PER_CELL,
    get_action_space_dims, get_valid_actions_mask, MAX_ATTACK_RANGE,
)


class BaseAI:
    """Base class for scripted AI opponents."""

    def get_action(self, state: GameState, player: int) -> np.ndarray:
        """Return action array for all cells: (H*W*DIMS_PER_CELL,)."""
        h, w = state.game_map.height, state.game_map.width
        actions = np.zeros(h * w * DIMS_PER_CELL, dtype=np.int32)
        return actions

    def _encode_action(self, action_type: int, move_dir: int = 0,
                       harvest_dir: int = 0, return_dir: int = 0,
                       produce_dir: int = 0, produce_type: int = 0,
                       attack_pos: int = 0) -> List[int]:
        return [action_type, move_dir, harvest_dir, return_dir,
                produce_dir, produce_type, attack_pos]

    def _set_cell_action(self, actions: np.ndarray, x: int, y: int,
                         width: int, cell_action: List[int]):
        idx = (y * width + x) * DIMS_PER_CELL
        for i, v in enumerate(cell_action):
            actions[idx + i] = v

    def _find_nearest(self, unit: Unit, targets: List[Unit]) -> Optional[Unit]:
        if not targets:
            return None
        return min(targets, key=lambda t: unit.distance_to(t.x, t.y))

    def _direction_toward(self, ux: int, uy: int, tx: int, ty: int) -> Direction:
        dx = tx - ux
        dy = ty - uy
        if abs(dx) >= abs(dy):
            return Direction.EAST if dx > 0 else Direction.WEST
        return Direction.SOUTH if dy > 0 else Direction.NORTH

    def _direction_to_adjacent(self, ux: int, uy: int,
                               tx: int, ty: int) -> Optional[Direction]:
        dx, dy = tx - ux, ty - uy
        if dx == 0 and dy == -1:
            return Direction.NORTH
        if dx == 1 and dy == 0:
            return Direction.EAST
        if dx == 0 and dy == 1:
            return Direction.SOUTH
        if dx == -1 and dy == 0:
            return Direction.WEST
        return None

    def _find_empty_adjacent_dir(self, unit: Unit,
                                 game_map: GameMap) -> Optional[Direction]:
        for d in Direction:
            dx, dy = DIR_OFFSETS[d]
            nx, ny = unit.x + dx, unit.y + dy
            if game_map.is_empty(nx, ny):
                return d
        return None

    def _move_toward(self, unit: Unit, tx: int, ty: int,
                     game_map: GameMap) -> Optional[List[int]]:
        """Try to move toward (tx,ty), trying alternate directions if blocked."""
        # Build list of directions sorted by preference (closest to target first)
        dx_total = tx - unit.x
        dy_total = ty - unit.y
        candidates = []

        # Primary direction (largest delta axis)
        if abs(dx_total) >= abs(dy_total):
            if dx_total > 0:
                candidates = [Direction.EAST, Direction.SOUTH, Direction.NORTH, Direction.WEST]
            elif dx_total < 0:
                candidates = [Direction.WEST, Direction.SOUTH, Direction.NORTH, Direction.EAST]
            else:
                candidates = list(Direction)
        else:
            if dy_total > 0:
                candidates = [Direction.SOUTH, Direction.EAST, Direction.WEST, Direction.NORTH]
            else:
                candidates = [Direction.NORTH, Direction.EAST, Direction.WEST, Direction.SOUTH]

        for d in candidates:
            ddx, ddy = DIR_OFFSETS[d]
            nx, ny = unit.x + ddx, unit.y + ddy
            if game_map.is_empty(nx, ny):
                return self._encode_action(ActionType.MOVE, move_dir=d)
        return None


class RandomAI(BaseAI):
    """Selects random valid actions for each unit."""

    def get_action(self, state: GameState, player: int) -> np.ndarray:
        h, w = state.game_map.height, state.game_map.width
        actions = np.zeros(h * w * DIMS_PER_CELL, dtype=np.int32)

        for unit in state.game_map.get_player_units(player):
            if not unit.is_idle:
                continue

            mask = get_valid_actions_mask(unit, state.game_map,
                                          state.player_resources[player])

            # Pick a random valid action type
            valid_types = [i for i, v in enumerate(mask[0]) if v == 1]
            if not valid_types or valid_types == [0]:
                continue

            # Prefer non-NOOP
            non_noop = [t for t in valid_types if t != 0]
            action_type = random.choice(non_noop) if non_noop else 0

            cell_action = [action_type, 0, 0, 0, 0, 0, 0]

            # Pick random valid sub-action
            if action_type == ActionType.MOVE:
                valid_dirs = [i for i, v in enumerate(mask[1]) if v == 1]
                cell_action[1] = random.choice(valid_dirs) if valid_dirs else 0
            elif action_type == ActionType.HARVEST:
                valid_dirs = [i for i, v in enumerate(mask[2]) if v == 1]
                cell_action[2] = random.choice(valid_dirs) if valid_dirs else 0
            elif action_type == ActionType.RETURN:
                valid_dirs = [i for i, v in enumerate(mask[3]) if v == 1]
                cell_action[3] = random.choice(valid_dirs) if valid_dirs else 0
            elif action_type == ActionType.PRODUCE:
                valid_dirs = [i for i, v in enumerate(mask[4]) if v == 1]
                valid_types_p = [i for i, v in enumerate(mask[5]) if v == 1]
                cell_action[4] = random.choice(valid_dirs) if valid_dirs else 0
                cell_action[5] = random.choice(valid_types_p) if valid_types_p else 0
            elif action_type == ActionType.ATTACK:
                valid_pos = [i for i, v in enumerate(mask[6]) if v == 1]
                cell_action[6] = random.choice(valid_pos) if valid_pos else 0

            self._set_cell_action(actions, unit.x, unit.y, w, cell_action)

        return actions


class RushAI(BaseAI):
    """
    Aggressive rush strategy:
    1. One worker harvests resources
    2. Base produces workers until 2, then barracks is built
    3. Barracks produces light units
    4. All combat units rush toward enemy base
    """

    def get_action(self, state: GameState, player: int) -> np.ndarray:
        h, w = state.game_map.height, state.game_map.width
        actions = np.zeros(h * w * DIMS_PER_CELL, dtype=np.int32)
        gm = state.game_map
        resources = state.player_resources[player]

        workers = gm.get_units_of_type(player, UnitType.WORKER)
        bases = gm.get_units_of_type(player, UnitType.BASE)
        barracks = gm.get_units_of_type(player, UnitType.BARRACKS)
        combat_units = (gm.get_units_of_type(player, UnitType.LIGHT)
                        + gm.get_units_of_type(player, UnitType.HEAVY)
                        + gm.get_units_of_type(player, UnitType.RANGED))

        enemy_units = gm.get_player_units(1 - player)
        resource_deposits = gm.get_resources()

        for unit in gm.get_player_units(player):
            if not unit.is_idle:
                continue

            cell_action = None

            if unit.unit_type == UnitType.BASE:
                # Produce workers if we need them
                if len(workers) < 2 and resources >= UNIT_STATS[UnitType.WORKER].cost:
                    d = self._find_empty_adjacent_dir(unit, gm)
                    if d is not None:
                        cell_action = self._encode_action(
                            ActionType.PRODUCE, produce_dir=d,
                            produce_type=UnitType.WORKER
                        )

            elif unit.unit_type == UnitType.BARRACKS:
                # Produce light units
                if resources >= UNIT_STATS[UnitType.LIGHT].cost:
                    d = self._find_empty_adjacent_dir(unit, gm)
                    if d is not None:
                        cell_action = self._encode_action(
                            ActionType.PRODUCE, produce_dir=d,
                            produce_type=UnitType.LIGHT
                        )

            elif unit.unit_type == UnitType.WORKER:
                if len(workers) >= 2 and not barracks and resources >= UNIT_STATS[UnitType.BARRACKS].cost:
                    # Build barracks
                    d = self._find_empty_adjacent_dir(unit, gm)
                    if d is not None:
                        cell_action = self._encode_action(
                            ActionType.PRODUCE, produce_dir=d,
                            produce_type=UnitType.BARRACKS
                        )
                elif unit.resources_carried > 0:
                    # Return to base
                    base = self._find_nearest(unit, bases)
                    if base and unit.distance_to(base.x, base.y) == 1:
                        d = self._direction_to_adjacent(unit.x, unit.y, base.x, base.y)
                        if d is not None:
                            cell_action = self._encode_action(
                                ActionType.RETURN, return_dir=d
                            )
                    elif base:
                        cell_action = self._move_toward(unit, base.x, base.y, gm)
                else:
                    # Harvest
                    res = self._find_nearest(unit, resource_deposits)
                    if res and unit.distance_to(res.x, res.y) == 1:
                        d = self._direction_to_adjacent(unit.x, unit.y, res.x, res.y)
                        if d is not None:
                            cell_action = self._encode_action(
                                ActionType.HARVEST, harvest_dir=d
                            )
                    elif res:
                        cell_action = self._move_toward(unit, res.x, res.y, gm)

            elif unit.unit_type in (UnitType.LIGHT, UnitType.HEAVY, UnitType.RANGED):
                # Attack nearest enemy
                enemy = self._find_nearest(unit, enemy_units)
                if enemy:
                    if unit.in_attack_range(enemy.x, enemy.y):
                        rel_x = enemy.x - unit.x + MAX_ATTACK_RANGE // 2
                        rel_y = enemy.y - unit.y + MAX_ATTACK_RANGE // 2
                        attack_idx = rel_y * MAX_ATTACK_RANGE + rel_x
                        cell_action = self._encode_action(
                            ActionType.ATTACK, attack_pos=attack_idx
                        )
                    else:
                        cell_action = self._move_toward(unit, enemy.x, enemy.y, gm)

            if cell_action:
                self._set_cell_action(actions, unit.x, unit.y, w, cell_action)

        return actions


class EconomyAI(BaseAI):
    """
    Economy-focused strategy:
    1. Build up to 4 workers
    2. Harvest aggressively
    3. Build barracks when 10+ resources
    4. Produce heavy units (high damage)
    5. Attack when 3+ combat units
    """

    def get_action(self, state: GameState, player: int) -> np.ndarray:
        h, w = state.game_map.height, state.game_map.width
        actions = np.zeros(h * w * DIMS_PER_CELL, dtype=np.int32)
        gm = state.game_map
        resources = state.player_resources[player]

        workers = gm.get_units_of_type(player, UnitType.WORKER)
        bases = gm.get_units_of_type(player, UnitType.BASE)
        barracks = gm.get_units_of_type(player, UnitType.BARRACKS)
        combat_units = (gm.get_units_of_type(player, UnitType.LIGHT)
                        + gm.get_units_of_type(player, UnitType.HEAVY)
                        + gm.get_units_of_type(player, UnitType.RANGED))
        enemy_units = gm.get_player_units(1 - player)
        resource_deposits = gm.get_resources()

        for unit in gm.get_player_units(player):
            if not unit.is_idle:
                continue

            cell_action = None

            if unit.unit_type == UnitType.BASE:
                if len(workers) < 4 and resources >= UNIT_STATS[UnitType.WORKER].cost:
                    d = self._find_empty_adjacent_dir(unit, gm)
                    if d is not None:
                        cell_action = self._encode_action(
                            ActionType.PRODUCE, produce_dir=d,
                            produce_type=UnitType.WORKER
                        )

            elif unit.unit_type == UnitType.BARRACKS:
                if resources >= UNIT_STATS[UnitType.HEAVY].cost:
                    d = self._find_empty_adjacent_dir(unit, gm)
                    if d is not None:
                        cell_action = self._encode_action(
                            ActionType.PRODUCE, produce_dir=d,
                            produce_type=UnitType.HEAVY
                        )

            elif unit.unit_type == UnitType.WORKER:
                # First worker tries to build barracks if enough resources
                if (workers and unit.unit_id == workers[0].unit_id
                        and not barracks
                        and resources >= UNIT_STATS[UnitType.BARRACKS].cost):
                    d = self._find_empty_adjacent_dir(unit, gm)
                    if d is not None:
                        cell_action = self._encode_action(
                            ActionType.PRODUCE, produce_dir=d,
                            produce_type=UnitType.BARRACKS
                        )
                elif unit.resources_carried > 0:
                    base = self._find_nearest(unit, bases)
                    if base and unit.distance_to(base.x, base.y) == 1:
                        d = self._direction_to_adjacent(unit.x, unit.y, base.x, base.y)
                        if d is not None:
                            cell_action = self._encode_action(ActionType.RETURN, return_dir=d)
                    elif base:
                        cell_action = self._move_toward(unit, base.x, base.y, gm)
                else:
                    res = self._find_nearest(unit, resource_deposits)
                    if res and unit.distance_to(res.x, res.y) == 1:
                        d = self._direction_to_adjacent(unit.x, unit.y, res.x, res.y)
                        if d is not None:
                            cell_action = self._encode_action(ActionType.HARVEST, harvest_dir=d)
                    elif res:
                        cell_action = self._move_toward(unit, res.x, res.y, gm)

            elif unit.unit_type in (UnitType.LIGHT, UnitType.HEAVY, UnitType.RANGED):
                # Attack when we have enough units
                if len(combat_units) >= 3 and enemy_units:
                    enemy = self._find_nearest(unit, enemy_units)
                    if enemy and unit.in_attack_range(enemy.x, enemy.y):
                        rel_x = enemy.x - unit.x + MAX_ATTACK_RANGE // 2
                        rel_y = enemy.y - unit.y + MAX_ATTACK_RANGE // 2
                        attack_idx = rel_y * MAX_ATTACK_RANGE + rel_x
                        cell_action = self._encode_action(ActionType.ATTACK, attack_pos=attack_idx)
                    elif enemy:
                        cell_action = self._move_toward(unit, enemy.x, enemy.y, gm)

            if cell_action:
                self._set_cell_action(actions, unit.x, unit.y, w, cell_action)

        return actions


class DefensiveAI(BaseAI):
    """
    Defensive strategy:
    1. Build 3 workers, harvest
    2. Build barracks
    3. Produce ranged units (stay back and shoot)
    4. Only attack if enemy gets close or we have overwhelming force
    """

    def get_action(self, state: GameState, player: int) -> np.ndarray:
        h, w = state.game_map.height, state.game_map.width
        actions = np.zeros(h * w * DIMS_PER_CELL, dtype=np.int32)
        gm = state.game_map
        resources = state.player_resources[player]

        workers = gm.get_units_of_type(player, UnitType.WORKER)
        bases = gm.get_units_of_type(player, UnitType.BASE)
        barracks = gm.get_units_of_type(player, UnitType.BARRACKS)
        combat_units = (gm.get_units_of_type(player, UnitType.LIGHT)
                        + gm.get_units_of_type(player, UnitType.HEAVY)
                        + gm.get_units_of_type(player, UnitType.RANGED))
        enemy_units = gm.get_player_units(1 - player)
        resource_deposits = gm.get_resources()

        # Determine base position for defense reference
        base = bases[0] if bases else None
        base_x = base.x if base else w // 2
        base_y = base.y if base else h // 2

        for unit in gm.get_player_units(player):
            if not unit.is_idle:
                continue

            cell_action = None

            if unit.unit_type == UnitType.BASE:
                if len(workers) < 3 and resources >= UNIT_STATS[UnitType.WORKER].cost:
                    d = self._find_empty_adjacent_dir(unit, gm)
                    if d is not None:
                        cell_action = self._encode_action(
                            ActionType.PRODUCE, produce_dir=d,
                            produce_type=UnitType.WORKER
                        )

            elif unit.unit_type == UnitType.BARRACKS:
                if resources >= UNIT_STATS[UnitType.RANGED].cost:
                    d = self._find_empty_adjacent_dir(unit, gm)
                    if d is not None:
                        cell_action = self._encode_action(
                            ActionType.PRODUCE, produce_dir=d,
                            produce_type=UnitType.RANGED
                        )

            elif unit.unit_type == UnitType.WORKER:
                if (workers and unit.unit_id == workers[0].unit_id
                        and not barracks
                        and resources >= UNIT_STATS[UnitType.BARRACKS].cost):
                    d = self._find_empty_adjacent_dir(unit, gm)
                    if d is not None:
                        cell_action = self._encode_action(
                            ActionType.PRODUCE, produce_dir=d,
                            produce_type=UnitType.BARRACKS
                        )
                elif unit.resources_carried > 0:
                    nearest_base = self._find_nearest(unit, bases)
                    if nearest_base and unit.distance_to(nearest_base.x, nearest_base.y) == 1:
                        d = self._direction_to_adjacent(unit.x, unit.y, nearest_base.x, nearest_base.y)
                        if d is not None:
                            cell_action = self._encode_action(ActionType.RETURN, return_dir=d)
                    elif nearest_base:
                        cell_action = self._move_toward(unit, nearest_base.x, nearest_base.y, gm)
                else:
                    res = self._find_nearest(unit, resource_deposits)
                    if res and unit.distance_to(res.x, res.y) == 1:
                        d = self._direction_to_adjacent(unit.x, unit.y, res.x, res.y)
                        if d is not None:
                            cell_action = self._encode_action(ActionType.HARVEST, harvest_dir=d)
                    elif res:
                        cell_action = self._move_toward(unit, res.x, res.y, gm)

            elif unit.unit_type in (UnitType.LIGHT, UnitType.HEAVY, UnitType.RANGED):
                # Check for nearby enemies
                nearby_enemy = None
                for eu in enemy_units:
                    if unit.distance_to(eu.x, eu.y) <= 5:
                        nearby_enemy = eu
                        break

                if nearby_enemy and unit.in_attack_range(nearby_enemy.x, nearby_enemy.y):
                    rel_x = nearby_enemy.x - unit.x + MAX_ATTACK_RANGE // 2
                    rel_y = nearby_enemy.y - unit.y + MAX_ATTACK_RANGE // 2
                    attack_idx = rel_y * MAX_ATTACK_RANGE + rel_x
                    cell_action = self._encode_action(ActionType.ATTACK, attack_pos=attack_idx)
                elif nearby_enemy:
                    cell_action = self._move_toward(unit, nearby_enemy.x, nearby_enemy.y, gm)
                elif len(combat_units) >= 5:
                    # Overwhelming force: attack
                    enemy = self._find_nearest(unit, enemy_units)
                    if enemy:
                        cell_action = self._move_toward(unit, enemy.x, enemy.y, gm)
                else:
                    # Stay near base
                    if base and unit.distance_to(base_x, base_y) > 3:
                        cell_action = self._move_toward(unit, base_x, base_y, gm)

            if cell_action:
                self._set_cell_action(actions, unit.x, unit.y, w, cell_action)

        return actions


class PPOAgentWrapper(BaseAI):
    """
    Wraps a trained PPO agent to match the scripted AI interface.

    This allows the trained RL agent to be used anywhere a scripted AI
    can be used — including the visualizer.

    Usage:
        wrapper = PPOAgentWrapper("checkpoints/")
        action = wrapper.get_action(state, player=0)
    """

    def __init__(self, checkpoint_path: str, map_size: int = 8):
        # Lazy import to avoid circular dependency (rts_ai imports game)
        from rts_ai.agent import PPOAgent

        self.agent = PPOAgent(map_height=map_size, map_width=map_size)
        self.agent.load(checkpoint_path)

    def get_action(self, state: GameState, player: int) -> np.ndarray:
        """Get action from the trained PPO policy."""
        obs = state.get_observation(player)
        mask = state.get_action_mask(player)

        # Add batch dimension (policy expects batched input)
        obs_batch = obs[np.newaxis, ...]
        mask_batch = mask[np.newaxis, ...]

        actions, _, _ = self.agent.policy.get_action(
            obs_batch, mask_batch, deterministic=True
        )

        # Remove batch dim and flatten to (H*W*DIMS_PER_CELL,)
        return actions[0].reshape(-1)


class PatternAgentWrapper(BaseAI):
    """
    Wraps a trained pattern-based AI to match the scripted AI interface.

    Uses the four pillars of learning (no neural network) to select
    strategies and generate actions.

    Usage:
        wrapper = PatternAgentWrapper("checkpoints_pattern/")
        action = wrapper.get_action(state, player=0)
    """

    def __init__(self, checkpoint_path: str, map_size: int = 8):
        from rts_ai.pattern_policy import PatternPolicy

        self.policy = PatternPolicy(map_height=map_size, map_width=map_size)
        self.policy.load(checkpoint_path)
        # Disable exploration during playback
        self.policy.explorer.exploration_rate = 0.0

    def get_action(self, state: GameState, player: int) -> np.ndarray:
        """Get action from the trained pattern policy."""
        return self.policy.get_action(state, player)
