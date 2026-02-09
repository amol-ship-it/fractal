"""
Action System - Action types, encoding/decoding, and validation.

Each unit can be given one action per tick. Actions are durative (they take
multiple game ticks to complete). Once assigned, actions run to completion.

Action encoding follows MicroRTS GridNet format:
Each cell gets 7 action components:
  [action_type, move_dir, harvest_dir, return_dir, produce_dir, produce_type, attack_pos]
"""

from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, Tuple, List

from game.units import UnitType, Unit, UNIT_STATS, PRODUCTION_TABLE, BUILD_TABLE


class ActionType(IntEnum):
    NOOP = 0
    MOVE = 1
    HARVEST = 2
    RETURN = 3
    PRODUCE = 4
    ATTACK = 5


class Direction(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


# Direction offsets: (dx, dy)
DIR_OFFSETS = {
    Direction.NORTH: (0, -1),
    Direction.EAST: (1, 0),
    Direction.SOUTH: (0, 1),
    Direction.WEST: (-1, 0),
}


@dataclass
class Action:
    """A single action assigned to a unit."""
    unit_id: int
    action_type: ActionType
    direction: Optional[Direction] = None   # For move/harvest/return/produce
    produce_type: Optional[UnitType] = None  # What to produce
    target_x: Optional[int] = None          # For attack
    target_y: Optional[int] = None          # For attack

    def target_position(self, unit: Unit) -> Optional[Tuple[int, int]]:
        """Get the target position for directional actions."""
        if self.direction is not None:
            dx, dy = DIR_OFFSETS[self.direction]
            return (unit.x + dx, unit.y + dy)
        if self.target_x is not None and self.target_y is not None:
            return (self.target_x, self.target_y)
        return None


# Action space dimensions per cell (matches MicroRTS GridNet)
NUM_ACTION_TYPES = 6       # NOOP, move, harvest, return, produce, attack
NUM_DIRECTIONS = 4         # N, E, S, W
NUM_PRODUCE_TYPES = 7      # resource(unused), base, barracks, worker, light, heavy, ranged
MAX_ATTACK_RANGE = 7       # Positions within max attack range


def get_action_space_dims() -> List[int]:
    """Return the multi-discrete action space dimensions per cell."""
    return [
        NUM_ACTION_TYPES,     # action_type
        NUM_DIRECTIONS,       # move direction
        NUM_DIRECTIONS,       # harvest direction
        NUM_DIRECTIONS,       # return direction
        NUM_DIRECTIONS,       # produce direction
        NUM_PRODUCE_TYPES,    # produce unit type
        MAX_ATTACK_RANGE * MAX_ATTACK_RANGE,  # attack position (relative)
    ]


DIMS_PER_CELL = len(get_action_space_dims())  # 7


def decode_cell_action(cell_action: List[int], unit: Unit,
                       cell_x: int, cell_y: int) -> Optional[Action]:
    """
    Decode a cell's action vector into an Action object.

    cell_action: [action_type, move_dir, harvest_dir, return_dir,
                  produce_dir, produce_type, attack_pos]
    """
    if len(cell_action) != DIMS_PER_CELL:
        return None

    action_type = ActionType(cell_action[0])

    if action_type == ActionType.NOOP:
        return Action(unit_id=unit.unit_id, action_type=ActionType.NOOP)

    if action_type == ActionType.MOVE:
        direction = Direction(cell_action[1])
        return Action(unit_id=unit.unit_id, action_type=ActionType.MOVE,
                      direction=direction)

    if action_type == ActionType.HARVEST:
        direction = Direction(cell_action[2])
        return Action(unit_id=unit.unit_id, action_type=ActionType.HARVEST,
                      direction=direction)

    if action_type == ActionType.RETURN:
        direction = Direction(cell_action[3])
        return Action(unit_id=unit.unit_id, action_type=ActionType.RETURN,
                      direction=direction)

    if action_type == ActionType.PRODUCE:
        direction = Direction(cell_action[4])
        produce_type = UnitType(cell_action[5])
        return Action(unit_id=unit.unit_id, action_type=ActionType.PRODUCE,
                      direction=direction, produce_type=produce_type)

    if action_type == ActionType.ATTACK:
        # Decode relative attack position
        attack_idx = cell_action[6]
        rel_x = (attack_idx % MAX_ATTACK_RANGE) - MAX_ATTACK_RANGE // 2
        rel_y = (attack_idx // MAX_ATTACK_RANGE) - MAX_ATTACK_RANGE // 2
        return Action(unit_id=unit.unit_id, action_type=ActionType.ATTACK,
                      target_x=cell_x + rel_x, target_y=cell_y + rel_y)

    return None


def get_valid_actions_mask(unit: Unit, game_map, player_resources: int) -> List[List[int]]:
    """
    Generate a valid action mask for a single unit.

    Returns a list of 7 binary masks, one per action component.
    """
    from game.game_map import GameMap

    dims = get_action_space_dims()
    masks = [[0] * d for d in dims]

    # NOOP is always valid
    masks[0][ActionType.NOOP] = 1

    if not unit.is_idle:
        return masks  # Busy units can only NOOP

    # MOVE
    if unit.can_move:
        for d in Direction:
            dx, dy = DIR_OFFSETS[d]
            nx, ny = unit.x + dx, unit.y + dy
            if game_map.is_empty(nx, ny):
                masks[0][ActionType.MOVE] = 1
                masks[1][d] = 1

    # HARVEST (workers only, must be empty-handed, adjacent to resource)
    if unit.can_harvest:
        for d in Direction:
            dx, dy = DIR_OFFSETS[d]
            nx, ny = unit.x + dx, unit.y + dy
            target = game_map.get_unit_at(nx, ny)
            if target and target.unit_type == UnitType.RESOURCE:
                masks[0][ActionType.HARVEST] = 1
                masks[2][d] = 1

    # RETURN (workers carrying resources, adjacent to own base)
    if unit.can_return:
        for d in Direction:
            dx, dy = DIR_OFFSETS[d]
            nx, ny = unit.x + dx, unit.y + dy
            target = game_map.get_unit_at(nx, ny)
            if (target and target.unit_type == UnitType.BASE
                    and target.player == unit.player):
                masks[0][ActionType.RETURN] = 1
                masks[3][d] = 1

    # PRODUCE (bases and barracks)
    if unit.can_produce and unit.unit_type in PRODUCTION_TABLE:
        produceable = PRODUCTION_TABLE[unit.unit_type]
        for d in Direction:
            dx, dy = DIR_OFFSETS[d]
            nx, ny = unit.x + dx, unit.y + dy
            if game_map.is_empty(nx, ny):
                for pt in produceable:
                    if player_resources >= UNIT_STATS[pt].cost:
                        masks[0][ActionType.PRODUCE] = 1
                        masks[4][d] = 1
                        masks[5][pt] = 1

    # PRODUCE for workers (build structures)
    if unit.can_build:
        for d in Direction:
            dx, dy = DIR_OFFSETS[d]
            nx, ny = unit.x + dx, unit.y + dy
            if game_map.is_empty(nx, ny):
                for bt in BUILD_TABLE:
                    if player_resources >= UNIT_STATS[bt].cost:
                        masks[0][ActionType.PRODUCE] = 1
                        masks[4][d] = 1
                        masks[5][bt] = 1

    # ATTACK
    if unit.can_attack:
        enemy_player = 1 - unit.player
        targets = game_map.units_in_range(
            unit.x, unit.y, unit.stats.attack_range, enemy_player
        )
        for target in targets:
            rel_x = target.x - unit.x + MAX_ATTACK_RANGE // 2
            rel_y = target.y - unit.y + MAX_ATTACK_RANGE // 2
            if 0 <= rel_x < MAX_ATTACK_RANGE and 0 <= rel_y < MAX_ATTACK_RANGE:
                attack_idx = rel_y * MAX_ATTACK_RANGE + rel_x
                masks[0][ActionType.ATTACK] = 1
                masks[6][attack_idx] = 1

    return masks
