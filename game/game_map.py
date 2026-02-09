"""
Game Map - Grid-based map with terrain, resources, and units.

Maps are square grids where each cell can contain:
- Empty terrain (passable)
- Wall terrain (impassable)
- A resource deposit
- A unit (only one unit per cell)
"""

from typing import List, Optional, Tuple, Dict
from enum import IntEnum

from game.units import Unit, UnitType, UNIT_STATS


class Terrain(IntEnum):
    EMPTY = 0
    WALL = 1


class GameMap:
    """Grid-based RTS game map."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.terrain: List[List[int]] = [
            [Terrain.EMPTY] * width for _ in range(height)
        ]
        self.units: Dict[int, Unit] = {}  # unit_id -> Unit
        self._next_unit_id = 0
        # Spatial index: (x, y) -> unit_id
        self._pos_index: Dict[Tuple[int, int], int] = {}

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_passable(self, x: int, y: int) -> bool:
        """Check if a cell is passable (in bounds, not wall, not occupied)."""
        if not self.in_bounds(x, y):
            return False
        if self.terrain[y][x] == Terrain.WALL:
            return False
        return True

    def is_empty(self, x: int, y: int) -> bool:
        """Check if a cell is passable and has no unit."""
        return self.is_passable(x, y) and (x, y) not in self._pos_index

    def get_unit_at(self, x: int, y: int) -> Optional[Unit]:
        """Get unit at position, or None."""
        uid = self._pos_index.get((x, y))
        if uid is not None:
            return self.units.get(uid)
        return None

    def add_unit(self, unit_type: UnitType, player: int, x: int, y: int,
                 hp: int = -1) -> Optional[Unit]:
        """Add a new unit to the map. Returns the unit or None if cell occupied."""
        if not self.in_bounds(x, y):
            return None
        if (x, y) in self._pos_index:
            return None

        uid = self._next_unit_id
        self._next_unit_id += 1
        unit = Unit(unit_id=uid, unit_type=unit_type, player=player,
                    x=x, y=y, hp=hp)
        self.units[uid] = unit
        self._pos_index[(x, y)] = uid
        return unit

    def remove_unit(self, unit_id: int):
        """Remove a unit from the map."""
        unit = self.units.pop(unit_id, None)
        if unit:
            self._pos_index.pop((unit.x, unit.y), None)

    def move_unit(self, unit_id: int, new_x: int, new_y: int) -> bool:
        """Move a unit to a new position. Returns success."""
        unit = self.units.get(unit_id)
        if not unit:
            return False
        if not self.is_empty(new_x, new_y):
            return False

        self._pos_index.pop((unit.x, unit.y), None)
        unit.x = new_x
        unit.y = new_y
        self._pos_index[(new_x, new_y)] = unit_id
        return True

    def get_player_units(self, player: int) -> List[Unit]:
        """Get all units belonging to a player."""
        return [u for u in self.units.values() if u.player == player and u.is_alive]

    def get_units_of_type(self, player: int, unit_type: UnitType) -> List[Unit]:
        """Get all units of a specific type for a player."""
        return [u for u in self.units.values()
                if u.player == player and u.unit_type == unit_type and u.is_alive]

    def get_resources(self) -> List[Unit]:
        """Get all resource deposits (neutral units)."""
        return [u for u in self.units.values()
                if u.unit_type == UnitType.RESOURCE and u.is_alive]

    def adjacent_positions(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get adjacent positions (N, E, S, W)."""
        dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W
        return [(x + dx, y + dy) for dx, dy in dirs
                if self.in_bounds(x + dx, y + dy)]

    def empty_adjacent(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get empty adjacent positions."""
        return [(ax, ay) for ax, ay in self.adjacent_positions(x, y)
                if self.is_empty(ax, ay)]

    def units_in_range(self, x: int, y: int, attack_range: int,
                       enemy_player: int) -> List[Unit]:
        """Get enemy units within attack range."""
        targets = []
        for dx in range(-attack_range, attack_range + 1):
            for dy in range(-attack_range, attack_range + 1):
                if abs(dx) + abs(dy) > attack_range or (dx == 0 and dy == 0):
                    continue
                tx, ty = x + dx, y + dy
                target = self.get_unit_at(tx, ty)
                if target and target.player == enemy_player and target.is_alive:
                    targets.append(target)
        return targets

    def set_wall(self, x: int, y: int):
        """Set a cell as wall terrain."""
        if self.in_bounds(x, y):
            self.terrain[y][x] = Terrain.WALL

    @classmethod
    def create_standard_map(cls, size: int = 8) -> 'GameMap':
        """
        Create a standard symmetric 2-player map.
        Player 0 starts top-left, Player 1 starts bottom-right.
        Resources in predictable positions.
        """
        gm = cls(size, size)

        # Player 0: base top-left, worker nearby
        gm.add_unit(UnitType.BASE, player=0, x=1, y=1)
        gm.add_unit(UnitType.WORKER, player=0, x=2, y=1)

        # Player 1: base bottom-right, worker nearby
        gm.add_unit(UnitType.BASE, player=1, x=size - 2, y=size - 2)
        gm.add_unit(UnitType.WORKER, player=1, x=size - 3, y=size - 2)

        # Resources: symmetrically placed
        resource_positions = [
            (0, 0), (1, 0), (0, 1),
            (size - 1, size - 1), (size - 2, size - 1), (size - 1, size - 2),
        ]
        # Mid-map resources
        mid = size // 2
        resource_positions.extend([
            (mid - 1, mid - 1), (mid, mid - 1),
            (mid - 1, mid), (mid, mid),
        ])

        for rx, ry in resource_positions:
            if gm.is_empty(rx, ry):
                gm.add_unit(UnitType.RESOURCE, player=-1, x=rx, y=ry)

        return gm

    @classmethod
    def create_16x16_map(cls) -> 'GameMap':
        """Create a larger 16x16 map with more resources."""
        gm = cls(16, 16)

        # Player 0: top-left corner
        gm.add_unit(UnitType.BASE, player=0, x=2, y=2)
        gm.add_unit(UnitType.WORKER, player=0, x=3, y=2)
        gm.add_unit(UnitType.WORKER, player=0, x=2, y=3)

        # Player 1: bottom-right corner
        gm.add_unit(UnitType.BASE, player=1, x=13, y=13)
        gm.add_unit(UnitType.WORKER, player=1, x=12, y=13)
        gm.add_unit(UnitType.WORKER, player=1, x=13, y=12)

        # Corner resources for each player
        for rx, ry in [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (2, 0)]:
            if gm.is_empty(rx, ry):
                gm.add_unit(UnitType.RESOURCE, player=-1, x=rx, y=ry)
        for rx, ry in [(15, 15), (14, 15), (15, 14), (14, 14), (15, 13), (13, 15)]:
            if gm.is_empty(rx, ry):
                gm.add_unit(UnitType.RESOURCE, player=-1, x=rx, y=ry)

        # Center resources
        for rx, ry in [(7, 7), (8, 7), (7, 8), (8, 8)]:
            if gm.is_empty(rx, ry):
                gm.add_unit(UnitType.RESOURCE, player=-1, x=rx, y=ry)

        return gm
