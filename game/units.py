"""
Unit System - All unit types, stats, and combat mechanics for the RTS game.

Modeled after MicroRTS unit definitions:
- Resource: Neutral harvestable resource deposit
- Base: Produces workers, stores returned resources
- Barracks: Produces combat units (light, heavy, ranged)
- Worker: Harvests resources, builds structures, weak combat
- Light: Fast melee combat unit
- Heavy: Slow, high-damage melee combat unit
- Ranged: Fragile ranged combat unit
"""

from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional, Tuple


class UnitType(IntEnum):
    RESOURCE = 0
    BASE = 1
    BARRACKS = 2
    WORKER = 3
    LIGHT = 4
    HEAVY = 5
    RANGED = 6


@dataclass
class UnitStats:
    """Immutable stats for a unit type."""
    hp: int
    cost: int
    build_time: int        # Ticks to produce
    move_time: int         # Ticks per move (0 = immobile)
    attack_time: int       # Ticks per attack
    damage: int
    attack_range: int      # 0 = cannot attack, 1 = melee, >1 = ranged
    harvest_time: int      # Ticks to harvest (0 = cannot harvest)
    return_time: int       # Ticks to return resource (0 = cannot return)
    can_build: bool        # Can build structures
    can_produce: bool      # Can produce units
    is_structure: bool


# Unit stats table matching MicroRTS definitions
UNIT_STATS = {
    UnitType.RESOURCE: UnitStats(
        hp=1, cost=0, build_time=0,
        move_time=0, attack_time=0, damage=0, attack_range=0,
        harvest_time=0, return_time=0,
        can_build=False, can_produce=False, is_structure=False,
    ),
    UnitType.BASE: UnitStats(
        hp=10, cost=10, build_time=250,
        move_time=0, attack_time=0, damage=0, attack_range=0,
        harvest_time=0, return_time=0,
        can_build=False, can_produce=True, is_structure=True,
    ),
    UnitType.BARRACKS: UnitStats(
        hp=4, cost=5, build_time=200,
        move_time=0, attack_time=0, damage=0, attack_range=0,
        harvest_time=0, return_time=0,
        can_build=False, can_produce=True, is_structure=True,
    ),
    UnitType.WORKER: UnitStats(
        hp=1, cost=1, build_time=50,
        move_time=10, attack_time=5, damage=1, attack_range=1,
        harvest_time=20, return_time=10,
        can_build=True, can_produce=False, is_structure=False,
    ),
    UnitType.LIGHT: UnitStats(
        hp=4, cost=2, build_time=80,
        move_time=8, attack_time=5, damage=2, attack_range=1,
        harvest_time=0, return_time=0,
        can_build=False, can_produce=False, is_structure=False,
    ),
    UnitType.HEAVY: UnitStats(
        hp=4, cost=3, build_time=120,
        move_time=12, attack_time=5, damage=4, attack_range=1,
        harvest_time=0, return_time=0,
        can_build=False, can_produce=False, is_structure=False,
    ),
    UnitType.RANGED: UnitStats(
        hp=1, cost=2, build_time=100,
        move_time=12, attack_time=5, damage=1, attack_range=3,
        harvest_time=0, return_time=0,
        can_build=False, can_produce=False, is_structure=False,
    ),
}

# What each producer can produce
PRODUCTION_TABLE = {
    UnitType.BASE: [UnitType.WORKER],
    UnitType.BARRACKS: [UnitType.LIGHT, UnitType.HEAVY, UnitType.RANGED],
}

# What workers can build
BUILD_TABLE = [UnitType.BASE, UnitType.BARRACKS]


class ActionState(IntEnum):
    """Current action state of a unit."""
    IDLE = 0
    MOVING = 1
    HARVESTING = 2
    RETURNING = 3
    PRODUCING = 4
    ATTACKING = 5
    BUILDING = 6


@dataclass
class Unit:
    """A unit instance in the game."""
    unit_id: int
    unit_type: UnitType
    player: int            # 0 or 1
    x: int
    y: int
    hp: int = -1           # -1 means use max from stats
    resources_carried: int = 0

    # Action state
    action_state: ActionState = ActionState.IDLE
    action_target: Optional[Tuple[int, int]] = None
    action_ticks_remaining: int = 0
    producing_type: Optional[UnitType] = None

    def __post_init__(self):
        if self.hp == -1:
            self.hp = self.stats.hp

    @property
    def stats(self) -> UnitStats:
        return UNIT_STATS[self.unit_type]

    @property
    def is_idle(self) -> bool:
        return self.action_state == ActionState.IDLE

    @property
    def is_alive(self) -> bool:
        return self.hp > 0

    @property
    def is_structure(self) -> bool:
        return self.stats.is_structure

    @property
    def can_move(self) -> bool:
        return self.stats.move_time > 0 and self.is_idle

    @property
    def can_attack(self) -> bool:
        return self.stats.attack_range > 0 and self.is_idle

    @property
    def can_harvest(self) -> bool:
        return (self.unit_type == UnitType.WORKER
                and self.is_idle
                and self.resources_carried == 0)

    @property
    def can_return(self) -> bool:
        return (self.unit_type == UnitType.WORKER
                and self.is_idle
                and self.resources_carried > 0)

    @property
    def can_produce(self) -> bool:
        return self.stats.can_produce and self.is_idle

    @property
    def can_build(self) -> bool:
        return self.stats.can_build and self.is_idle and self.resources_carried == 0

    def take_damage(self, damage: int):
        """Apply damage to this unit."""
        self.hp = max(0, self.hp - damage)

    def distance_to(self, x: int, y: int) -> int:
        """Manhattan distance to a position."""
        return abs(self.x - x) + abs(self.y - y)

    def in_attack_range(self, x: int, y: int) -> bool:
        """Check if a position is within attack range."""
        return self.distance_to(x, y) <= self.stats.attack_range
