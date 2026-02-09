"""
Pattern-Based Policy - Uses the four pillars of learning to play the RTS game.

Instead of a neural network, this policy:
1. Encodes the game state as a feature vector
2. Matches the situation against known patterns (PatternEngine + DualMemory)
3. Selects a strategy based on which pattern matched and what worked before
4. Executes the strategy as concrete per-unit actions
5. Learns from feedback (wins/losses refine pattern confidence)

The four pillars drive every decision:
- Feedback Loops: Win/loss outcomes refine which situation->strategy pairs work
- Approximability: Pattern signatures improve with each game played
- Composability: Simple patterns compose into higher-level strategic concepts
- Exploration: Agent occasionally tries novel strategies to discover better ones
"""

import random
import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import IntEnum

from core.pattern import Pattern
from core.engine import PatternEngine
from core.memory import DualMemory
from core.feedback import (
    FeedbackLoop, FeedbackType, FeedbackSignal, ExplorationStrategy
)

from rts_ai.encoder import GameStateEncoder
from game.game_state import GameState
from game.game_map import GameMap
from game.units import Unit, UnitType, UNIT_STATS, PRODUCTION_TABLE
from game.actions import (
    ActionType, Direction, DIR_OFFSETS, DIMS_PER_CELL,
    MAX_ATTACK_RANGE,
)


# ── Strategies ────────────────────────────────────────────────────────────

class Strategy(IntEnum):
    """
    The action vocabulary. The agent learns WHEN to use each strategy;
    the strategies define WHAT each unit type does.
    """
    HARVEST = 0          # Focus on gathering resources
    BUILD_BARRACKS = 1   # Build infrastructure
    PRODUCE_LIGHT = 2    # Produce fast melee units
    PRODUCE_HEAVY = 3    # Produce slow high-damage units
    PRODUCE_RANGED = 4   # Produce fragile ranged units
    ATTACK = 5           # Send combat units to rush enemy


STRATEGY_NAMES = [s.name for s in Strategy]
NUM_STRATEGIES = len(Strategy)


# ── Strategy Binding ──────────────────────────────────────────────────────

class StrategyBinding:
    """
    Links a situation pattern to a strategy, tracking how well it performs.
    This is what the agent LEARNS — which situation calls for which strategy.
    """

    def __init__(self, strategy: int, pattern_id: str):
        self.strategy = strategy
        self.pattern_id = pattern_id
        self.wins = 0
        self.losses = 0
        self.times_used = 0

    @property
    def confidence(self) -> float:
        total = self.wins + self.losses
        if total == 0:
            return 0.5  # Uninformed prior
        # Bayesian smoothing: blend observed win rate with prior (0.5)
        # using a pseudo-count of 2. This prevents premature lock-in
        # from a single win (1/1 = 1.0) or single loss (0/1 = 0.0).
        # With pseudo_count=2: after 1 win, confidence = 2/3 ≈ 0.67
        # After 1 loss: 1/3 ≈ 0.33. After 10 wins: 11/12 ≈ 0.92
        # The smoothing fades as more data accumulates.
        pseudo_count = 2
        return (self.wins + pseudo_count * 0.5) / (total + pseudo_count)

    def to_dict(self) -> dict:
        return {
            'strategy': self.strategy,
            'pattern_id': self.pattern_id,
            'wins': self.wins,
            'losses': self.losses,
            'times_used': self.times_used,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'StrategyBinding':
        b = cls(data['strategy'], data['pattern_id'])
        b.wins = data.get('wins', 0)
        b.losses = data.get('losses', 0)
        b.times_used = data.get('times_used', 0)
        return b


# ── Action Helpers (same logic as BaseAI) ─────────────────────────────────

def _encode_action(action_type: int, move_dir: int = 0,
                   harvest_dir: int = 0, return_dir: int = 0,
                   produce_dir: int = 0, produce_type: int = 0,
                   attack_pos: int = 0) -> List[int]:
    return [action_type, move_dir, harvest_dir, return_dir,
            produce_dir, produce_type, attack_pos]


def _set_cell_action(actions: np.ndarray, x: int, y: int,
                     width: int, cell_action: List[int]):
    idx = (y * width + x) * DIMS_PER_CELL
    for i, v in enumerate(cell_action):
        actions[idx + i] = v


def _find_nearest(unit: Unit, targets: List[Unit]) -> Optional[Unit]:
    if not targets:
        return None
    return min(targets, key=lambda t: unit.distance_to(t.x, t.y))


def _direction_to_adjacent(ux: int, uy: int,
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


def _find_empty_adjacent_dir(unit: Unit, game_map: GameMap) -> Optional[Direction]:
    for d in Direction:
        dx, dy = DIR_OFFSETS[d]
        nx, ny = unit.x + dx, unit.y + dy
        if game_map.is_empty(nx, ny):
            return d
    return None


def _move_toward(unit: Unit, tx: int, ty: int,
                 game_map: GameMap) -> Optional[List[int]]:
    """Try to move toward (tx,ty), trying alternate directions if blocked."""
    dx_total = tx - unit.x
    dy_total = ty - unit.y
    candidates = []
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
            return _encode_action(ActionType.MOVE, move_dir=d)
    return None


# ── Worker Behavior (shared across all strategies) ────────────────────────

def _worker_harvest_cycle(unit: Unit, state: GameState, player: int,
                          game_map: GameMap) -> Optional[List[int]]:
    """Standard worker behavior: harvest → return → repeat."""
    bases = game_map.get_units_of_type(player, UnitType.BASE)
    resources = game_map.get_resources()

    if unit.resources_carried > 0:
        # Return to base
        base = _find_nearest(unit, bases)
        if base and unit.distance_to(base.x, base.y) == 1:
            d = _direction_to_adjacent(unit.x, unit.y, base.x, base.y)
            if d is not None:
                return _encode_action(ActionType.RETURN, return_dir=d)
        elif base:
            return _move_toward(unit, base.x, base.y, game_map)
    else:
        # Go harvest
        res = _find_nearest(unit, resources)
        if res and unit.distance_to(res.x, res.y) == 1:
            d = _direction_to_adjacent(unit.x, unit.y, res.x, res.y)
            if d is not None:
                return _encode_action(ActionType.HARVEST, harvest_dir=d)
        elif res:
            return _move_toward(unit, res.x, res.y, game_map)
    return None


# ── Strategy Executors ────────────────────────────────────────────────────

def _execute_harvest(state: GameState, player: int) -> np.ndarray:
    """HARVEST: maximize resource gathering. Produce extra workers, defend if attacked."""
    gm = state.game_map
    h, w = gm.height, gm.width
    actions = np.zeros(h * w * DIMS_PER_CELL, dtype=np.int32)
    resources = state.player_resources[player]
    workers = gm.get_units_of_type(player, UnitType.WORKER)
    enemy_units = gm.get_player_units(1 - player)

    for unit in gm.get_player_units(player):
        if not unit.is_idle:
            continue
        cell_action = None

        if unit.unit_type == UnitType.BASE:
            # Produce up to 3 workers for maximum economy
            if len(workers) < 3 and resources >= UNIT_STATS[UnitType.WORKER].cost:
                d = _find_empty_adjacent_dir(unit, gm)
                if d is not None:
                    cell_action = _encode_action(
                        ActionType.PRODUCE, produce_dir=d,
                        produce_type=UnitType.WORKER)

        elif unit.unit_type == UnitType.WORKER:
            cell_action = _worker_harvest_cycle(unit, state, player, gm)

        # Combat units: defend base — attack nearby enemies
        elif unit.unit_type in (UnitType.LIGHT, UnitType.HEAVY, UnitType.RANGED):
            enemy = _find_nearest(unit, enemy_units)
            if enemy and unit.distance_to(enemy.x, enemy.y) <= 5:
                if unit.in_attack_range(enemy.x, enemy.y):
                    rel_x = enemy.x - unit.x + MAX_ATTACK_RANGE // 2
                    rel_y = enemy.y - unit.y + MAX_ATTACK_RANGE // 2
                    attack_idx = rel_y * MAX_ATTACK_RANGE + rel_x
                    cell_action = _encode_action(
                        ActionType.ATTACK, attack_pos=attack_idx)
                else:
                    cell_action = _move_toward(unit, enemy.x, enemy.y, gm)
            else:
                # Stay near base if no nearby threat
                bases = gm.get_units_of_type(player, UnitType.BASE)
                if bases and unit.distance_to(bases[0].x, bases[0].y) > 3:
                    cell_action = _move_toward(unit, bases[0].x, bases[0].y, gm)

        if cell_action:
            _set_cell_action(actions, unit.x, unit.y, w, cell_action)
    return actions


def _execute_build_barracks(state: GameState, player: int) -> np.ndarray:
    """BUILD_BARRACKS: build barracks, then produce light units and attack."""
    gm = state.game_map
    h, w = gm.height, gm.width
    actions = np.zeros(h * w * DIMS_PER_CELL, dtype=np.int32)
    resources = state.player_resources[player]
    workers = gm.get_units_of_type(player, UnitType.WORKER)
    barracks = gm.get_units_of_type(player, UnitType.BARRACKS)
    enemy_units = gm.get_player_units(1 - player)

    for unit in gm.get_player_units(player):
        if not unit.is_idle:
            continue
        cell_action = None

        if unit.unit_type == UnitType.BASE:
            if len(workers) < 2 and resources >= UNIT_STATS[UnitType.WORKER].cost:
                d = _find_empty_adjacent_dir(unit, gm)
                if d is not None:
                    cell_action = _encode_action(
                        ActionType.PRODUCE, produce_dir=d,
                        produce_type=UnitType.WORKER)

        elif unit.unit_type == UnitType.BARRACKS:
            # Once barracks exists, produce light units from it
            if resources >= UNIT_STATS[UnitType.LIGHT].cost:
                d = _find_empty_adjacent_dir(unit, gm)
                if d is not None:
                    cell_action = _encode_action(
                        ActionType.PRODUCE, produce_dir=d,
                        produce_type=UnitType.LIGHT)

        elif unit.unit_type == UnitType.WORKER:
            # First worker builds barracks if none exist
            if (workers and unit.unit_id == workers[0].unit_id
                    and not barracks
                    and resources >= UNIT_STATS[UnitType.BARRACKS].cost):
                d = _find_empty_adjacent_dir(unit, gm)
                if d is not None:
                    cell_action = _encode_action(
                        ActionType.PRODUCE, produce_dir=d,
                        produce_type=UnitType.BARRACKS)
            if cell_action is None:
                cell_action = _worker_harvest_cycle(unit, state, player, gm)

        # Combat units attack the nearest enemy
        elif unit.unit_type in (UnitType.LIGHT, UnitType.HEAVY, UnitType.RANGED):
            enemy = _find_nearest(unit, enemy_units)
            if enemy:
                if unit.in_attack_range(enemy.x, enemy.y):
                    rel_x = enemy.x - unit.x + MAX_ATTACK_RANGE // 2
                    rel_y = enemy.y - unit.y + MAX_ATTACK_RANGE // 2
                    attack_idx = rel_y * MAX_ATTACK_RANGE + rel_x
                    cell_action = _encode_action(
                        ActionType.ATTACK, attack_pos=attack_idx)
                else:
                    cell_action = _move_toward(unit, enemy.x, enemy.y, gm)

        if cell_action:
            _set_cell_action(actions, unit.x, unit.y, w, cell_action)
    return actions


def _execute_produce(state: GameState, player: int,
                     unit_type: UnitType) -> np.ndarray:
    """PRODUCE: harvest, build barracks, produce a specific unit type, and attack."""
    gm = state.game_map
    h, w = gm.height, gm.width
    actions = np.zeros(h * w * DIMS_PER_CELL, dtype=np.int32)
    resources = state.player_resources[player]
    workers = gm.get_units_of_type(player, UnitType.WORKER)
    barracks = gm.get_units_of_type(player, UnitType.BARRACKS)
    enemy_units = gm.get_player_units(1 - player)

    for unit in gm.get_player_units(player):
        if not unit.is_idle:
            continue
        cell_action = None

        if unit.unit_type == UnitType.BASE:
            if len(workers) < 2 and resources >= UNIT_STATS[UnitType.WORKER].cost:
                d = _find_empty_adjacent_dir(unit, gm)
                if d is not None:
                    cell_action = _encode_action(
                        ActionType.PRODUCE, produce_dir=d,
                        produce_type=UnitType.WORKER)

        elif unit.unit_type == UnitType.BARRACKS:
            if resources >= UNIT_STATS[unit_type].cost:
                d = _find_empty_adjacent_dir(unit, gm)
                if d is not None:
                    cell_action = _encode_action(
                        ActionType.PRODUCE, produce_dir=d,
                        produce_type=unit_type)

        elif unit.unit_type == UnitType.WORKER:
            # If no barracks yet, first worker builds one
            if (workers and unit.unit_id == workers[0].unit_id
                    and not barracks
                    and resources >= UNIT_STATS[UnitType.BARRACKS].cost):
                d = _find_empty_adjacent_dir(unit, gm)
                if d is not None:
                    cell_action = _encode_action(
                        ActionType.PRODUCE, produce_dir=d,
                        produce_type=UnitType.BARRACKS)
            if cell_action is None:
                cell_action = _worker_harvest_cycle(unit, state, player, gm)

        # Combat units attack nearest enemy
        elif unit.unit_type in (UnitType.LIGHT, UnitType.HEAVY, UnitType.RANGED):
            enemy = _find_nearest(unit, enemy_units)
            if enemy:
                if unit.in_attack_range(enemy.x, enemy.y):
                    rel_x = enemy.x - unit.x + MAX_ATTACK_RANGE // 2
                    rel_y = enemy.y - unit.y + MAX_ATTACK_RANGE // 2
                    attack_idx = rel_y * MAX_ATTACK_RANGE + rel_x
                    cell_action = _encode_action(
                        ActionType.ATTACK, attack_pos=attack_idx)
                else:
                    cell_action = _move_toward(unit, enemy.x, enemy.y, gm)

        if cell_action:
            _set_cell_action(actions, unit.x, unit.y, w, cell_action)
    return actions


def _execute_attack(state: GameState, player: int) -> np.ndarray:
    """ATTACK: send all combat units toward the enemy, keep harvesting."""
    gm = state.game_map
    h, w = gm.height, gm.width
    actions = np.zeros(h * w * DIMS_PER_CELL, dtype=np.int32)
    resources = state.player_resources[player]
    workers = gm.get_units_of_type(player, UnitType.WORKER)
    barracks = gm.get_units_of_type(player, UnitType.BARRACKS)
    enemy_units = gm.get_player_units(1 - player)

    for unit in gm.get_player_units(player):
        if not unit.is_idle:
            continue
        cell_action = None

        if unit.unit_type == UnitType.BASE:
            if len(workers) < 2 and resources >= UNIT_STATS[UnitType.WORKER].cost:
                d = _find_empty_adjacent_dir(unit, gm)
                if d is not None:
                    cell_action = _encode_action(
                        ActionType.PRODUCE, produce_dir=d,
                        produce_type=UnitType.WORKER)

        elif unit.unit_type == UnitType.BARRACKS:
            # Produce cheapest available combat unit
            for ut in [UnitType.LIGHT, UnitType.HEAVY, UnitType.RANGED]:
                if resources >= UNIT_STATS[ut].cost:
                    d = _find_empty_adjacent_dir(unit, gm)
                    if d is not None:
                        cell_action = _encode_action(
                            ActionType.PRODUCE, produce_dir=d,
                            produce_type=ut)
                        break

        elif unit.unit_type == UnitType.WORKER:
            if (workers and unit.unit_id == workers[0].unit_id
                    and not barracks
                    and resources >= UNIT_STATS[UnitType.BARRACKS].cost):
                d = _find_empty_adjacent_dir(unit, gm)
                if d is not None:
                    cell_action = _encode_action(
                        ActionType.PRODUCE, produce_dir=d,
                        produce_type=UnitType.BARRACKS)
            if cell_action is None:
                cell_action = _worker_harvest_cycle(unit, state, player, gm)

        elif unit.unit_type in (UnitType.LIGHT, UnitType.HEAVY, UnitType.RANGED):
            # Rush toward nearest enemy
            enemy = _find_nearest(unit, enemy_units)
            if enemy:
                if unit.in_attack_range(enemy.x, enemy.y):
                    rel_x = enemy.x - unit.x + MAX_ATTACK_RANGE // 2
                    rel_y = enemy.y - unit.y + MAX_ATTACK_RANGE // 2
                    attack_idx = rel_y * MAX_ATTACK_RANGE + rel_x
                    cell_action = _encode_action(
                        ActionType.ATTACK, attack_pos=attack_idx)
                else:
                    cell_action = _move_toward(unit, enemy.x, enemy.y, gm)

        if cell_action:
            _set_cell_action(actions, unit.x, unit.y, w, cell_action)
    return actions


# Strategy dispatch table
STRATEGY_EXECUTORS = {
    Strategy.HARVEST: _execute_harvest,
    Strategy.BUILD_BARRACKS: _execute_build_barracks,
    Strategy.PRODUCE_LIGHT: lambda s, p: _execute_produce(s, p, UnitType.LIGHT),
    Strategy.PRODUCE_HEAVY: lambda s, p: _execute_produce(s, p, UnitType.HEAVY),
    Strategy.PRODUCE_RANGED: lambda s, p: _execute_produce(s, p, UnitType.RANGED),
    Strategy.ATTACK: _execute_attack,
}


# ── Situation Key (module-level for reuse by parallel workers) ────────────

def _situation_key_from_state(state: GameState, player: int) -> str:
    """
    Quantize the game state into a discrete situation key.

    This is the APPROXIMABILITY pillar: similar game states map to the
    same key, so the agent can generalize across similar situations.

    Extracted as a module-level function so parallel workers can reuse
    it without needing a full PatternPolicy instance.
    """
    gm = state.game_map
    workers = len(gm.get_units_of_type(player, UnitType.WORKER))
    barracks = len(gm.get_units_of_type(player, UnitType.BARRACKS))
    combat = sum(1 for u in gm.get_player_units(player)
                 if u.unit_type in (UnitType.LIGHT, UnitType.HEAVY,
                                    UnitType.RANGED))
    resources = state.player_resources[player]

    # Quantize into bins
    w_bin = min(workers, 3)       # 0, 1, 2, 3+
    b_bin = min(barracks, 1)      # 0 or 1
    c_bin = min(combat // 2, 3)   # 0, 1-2, 3-4, 5+
    r_bin = min(resources // 3, 3)  # 0-2, 3-5, 6-8, 9+

    # Game phase: early (0-25%), mid (25-60%), late (60%+)
    phase_pct = state.tick / max(state.max_ticks, 1)
    phase = 'e' if phase_pct < 0.25 else ('m' if phase_pct < 0.6 else 'l')

    return f"w{w_bin}_b{b_bin}_c{c_bin}_r{r_bin}_{phase}"


# ── Pattern Policy ────────────────────────────────────────────────────────

class PatternPolicy:
    """
    The core decision-maker. Uses the four pillars of learning to select
    strategies based on pattern-matched game situations.

    No neural network. No weight matrices. No backpropagation.
    Just pattern recognition, comparison, composition, and feedback.
    """

    def __init__(self, map_height: int = 8, map_width: int = 8):
        self.map_height = map_height
        self.map_width = map_width

        # Core AI components (the four pillars)
        self.memory = DualMemory(max_patterns=5000, max_state=500)
        self.engine = PatternEngine(self.memory)
        self.feedback_loop = FeedbackLoop(learning_rate=0.1, discount_factor=0.99)
        self.explorer = ExplorationStrategy(exploration_rate=0.3)
        self._decay_rate = 0.995

        # State encoder
        self.encoder = GameStateEncoder(map_height, map_width)

        # Strategy bindings: situation_key -> {strategy -> StrategyBinding}
        # Each situation can have bindings for multiple strategies;
        # the one with highest confidence wins.
        # Situation keys are quantized game state descriptions like
        # "w2_b1_c0_r10_early" (2 workers, 1 barracks, 0 combat, 10 resources, early game)
        self.strategy_bindings: Dict[str, Dict[int, StrategyBinding]] = {}

        # Per-episode tracking
        self._episode_decisions: List[Tuple[str, int]] = []  # (situation_key, strategy)

        # Strategy persistence: stick with a strategy for several ticks
        self._current_strategy: Optional[int] = None
        self._current_pattern_id: Optional[str] = None
        self._strategy_ticks: int = 0
        self._strategy_hold_ticks: int = 15  # Re-evaluate every N ticks

    def _situation_key(self, state: GameState, player: int) -> str:
        """Quantize the game state into a discrete situation key.

        Delegates to the module-level function for reuse by parallel workers.
        """
        return _situation_key_from_state(state, player)

    def select_strategy(self, state: GameState,
                        player: int) -> Tuple[int, str]:
        """
        Use the core AI to select a strategy for the current game state.

        Returns:
            (strategy_index, situation_key)
        """
        # Strategy persistence: keep current strategy for several ticks
        # to give it time to take effect (e.g. producing a unit takes time)
        self._strategy_ticks += 1
        if (self._current_strategy is not None
                and self._strategy_ticks < self._strategy_hold_ticks
                and not self.explorer.should_explore()):
            # Stick with current strategy
            self._episode_decisions.append(
                (self._current_pattern_id, self._current_strategy))
            if self._current_pattern_id in self.strategy_bindings:
                bindings = self.strategy_bindings[self._current_pattern_id]
                if self._current_strategy in bindings:
                    bindings[self._current_strategy].times_used += 1
            return self._current_strategy, self._current_pattern_id

        # Time to re-evaluate strategy

        # 1. Compute situation key (quantized game state)
        sit_key = self._situation_key(state, player)

        # 2. Also feed features to PatternEngine for discovery/composition
        #    This is where subtraction (edge detection) and division (clustering)
        #    happen, building the pattern library (COMPOSABILITY).
        features = self.encoder.encode_for_pattern_engine(state, player)
        self.engine.process(features, domain="rts_situation")

        # 3. Look up best strategy for this situation key
        #    Deterministic tie-breaking: when confidences are equal,
        #    always pick the lowest strategy index. This ensures
        #    consistency between sequential and parallel modes.
        best_strategy = None
        best_confidence = -1.0

        if sit_key in self.strategy_bindings:
            for strat_idx in sorted(self.strategy_bindings[sit_key].keys()):
                binding = self.strategy_bindings[sit_key][strat_idx]
                if (binding.confidence > best_confidence
                        or (binding.confidence == best_confidence
                            and best_strategy is not None
                            and strat_idx < best_strategy)):
                    best_confidence = binding.confidence
                    best_strategy = strat_idx

        # 4. Exploration: sometimes try a random strategy
        #    This is the EXPLORATION pillar in action.
        if self.explorer.should_explore() or best_strategy is None:
            best_strategy = random.randint(0, NUM_STRATEGIES - 1)

        # 5. If this situation has no binding for the chosen strategy, create one
        if sit_key not in self.strategy_bindings:
            self.strategy_bindings[sit_key] = {}
        if best_strategy not in self.strategy_bindings[sit_key]:
            self.strategy_bindings[sit_key][best_strategy] = \
                StrategyBinding(best_strategy, sit_key)

        # Track this decision for end-of-episode feedback
        self._episode_decisions.append((sit_key, best_strategy))
        self.strategy_bindings[sit_key][best_strategy].times_used += 1

        # Set persistence state
        self._current_strategy = best_strategy
        self._current_pattern_id = sit_key
        self._strategy_ticks = 0

        return best_strategy, sit_key

    def execute_strategy(self, strategy: int, state: GameState,
                         player: int) -> np.ndarray:
        """
        Execute a strategy as concrete per-unit actions.
        Returns flat action array (H*W*DIMS_PER_CELL,).
        """
        executor = STRATEGY_EXECUTORS.get(strategy, _execute_harvest)
        return executor(state, player)

    def get_action(self, state: GameState, player: int) -> np.ndarray:
        """
        Full decision pipeline: perceive → match → select → execute.
        This is the main entry point called each game tick.
        """
        strategy, pattern_id = self.select_strategy(state, player)
        return self.execute_strategy(strategy, state, player)

    def begin_episode(self):
        """Call at the start of each training game."""
        self._episode_decisions = []
        self._current_strategy = None
        self._current_pattern_id = None
        self._strategy_ticks = 0

    def record_outcome(self, won: bool, game_info: Dict = None):
        """
        After a game ends, apply feedback to all patterns and bindings
        used during this episode. This is the FEEDBACK pillar in action.
        """
        game_info = game_info or {}

        # --- DOMINANT STRATEGY CREDIT ASSIGNMENT ---
        # For each situation encountered in this episode, only the strategy
        # that was used the MOST gets full credit. This focuses the learning
        # signal: instead of spreading fractional credit across 30+ bindings,
        # each situation's dominant strategy gets +1 win or +1 loss.
        #
        # This makes strategy selection converge faster: winning strategies
        # quickly build confidence, while barely-used strategies in the same
        # situation don't get noise added to their statistics.

        # Count usage per situation per strategy
        sit_strat_counts: Dict[str, Dict[int, int]] = {}
        for sit_key, strat in self._episode_decisions:
            if sit_key not in sit_strat_counts:
                sit_strat_counts[sit_key] = {}
            sit_strat_counts[sit_key][strat] = \
                sit_strat_counts[sit_key].get(strat, 0) + 1

        # For each situation, find the dominant strategy and give it credit
        feedback_value = 0.8 if won else 0.2

        for sit_key, strat_counts in sit_strat_counts.items():
            # Find the strategy used most in this situation
            dominant_strat = max(strat_counts, key=strat_counts.get)

            # Ensure binding exists (may be new from parallel workers)
            if sit_key not in self.strategy_bindings:
                self.strategy_bindings[sit_key] = {}
            if dominant_strat not in self.strategy_bindings[sit_key]:
                self.strategy_bindings[sit_key][dominant_strat] = \
                    StrategyBinding(dominant_strat, sit_key)

            # Full credit to the dominant strategy
            binding = self.strategy_bindings[sit_key][dominant_strat]
            if won:
                binding.wins += 1
            else:
                binding.losses += 1

        # Apply feedback to patterns in the pattern engine (APPROXIMABILITY)
        all_pattern_ids = list(self.memory.patterns.patterns.keys())
        # Refine the most recently used patterns
        for pid in all_pattern_ids[-20:]:  # Last 20 patterns
            pattern = self.memory.patterns.patterns.get(pid)
            if pattern:
                pattern.refine(feedback=feedback_value)

        # Apply feedback through FeedbackLoop to pattern memory
        if all_pattern_ids:
            self.feedback_loop.apply_feedback(
                FeedbackSignal(
                    signal_type=FeedbackType.EXTRINSIC,
                    value=1.0 if won else -0.5,
                    target_pattern_ids=all_pattern_ids[-10:],
                    context={'won': won}
                ),
                self.memory.patterns.patterns
            )

        # Process episode outcome as a composite pattern (COMPOSABILITY)
        outcome_features = [
            float(won),
            game_info.get('p0_resources', 0) / 20.0,
            game_info.get('p1_resources', 0) / 20.0,
            game_info.get('p0_units', 0) / 10.0,
            game_info.get('p1_units', 0) / 10.0,
            game_info.get('tick', 0) / 2000.0,
        ]
        self.engine.process(outcome_features, domain="rts_outcome")

        # Try composing novel patterns (COMPOSABILITY + EXPLORATION)
        all_patterns = list(self.memory.patterns.patterns.values())
        if len(all_patterns) >= 2:
            combo = self.explorer.suggest_combination(all_patterns)
            if combo and len(combo) >= 2:
                composite = Pattern.create_composite(combo, "rts_situation")
                self.memory.patterns.store(composite)

        # Decay exploration rate over time
        self.explorer.decay_exploration(self._decay_rate)

        # Clear episode tracking
        self._episode_decisions = []

    def save(self, path: str):
        """Save pattern policy state to disk."""
        os.makedirs(path, exist_ok=True)

        # Save patterns
        patterns_data = {
            pid: p.to_dict()
            for pid, p in self.memory.patterns.patterns.items()
        }
        with open(os.path.join(path, 'patterns.json'), 'w') as f:
            json.dump(patterns_data, f, indent=2)

        # Save strategy bindings
        bindings_data = {}
        for pid, strat_map in self.strategy_bindings.items():
            bindings_data[pid] = {
                str(strat_idx): binding.to_dict()
                for strat_idx, binding in strat_map.items()
            }
        with open(os.path.join(path, 'strategy_bindings.json'), 'w') as f:
            json.dump(bindings_data, f, indent=2)

        # Save exploration state
        with open(os.path.join(path, 'exploration.json'), 'w') as f:
            json.dump({
                'exploration_rate': self.explorer.exploration_rate,
            }, f, indent=2)

    def load(self, path: str):
        """Load pattern policy state from disk."""
        # Load patterns
        patterns_path = os.path.join(path, 'patterns.json')
        if os.path.exists(patterns_path):
            with open(patterns_path, 'r') as f:
                patterns_data = json.load(f)
            for pid, pdata in patterns_data.items():
                pattern = Pattern.from_dict(pdata)
                self.memory.patterns.store(pattern)

        # Load strategy bindings
        bindings_path = os.path.join(path, 'strategy_bindings.json')
        if os.path.exists(bindings_path):
            with open(bindings_path, 'r') as f:
                bindings_data = json.load(f)
            for pid, strat_map in bindings_data.items():
                self.strategy_bindings[pid] = {}
                for strat_idx_str, bdata in strat_map.items():
                    strat_idx = int(strat_idx_str)
                    self.strategy_bindings[pid][strat_idx] = \
                        StrategyBinding.from_dict(bdata)

        # Load exploration state
        explore_path = os.path.join(path, 'exploration.json')
        if os.path.exists(explore_path):
            with open(explore_path, 'r') as f:
                explore_data = json.load(f)
            self.explorer.exploration_rate = explore_data.get(
                'exploration_rate', 0.3)

    def get_stats(self) -> Dict:
        """Get current learning statistics."""
        total_bindings = sum(
            len(smap) for smap in self.strategy_bindings.values()
        )
        # Count strategy usage
        strategy_counts = {s.name: 0 for s in Strategy}
        for pid, smap in self.strategy_bindings.items():
            for strat_idx, binding in smap.items():
                name = Strategy(strat_idx).name
                strategy_counts[name] += binding.times_used

        return {
            'patterns_discovered': len(self.memory.patterns.patterns),
            'strategy_bindings': total_bindings,
            'exploration_rate': self.explorer.exploration_rate,
            'strategy_usage': strategy_counts,
        }

    # ── Parallel training helpers ────────────────────────────────────────

    def snapshot_for_workers(self) -> dict:
        """Return minimal policy state for parallel workers.

        Workers only need strategy bindings (for decision-making) and the
        exploration rate. They don't need the full pattern memory, engine,
        or feedback loop — those stay in the main process.
        """
        bindings_snapshot = {}
        for sit_key, strat_map in self.strategy_bindings.items():
            bindings_snapshot[sit_key] = {
                str(strat_idx): binding.to_dict()
                for strat_idx, binding in strat_map.items()
            }
        return {
            'strategy_bindings': bindings_snapshot,
            'exploration_rate': self.explorer.exploration_rate,
        }

    def apply_episode_result(self, decisions: list, won: bool,
                             game_info: dict,
                             discovered_patterns: list = None):
        """Apply a single episode's outcome to the shared policy state.

        Used by parallel training to merge worker results back into the
        main process. Merges worker-discovered patterns into the shared
        memory, counts times_used from worker decisions, then calls
        record_outcome to update wins/losses, refine patterns, and
        compose new ones.

        Args:
            decisions: List of (sit_key, strategy) tuples from the worker
            won: Whether the episode was won
            game_info: Dict with game outcome details
            discovered_patterns: List of pattern dicts from the worker's
                                 PatternEngine (Composability pillar)
        """
        # Merge worker-discovered patterns into the main memory
        if discovered_patterns:
            for pdata in discovered_patterns:
                pattern = Pattern.from_dict(pdata)
                existing = self.memory.patterns.patterns.get(pattern.id)
                if existing:
                    # Pattern exists: keep the one with higher activation count
                    if pattern.activation_count > existing.activation_count:
                        self.memory.patterns.store(pattern)
                else:
                    self.memory.patterns.store(pattern)

        # Count usage per (sit_key, strategy) from worker decisions
        usage_counts = {}
        for sit_key, strat in decisions:
            key = (sit_key, strat)
            usage_counts[key] = usage_counts.get(key, 0) + 1

        # Ensure all bindings exist and merge times_used
        for (sit_key, strategy), count in usage_counts.items():
            if sit_key not in self.strategy_bindings:
                self.strategy_bindings[sit_key] = {}
            if strategy not in self.strategy_bindings[sit_key]:
                self.strategy_bindings[sit_key][strategy] = \
                    StrategyBinding(strategy, sit_key)
            self.strategy_bindings[sit_key][strategy].times_used += count

        # Now apply feedback (wins/losses, pattern refinement)
        self._episode_decisions = decisions
        self.record_outcome(won, game_info)
