"""
Parallel Worker for batch-parallel Zork training.

Each worker gets its own PatternEngine + DualMemory so the Composability
pillar is active during gameplay — just like sequential training. Workers
discover patterns every turn via engine.process(), then return those patterns
alongside decisions for the main process to merge.

This ensures parallel training has the same learning quality as sequential:
all four pillars (Feedback, Approximability, Composability, Exploration)
are fully active in every episode.

Mirrors rts_ai/parallel_worker.py but adapted for text adventure play.
Each worker spawns its own dfrotz subprocess.
"""

import random
from typing import Dict, List, Tuple, Optional, Set

from core.pattern import Pattern
from core.engine import PatternEngine
from core.memory import DualMemory

from zork_ai.game_interface import FrotzInterface, ZorkState
from zork_ai.text_encoder import ZorkStateEncoder
from zork_ai.text_parser import (
    parse_exits,
    detect_death,
    identify_treasures,
    categorize_room,
    DIRECTIONS,
    DIR_ABBREVS,
)
from zork_ai.zork_policy import (
    ZorkStrategy, NUM_STRATEGIES, STRATEGY_EXECUTORS, StrategyBinding,
    RoomMemory, _situation_key_from_state, _all_possible_exits,
    _get_exploration_unlock, _get_early_game_navigation,
)


class WorkerPolicy:
    """
    Worker policy for parallel episode execution.

    Has its own PatternEngine + DualMemory so pattern discovery and
    composition happen during gameplay (the Composability pillar). Uses
    a frozen snapshot of strategy_bindings and exploration_rate for
    decision-making.

    After the episode, discovered patterns are returned to the main
    process for merging into the shared memory.
    """

    def __init__(self, snapshot: dict):
        # Own pattern engine for in-game pattern discovery (Composability)
        self.memory = DualMemory(max_patterns=5000, max_state=500)
        self.engine = PatternEngine(self.memory)
        self.encoder = ZorkStateEncoder()

        # Rebuild strategy bindings from snapshot
        self.strategy_bindings: Dict[str, Dict[int, StrategyBinding]] = {}
        for sit_key, strat_map in snapshot['strategy_bindings'].items():
            self.strategy_bindings[sit_key] = {}
            for strat_idx_str, bdata in strat_map.items():
                strat_idx = int(strat_idx_str)
                self.strategy_bindings[sit_key][strat_idx] = \
                    StrategyBinding.from_dict(bdata)

        self.exploration_rate = snapshot['exploration_rate']

        # Per-episode tracking
        self._episode_decisions: List[Tuple[str, int]] = []
        self._score_by_strategy: Dict[int, float] = {}
        self.room_memory = RoomMemory()

    def begin_episode(self):
        """Reset per-episode state."""
        self._episode_decisions = []
        self._score_by_strategy = {}
        self.room_memory = RoomMemory()
        self.encoder.reset()

    def get_decisions(self) -> List[Tuple[str, int]]:
        """Return the decisions made this episode."""
        return self._episode_decisions

    def get_discovered_patterns(self) -> List[dict]:
        """Return patterns discovered during gameplay."""
        return [p.to_dict() for p in self.memory.patterns.patterns.values()]

    def get_score_by_strategy(self) -> Dict[int, float]:
        """Return score deltas attributed to each strategy."""
        return self._score_by_strategy

    def _get_viable_strategies(self, state: ZorkState,
                               available_exits: List[str]) -> Set[int]:
        """
        Return strategies that can do something useful right now.
        Mirrors ZorkPatternPolicy._get_viable_strategies().
        """
        viable = {ZorkStrategy.EXPLORE_NEW, ZorkStrategy.EXPLORE_KNOWN}

        room_lower = state.location.lower() if state.location else ''

        # COLLECT_ITEMS: only if items visible or take_all not yet tried
        items_tried = self.room_memory.room_items_tried.get(room_lower, set())
        if state.visible_items or 'take_all' not in items_tried:
            viable.add(ZorkStrategy.COLLECT_ITEMS)

        # USE_ITEM: only if we have inventory items
        if state.inventory:
            viable.add(ZorkStrategy.USE_ITEM)

        # DEPOSIT_TROPHY: only if we have treasures
        if state.inventory:
            if identify_treasures(state.inventory):
                viable.add(ZorkStrategy.DEPOSIT_TROPHY)

        # FIGHT: only if enemies present
        if state.enemies:
            viable.add(ZorkStrategy.FIGHT)

        # MANAGE_LIGHT: only if dark or we see a lamp
        if state.is_dark:
            viable.add(ZorkStrategy.MANAGE_LIGHT)
        else:
            for item in state.visible_items:
                if 'lamp' in item.lower() or 'lantern' in item.lower():
                    viable.add(ZorkStrategy.MANAGE_LIGHT)

        # INTERACT: if untried objects exist
        objects_tried = self.room_memory.room_objects_tried.get(room_lower, set())
        desc_lower = state.description.lower()
        interactable = [
            'mailbox', 'door', 'window', 'trapdoor', 'trap door',
            'grating', 'gate', 'rug', 'carpet', 'leaves',
            'table', 'desk', 'pedestal', 'altar', 'basket',
            'case', 'trophy case', 'chest', 'nest',
            'dam', 'button', 'switch', 'lever', 'mirror',
            'machine', 'lid', 'bolt', 'boat', 'tree',
        ]
        has_untried = any(
            obj in desc_lower and obj not in objects_tried
            for obj in interactable
        )
        has_untried_vis = any(
            item not in objects_tried for item in state.visible_items
        )
        if has_untried or has_untried_vis or state.inventory:
            viable.add(ZorkStrategy.INTERACT)

        return viable

    def select_strategy(self, state: ZorkState,
                        available_exits: List[str] = None
                        ) -> Tuple[int, str]:
        """
        Select a strategy using the frozen policy snapshot.
        Uses context-aware filtering to only consider viable strategies.
        """
        sit_key = _situation_key_from_state(state)

        # Feed features to PatternEngine for discovery/composition
        tried_here = len(self.room_memory.room_exits_tried.get(
            state.location.lower(), set())) if state.location else 0
        features = self.encoder.encode(state, available_exits or [],
                                       tried_here)
        self.engine.process(features, domain="zork_situation")

        # Context-aware filtering
        viable = self._get_viable_strategies(state, available_exits or [])

        # Look up best viable strategy (with deterministic tie-breaking)
        best_strategy = None
        best_confidence = -1.0
        if sit_key in self.strategy_bindings:
            for strat_idx in sorted(self.strategy_bindings[sit_key].keys()):
                if strat_idx not in viable:
                    continue
                binding = self.strategy_bindings[sit_key][strat_idx]
                if (binding.confidence > best_confidence
                        or (binding.confidence == best_confidence
                            and best_strategy is not None
                            and strat_idx < best_strategy)):
                    best_confidence = binding.confidence
                    best_strategy = strat_idx

        # Exploration — independent random check
        if (random.random() < self.exploration_rate) or best_strategy is None:
            best_strategy = random.choice(list(viable))

        # Contextual overrides (safety)
        if state.is_dark and best_strategy != ZorkStrategy.MANAGE_LIGHT:
            inv_lower = [i.lower() for i in state.inventory]
            if any('lamp' in i or 'lantern' in i for i in inv_lower):
                if random.random() < 0.7:
                    best_strategy = ZorkStrategy.MANAGE_LIGHT

        if state.enemies and best_strategy not in (
                ZorkStrategy.FIGHT, ZorkStrategy.EXPLORE_NEW):
            if random.random() < 0.6:
                best_strategy = ZorkStrategy.FIGHT

        # Track decision
        self._episode_decisions.append((sit_key, best_strategy))

        return best_strategy, sit_key

    def get_command(self, state: ZorkState,
                    available_exits: List[str] = None) -> str:
        """Full decision pipeline with priority system.

        Priority order (mirrors ZorkPatternPolicy.get_command):
        1. Early-game navigation (deterministic path to house entry)
        2. Exploration unlocks (key chokepoint actions)
        3. Strategy-based decisions (learned from training)
        """
        # ── Priority 1: Early-game navigation ──
        early_nav = _get_early_game_navigation(state, self.room_memory)
        if early_nav:
            sit_key = _situation_key_from_state(state)
            self._episode_decisions.append((sit_key, ZorkStrategy.EXPLORE_NEW))
            self._track_direction(early_nav, state)
            return early_nav

        # ── Priority 2: Exploration unlocks ──
        unlock = _get_exploration_unlock(state, self.room_memory)
        if unlock:
            sit_key = _situation_key_from_state(state)
            self._episode_decisions.append((sit_key, ZorkStrategy.INTERACT))
            self._track_direction(unlock, state)
            return unlock

        # ── Priority 3: Strategy-based decisions ──
        strategy, sit_key = self.select_strategy(state, available_exits)

        executor = STRATEGY_EXECUTORS.get(
            strategy, STRATEGY_EXECUTORS[ZorkStrategy.INTERACT])
        command = executor(state, self.room_memory,
                           available_exits or [])

        self._track_direction(command, state)
        return command

    def _track_direction(self, command: str, state: ZorkState):
        """Track directional commands for room memory."""
        cmd_lower = command.lower().strip()
        if cmd_lower in DIRECTIONS or cmd_lower in DIR_ABBREVS:
            direction = DIR_ABBREVS.get(cmd_lower, cmd_lower)
            if state.location:
                self.room_memory.try_exit(state.location, direction)

    def record_score_delta(self, strategy: int, delta: int):
        """Track score changes attributed to each strategy."""
        if delta > 0:
            self._score_by_strategy[strategy] = \
                self._score_by_strategy.get(strategy, 0) + delta


def play_episode_worker(args) -> Dict:
    """
    Run one episode in a worker process.

    Each worker has its own PatternEngine and dfrotz process.
    Returns the outcome, decisions, AND discovered patterns for the
    main process to merge.

    Args:
        args: Tuple of (policy_snapshot, game_file, frotz_path, max_moves)

    Returns:
        Dict with score, decisions, discovered_patterns, score_by_strategy,
        game_info, rooms_visited, etc.
    """
    if len(args) == 4:
        policy_snapshot, game_file, frotz_path, max_moves = args
    else:
        policy_snapshot, game_file, frotz_path = args[:3]
        max_moves = 400

    worker = WorkerPolicy(policy_snapshot)
    fi = FrotzInterface(game_file, frotz_path)

    try:
        fi.start()
        worker.begin_episode()

        moves = 0
        moves_since_score = 0
        max_stale_moves = 100
        last_inv_check = 0
        INV_CHECK_INTERVAL = 10

        while moves < max_moves:
            state = fi.state

            if state.is_dead or state.is_won:
                break
            if not fi.is_running:
                break

            # Periodically refresh inventory
            if moves - last_inv_check >= INV_CHECK_INTERVAL:
                try:
                    fi.get_inventory()
                except Exception:
                    pass
                last_inv_check = moves

            available_exits = parse_exits(state.description)
            command = worker.get_command(state, available_exits)

            response = fi.send_command(command)

            # Track room visits
            if state.location:
                worker.room_memory.visit_room(state.location)

            # After a successful "take", refresh inventory
            if (command.lower().startswith('take')
                    and 'taken' in response.lower()):
                try:
                    fi.get_inventory()
                    last_inv_check = moves
                except Exception:
                    pass

            # Track score deltas
            score_delta = fi.score_delta
            if score_delta > 0:
                if worker._episode_decisions:
                    last_strat = worker._episode_decisions[-1][1]
                    worker.record_score_delta(last_strat, score_delta)
                moves_since_score = 0
            else:
                moves_since_score += 1

            # Staleness check
            if moves_since_score >= max_stale_moves:
                break

            moves += 1

            if detect_death(fi.state.last_response):
                break

        final_score = fi.state.score
        rooms_visited = len(fi.state.visited_rooms)

        return {
            'score': final_score,
            'rooms_visited': rooms_visited,
            'decisions': worker.get_decisions(),
            'discovered_patterns': worker.get_discovered_patterns(),
            'score_by_strategy': worker.get_score_by_strategy(),
            'game_info': {
                'rooms_visited': rooms_visited,
                'items_collected': len(fi.state.items_collected),
                'moves': moves,
                'is_dead': fi.state.is_dead,
                'is_won': fi.state.is_won,
            },
        }

    finally:
        fi.close()
