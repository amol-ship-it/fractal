"""
Parallel Worker for batch-parallel training.

Each worker gets its own PatternEngine + DualMemory so the Composability
pillar is active during gameplay — just like sequential training. Workers
discover patterns every tick via engine.process(), then return those patterns
alongside decisions for the main process to merge.

This ensures parallel training has the same learning quality as sequential:
all four pillars (Feedback, Approximability, Composability, Exploration)
are fully active in every episode.
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional

from core.pattern import Pattern
from core.engine import PatternEngine
from core.memory import DualMemory

from game.engine import GameEngine
from game.game_state import GameState
from game.units import UnitType
from rts_ai.encoder import GameStateEncoder
from rts_ai.pattern_policy import (
    Strategy, NUM_STRATEGIES, STRATEGY_EXECUTORS, StrategyBinding,
    _situation_key_from_state,
)
from game.ai_opponents import RushAI, EconomyAI, DefensiveAI, RandomAI


AI_OPPONENTS = {
    'rush': RushAI,
    'economy': EconomyAI,
    'defensive': DefensiveAI,
    'random': RandomAI,
}


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

    def __init__(self, snapshot: dict, map_size: int):
        self.map_height = map_size
        self.map_width = map_size

        # Own pattern engine for in-game pattern discovery (Composability)
        self.memory = DualMemory(max_patterns=5000, max_state=500)
        self.engine = PatternEngine(self.memory)
        self.encoder = GameStateEncoder(map_size, map_size)

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

        # Strategy persistence
        self._current_strategy: Optional[int] = None
        self._current_pattern_id: Optional[str] = None
        self._strategy_ticks: int = 0
        self._strategy_hold_ticks: int = 15

    def begin_episode(self):
        """Reset per-episode state."""
        self._episode_decisions = []
        self._current_strategy = None
        self._current_pattern_id = None
        self._strategy_ticks = 0

    def get_decisions(self) -> List[Tuple[str, int]]:
        """Return the decisions made this episode for feedback in main process."""
        return self._episode_decisions

    def get_discovered_patterns(self) -> List[dict]:
        """Return patterns discovered during gameplay for merging into main memory."""
        return [p.to_dict() for p in self.memory.patterns.patterns.values()]

    def select_strategy(self, state: GameState,
                        player: int) -> Tuple[int, str]:
        """Select a strategy using the frozen policy snapshot.

        IMPORTANT: This must match PatternPolicy.select_strategy exactly in
        its exploration behavior. The sequential policy calls should_explore()
        twice — once for the persistence check and once for the strategy
        selection — using TWO independent random draws. We must do the same,
        otherwise the worker explores ~20× more often than sequential mode.
        """
        # Strategy persistence — uses its own exploration check
        self._strategy_ticks += 1
        if (self._current_strategy is not None
                and self._strategy_ticks < self._strategy_hold_ticks
                and not (random.random() < self.exploration_rate)):
            self._episode_decisions.append(
                (self._current_pattern_id, self._current_strategy))
            return self._current_strategy, self._current_pattern_id

        # Compute situation key
        sit_key = _situation_key_from_state(state, player)

        # Feed features to PatternEngine for discovery/composition
        # (COMPOSABILITY pillar — same as sequential training)
        features = self.encoder.encode_for_pattern_engine(state, player)
        self.engine.process(features, domain="rts_situation")

        # Look up best strategy (with deterministic tie-breaking)
        # When multiple strategies have the same confidence (e.g. all 0.5
        # when uninformed), always pick the same one so that all concurrent
        # workers with the same snapshot converge on the same choice rather
        # than each picking randomly via dict iteration order.
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

        # Exploration — SECOND independent random check (matching sequential)
        if (random.random() < self.exploration_rate) or best_strategy is None:
            best_strategy = random.randint(0, NUM_STRATEGIES - 1)

        # Track decision
        self._episode_decisions.append((sit_key, best_strategy))

        # Set persistence
        self._current_strategy = best_strategy
        self._current_pattern_id = sit_key
        self._strategy_ticks = 0

        return best_strategy, sit_key

    def get_action(self, state: GameState, player: int) -> np.ndarray:
        """Full decision pipeline: select strategy → execute."""
        strategy, _ = self.select_strategy(state, player)
        executor = STRATEGY_EXECUTORS.get(strategy, STRATEGY_EXECUTORS[Strategy.HARVEST])
        return executor(state, player)


def play_episode_worker(args) -> Dict:
    """
    Run one episode in a worker process.

    Each worker has its own PatternEngine so pattern discovery happens
    during gameplay. Returns the outcome, decisions, AND discovered
    patterns for the main process to merge.

    Args:
        args: Tuple of (policy_snapshot, opponent_name, max_ticks, map_size)
              or (policy_snapshot, opponent_name, max_ticks, map_size, explore_boost)

    Returns:
        Dict with won, winner, tick, decisions, discovered_patterns, and game_info.
    """
    if len(args) == 5:
        policy_snapshot, opponent_name, max_ticks, map_size, explore_boost = args
    else:
        policy_snapshot, opponent_name, max_ticks, map_size = args
        explore_boost = 0.0

    worker = WorkerPolicy(policy_snapshot, map_size)
    # Apply per-worker exploration diversity: different workers in the same
    # batch try different strategies, preventing them all from exploiting
    # the same stale "best" strategy.
    worker.exploration_rate = min(1.0, worker.exploration_rate + explore_boost)
    engine = GameEngine(map_size=map_size, max_ticks=max_ticks)
    state = engine.reset()
    opponent = AI_OPPONENTS[opponent_name]()

    worker.begin_episode()

    while not state.done:
        p0_action = worker.get_action(state, player=0)
        p1_action = opponent.get_action(state, player=1)
        state, info = engine.step(p0_action, p1_action)

    return {
        'won': state.winner == 0,
        'winner': state.winner,
        'tick': state.tick,
        'decisions': worker.get_decisions(),
        'discovered_patterns': worker.get_discovered_patterns(),
        'game_info': {
            'p0_units': len(state.game_map.get_player_units(0)),
            'p1_units': len(state.game_map.get_player_units(1)),
            'p0_resources': state.player_resources[0],
            'p1_resources': state.player_resources[1],
            'tick': state.tick,
        },
    }
