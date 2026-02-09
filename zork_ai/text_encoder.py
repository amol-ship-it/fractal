"""
Text Encoder — Converts ZorkState into feature vectors for PatternEngine.

Same role as rts_ai/encoder.py but for text adventure state. Produces a
flat List[float] with ~25 features, all normalized to [0, 1].

Features capture:
- Score progress (score/350)
- Exploration progress (rooms_visited, new_room_this_turn)
- Inventory features (item_count, has_lamp, has_sword)
- Room type (underground, outdoor, building, other)
- Danger indicators (is_dark, enemy_present, danger_keywords)
- Movement features (available_exits, untried_exits_ratio)
- Action history (moves_since_score_change, stuck_count)
- Game phase (early/mid/late)
"""

from typing import List, Dict

from zork_ai.game_interface import ZorkState
from zork_ai.text_parser import (
    categorize_room,
    identify_treasures,
    DIRECTIONS,
)


class ZorkStateEncoder:
    """
    Encodes ZorkState into a flat feature vector for the PatternEngine.

    The features are designed for the Composability pillar: the PatternEngine
    uses edge detection (subtraction) and clustering (division) to discover
    patterns in these features across many game states.
    """

    # Total score in Zork I
    MAX_SCORE = 350

    # Number of known rooms (~110 in Zork I)
    MAX_ROOMS = 110

    def __init__(self):
        # Track per-episode history for delta features
        self._moves_since_score = 0
        self._stuck_count = 0
        self._last_location = ''
        self._prev_score = 0
        self._tried_exits: Dict[str, set] = {}  # room -> set of tried dirs

    def reset(self):
        """Reset per-episode tracking."""
        self._moves_since_score = 0
        self._stuck_count = 0
        self._last_location = ''
        self._prev_score = 0
        self._tried_exits = {}

    def encode(self, state: ZorkState, available_exits: List[str] = None,
               tried_exits_here: int = 0) -> List[float]:
        """
        Encode a ZorkState as a flat feature vector.

        Args:
            state: The current game state
            available_exits: Exits available from current room
            tried_exits_here: Number of exits already tried from this room

        Returns:
            List of ~25 floats, all in [0, 1]
        """
        features = []

        # ── 1. Score features (3) ────────────────────────────────────
        # Score progress
        features.append(min(state.score / self.MAX_SCORE, 1.0))

        # Score changed this turn
        score_delta = state.score - self._prev_score
        features.append(1.0 if score_delta > 0 else 0.0)

        # Update score tracking
        if score_delta > 0:
            self._moves_since_score = 0
        else:
            self._moves_since_score += 1
        self._prev_score = state.score

        # Moves since last score change (staleness)
        features.append(min(self._moves_since_score / 50.0, 1.0))

        # ── 2. Exploration features (3) ──────────────────────────────
        # Rooms visited (fraction of total)
        features.append(min(len(state.visited_rooms) / self.MAX_ROOMS, 1.0))

        # New room this turn
        is_new_room = (state.location.lower() not in
                       {r.lower() for r in state.visited_rooms}
                       if state.location else False)
        features.append(1.0 if is_new_room else 0.0)

        # Location changed this turn
        location_changed = (state.location != self._last_location
                            and state.location != '')
        features.append(1.0 if location_changed else 0.0)
        if not location_changed and state.location:
            self._stuck_count += 1
        else:
            self._stuck_count = 0
        self._last_location = state.location

        # ── 3. Inventory features (4) ────────────────────────────────
        # Item count
        features.append(min(len(state.inventory) / 10.0, 1.0))

        # Has lamp/lantern (critical for underground areas)
        has_lamp = any('lamp' in i.lower() or 'lantern' in i.lower()
                       for i in state.inventory)
        features.append(1.0 if has_lamp else 0.0)

        # Has sword (critical for combat)
        has_sword = any('sword' in i.lower() for i in state.inventory)
        features.append(1.0 if has_sword else 0.0)

        # Has treasures (items for trophy case)
        treasures = identify_treasures(state.inventory)
        features.append(min(len(treasures) / 5.0, 1.0))

        # ── 4. Room type features (5) — one-hot encoding ────────────
        category = categorize_room(state.location)
        room_types = ['house', 'forest', 'cave_entrance', 'underground', 'other']
        for rt in room_types:
            if category == rt:
                features.append(1.0)
            elif category in ('river', 'maze', 'treasure_room'):
                # Map secondary categories
                if rt == 'underground':
                    features.append(0.7)  # close to underground
                else:
                    features.append(0.0)
            else:
                features.append(0.0)

        # ── 5. Danger features (3) ───────────────────────────────────
        # Is dark
        features.append(1.0 if state.is_dark else 0.0)

        # Enemy present
        features.append(1.0 if state.enemies else 0.0)

        # Number of enemies (normalized)
        features.append(min(len(state.enemies) / 2.0, 1.0))

        # ── 6. Movement features (2) ────────────────────────────────
        # Available exits count
        num_exits = len(available_exits) if available_exits else 0
        features.append(min(num_exits / 6.0, 1.0))

        # Untried exits ratio
        if num_exits > 0 and tried_exits_here < num_exits:
            features.append(
                (num_exits - tried_exits_here) / num_exits
            )
        else:
            features.append(0.0)

        # ── 7. Stuckness features (2) ───────────────────────────────
        # Stuck count (same room, no progress)
        features.append(min(self._stuck_count / 10.0, 1.0))

        # Visible items (things to interact with)
        features.append(min(len(state.visible_items) / 5.0, 1.0))

        # ── 8. Game phase (1) ───────────────────────────────────────
        # Based on score progress
        if state.score < 50:
            phase = 0.0   # early
        elif state.score < 150:
            phase = 0.33  # mid-early
        elif state.score < 250:
            phase = 0.66  # mid-late
        else:
            phase = 1.0   # late
        features.append(phase)

        return features

    @staticmethod
    def feature_names() -> List[str]:
        """Return human-readable names for each feature dimension."""
        return [
            'score_progress',
            'score_changed',
            'moves_since_score',
            'rooms_visited_frac',
            'new_room',
            'location_changed',
            'inventory_count',
            'has_lamp',
            'has_sword',
            'has_treasures',
            'room_house',
            'room_forest',
            'room_cave_entrance',
            'room_underground',
            'room_other',
            'is_dark',
            'enemy_present',
            'num_enemies',
            'available_exits',
            'untried_exits_ratio',
            'stuck_count',
            'visible_items',
            'game_phase',
        ]
