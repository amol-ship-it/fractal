"""
Knowledge Store - Persistent storage of learned strategic knowledge.

Stores high-level strategic concepts that transcend specific games:
- Resource management strategies
- Unit composition patterns
- Timing attack patterns
- Economic build orders
- Defensive formations
- Aggression levels per game phase

These concepts are stored in a game-agnostic format that can be
transferred to new games (e.g., from MicroRTS to Age of Empires).
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

from core.pattern import Pattern, PatternType
from core.memory import DualMemory


class StrategyCategory(Enum):
    ECONOMY = "economy"
    MILITARY = "military"
    TIMING = "timing"
    COMPOSITION = "composition"
    POSITIONING = "positioning"
    ADAPTATION = "adaptation"


@dataclass
class StrategicKnowledge:
    """A single piece of strategic knowledge."""
    knowledge_id: str
    category: str
    description: str
    conditions: Dict[str, float]     # When to apply this knowledge
    actions: Dict[str, float]        # What to do (abstract actions)
    effectiveness: float             # How well it worked (0-1)
    confidence: float                # How confident we are (0-1)
    game_source: str                 # Which game this was learned from
    times_applied: int = 0
    times_successful: int = 0
    pattern_ids: List[str] = field(default_factory=list)  # Linked patterns
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def success_rate(self) -> float:
        if self.times_applied == 0:
            return 0.5
        return self.times_successful / self.times_applied

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategicKnowledge':
        return cls(**data)


class KnowledgeStore:
    """
    Persistent storage and retrieval of strategic knowledge.

    Knowledge is organized by category and game phase, making it
    easy to query relevant strategies for any RTS-like game.
    """

    def __init__(self, store_path: str = "knowledge_store"):
        self.store_path = store_path
        self.knowledge: Dict[str, StrategicKnowledge] = {}
        self._category_index: Dict[str, List[str]] = {}
        self._game_index: Dict[str, List[str]] = {}

        os.makedirs(store_path, exist_ok=True)
        self._load()

    def add_knowledge(self, knowledge: StrategicKnowledge) -> str:
        """Add or update a piece of strategic knowledge."""
        self.knowledge[knowledge.knowledge_id] = knowledge

        # Update indices
        cat = knowledge.category
        if cat not in self._category_index:
            self._category_index[cat] = []
        if knowledge.knowledge_id not in self._category_index[cat]:
            self._category_index[cat].append(knowledge.knowledge_id)

        game = knowledge.game_source
        if game not in self._game_index:
            self._game_index[game] = []
        if knowledge.knowledge_id not in self._game_index[game]:
            self._game_index[game].append(knowledge.knowledge_id)

        return knowledge.knowledge_id

    def query_knowledge(self, category: str = None,
                        game_source: str = None,
                        min_confidence: float = 0.0,
                        min_effectiveness: float = 0.0,
                        conditions: Dict[str, float] = None,
                        limit: int = 10) -> List[StrategicKnowledge]:
        """Query knowledge base with filters."""
        results = list(self.knowledge.values())

        if category:
            cat_ids = set(self._category_index.get(category, []))
            results = [k for k in results if k.knowledge_id in cat_ids]

        if game_source:
            game_ids = set(self._game_index.get(game_source, []))
            results = [k for k in results if k.knowledge_id in game_ids]

        results = [k for k in results
                   if k.confidence >= min_confidence
                   and k.effectiveness >= min_effectiveness]

        if conditions:
            # Score by condition match
            scored = []
            for k in results:
                match_score = self._condition_match_score(conditions, k.conditions)
                scored.append((k, match_score))
            scored.sort(key=lambda x: x[1] * x[0].effectiveness, reverse=True)
            results = [k for k, _ in scored[:limit]]
        else:
            results.sort(key=lambda k: k.effectiveness * k.confidence, reverse=True)
            results = results[:limit]

        return results

    def _condition_match_score(self, query_conditions: Dict[str, float],
                               knowledge_conditions: Dict[str, float]) -> float:
        """Score how well query conditions match knowledge conditions."""
        if not query_conditions or not knowledge_conditions:
            return 0.5

        total_score = 0.0
        count = 0
        for key, query_val in query_conditions.items():
            if key in knowledge_conditions:
                diff = abs(query_val - knowledge_conditions[key])
                total_score += max(0, 1.0 - diff)
                count += 1

        return total_score / max(count, 1)

    def update_effectiveness(self, knowledge_id: str, success: bool):
        """Update a knowledge entry based on application outcome."""
        k = self.knowledge.get(knowledge_id)
        if not k:
            return

        k.times_applied += 1
        if success:
            k.times_successful += 1

        # Update effectiveness using exponential moving average
        alpha = 0.1
        outcome = 1.0 if success else 0.0
        k.effectiveness = k.effectiveness * (1 - alpha) + outcome * alpha

        # Update confidence (increases with more applications)
        k.confidence = min(1.0, k.confidence + 0.01)

        k.updated_at = time.time()

    def extract_from_patterns(self, pattern_memory: DualMemory,
                              game_source: str = "micrortsai") -> int:
        """
        Extract strategic knowledge from patterns in the dual memory.
        Returns number of new knowledge entries created.
        """
        count = 0

        # Get high-confidence patterns from the RTS domain
        rts_patterns = pattern_memory.patterns.get_by_domain("rts_strategy")
        high_conf = [p for p in rts_patterns if p.confidence > 0.6]

        for pattern in high_conf:
            # Check if we already have knowledge for this pattern
            existing = [k for k in self.knowledge.values()
                        if pattern.id in k.pattern_ids]
            if existing:
                continue

            # Convert pattern to strategic knowledge
            knowledge = self._pattern_to_knowledge(pattern, game_source)
            if knowledge:
                self.add_knowledge(knowledge)
                count += 1

        return count

    def _pattern_to_knowledge(self, pattern: Pattern,
                              game_source: str) -> Optional[StrategicKnowledge]:
        """Convert a learned pattern into strategic knowledge."""
        sig = pattern.signature
        if len(sig) < 4:
            return None

        # Interpret pattern signature as strategic indicators
        # This mapping depends on how we encode game states
        knowledge_id = f"sk_{pattern.id[:12]}"

        # Determine category based on dominant signal
        if len(sig) >= 6:
            if sig[0] > 0.5:
                category = StrategyCategory.MILITARY.value
            elif sig[1] > 0.3:
                category = StrategyCategory.ECONOMY.value
            else:
                category = StrategyCategory.TIMING.value
        else:
            category = StrategyCategory.ADAPTATION.value

        conditions = {
            'game_phase': sig[min(5, len(sig) - 1)] if len(sig) > 5 else 0.5,
            'resource_level': sig[min(1, len(sig) - 1)] if len(sig) > 1 else 0.5,
        }

        actions = {
            'aggression': max(0, min(1, sig[0])),
            'economy_focus': max(0, min(1, sig[min(1, len(sig) - 1)])),
        }

        return StrategicKnowledge(
            knowledge_id=knowledge_id,
            category=category,
            description=f"Pattern from {game_source}: {pattern.pattern_type.value}",
            conditions=conditions,
            actions=actions,
            effectiveness=pattern.confidence,
            confidence=min(1.0, pattern.activation_count / 10.0),
            game_source=game_source,
            pattern_ids=[pattern.id],
        )

    def save(self):
        """Persist knowledge store to disk."""
        data = {
            kid: k.to_dict() for kid, k in self.knowledge.items()
        }
        filepath = os.path.join(self.store_path, 'strategic_knowledge.json')
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load knowledge store from disk."""
        filepath = os.path.join(self.store_path, 'strategic_knowledge.json')
        if not os.path.exists(filepath):
            return

        with open(filepath, 'r') as f:
            data = json.load(f)

        for kid, kdata in data.items():
            knowledge = StrategicKnowledge.from_dict(kdata)
            self.add_knowledge(knowledge)

    def get_stats(self) -> Dict:
        """Get knowledge store statistics."""
        return {
            'total_knowledge': len(self.knowledge),
            'categories': {cat: len(ids) for cat, ids in self._category_index.items()},
            'games': {game: len(ids) for game, ids in self._game_index.items()},
            'avg_effectiveness': (
                sum(k.effectiveness for k in self.knowledge.values())
                / max(len(self.knowledge), 1)
            ),
            'avg_confidence': (
                sum(k.confidence for k in self.knowledge.values())
                / max(len(self.knowledge), 1)
            ),
        }
