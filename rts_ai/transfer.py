"""
Transfer Learning Bridge - Connects MicroRTS learnings to future games.

Provides a framework for transferring learned RTS concepts across games.
The key insight: strategic concepts like "economy before military",
"map control", "timing attacks", and "unit composition" are universal
across all RTS games, from MicroRTS to Age of Empires.

Transfer hierarchy:
1. Game-specific patterns (MicroRTS unit builds, map layouts)
2. Strategic concepts (build orders, timing, aggression levels)
3. Abstract principles (resource management, risk/reward, adaptation)

The abstract principles transfer most readily to new games.
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

from core.pattern import Pattern, PatternType
from core.memory import DualMemory
from rts_ai.knowledge_store import KnowledgeStore, StrategicKnowledge


class AbstractionLevel(Enum):
    """How abstract/transferable a concept is."""
    GAME_SPECIFIC = 0   # Only works in this exact game
    GENRE_SPECIFIC = 1  # Works in similar RTS games
    UNIVERSAL = 2       # Works across all strategy games


@dataclass
class StrategicConcept:
    """
    A high-level strategic concept that can transfer across games.

    Examples:
    - "Rush": Attack early before opponent builds up (universal)
    - "Boom": Prioritize economy, build many workers (genre-specific)
    - "Contain": Limit enemy expansion (universal)
    - "Tech switch": Change unit composition mid-game (genre-specific)
    """
    concept_id: str
    name: str
    description: str
    abstraction_level: str
    preconditions: Dict[str, str]    # Abstract conditions
    expected_outcomes: Dict[str, str]  # What should happen
    effectiveness_by_game: Dict[str, float] = field(default_factory=dict)
    source_patterns: List[str] = field(default_factory=list)
    source_knowledge: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategicConcept':
        return cls(**data)


# Pre-defined universal strategic concepts that emerge from RTS play
UNIVERSAL_CONCEPTS = [
    StrategicConcept(
        concept_id="rush",
        name="Early Aggression (Rush)",
        description="Attack the opponent before they can build up defenses. "
                    "Sacrifice long-term economy for immediate military advantage.",
        abstraction_level=AbstractionLevel.UNIVERSAL.name,
        preconditions={
            "game_phase": "early",
            "relative_military": "any",
            "enemy_defenses": "low",
        },
        expected_outcomes={
            "enemy_economy": "disrupted",
            "own_economy": "sacrificed",
            "tempo": "seized",
        },
    ),
    StrategicConcept(
        concept_id="boom",
        name="Economic Boom",
        description="Prioritize resource gathering and infrastructure. "
                    "Build many workers and expand before creating military.",
        abstraction_level=AbstractionLevel.UNIVERSAL.name,
        preconditions={
            "game_phase": "early_to_mid",
            "enemy_aggression": "low",
            "resource_availability": "high",
        },
        expected_outcomes={
            "own_economy": "maximized",
            "late_game_advantage": "high",
            "early_vulnerability": "high",
        },
    ),
    StrategicConcept(
        concept_id="timing_attack",
        name="Timing Attack",
        description="Build up to a specific military threshold, then attack "
                    "at a timing when you have a temporary advantage.",
        abstraction_level=AbstractionLevel.UNIVERSAL.name,
        preconditions={
            "game_phase": "mid",
            "army_size": "threshold_reached",
            "enemy_preparation": "incomplete",
        },
        expected_outcomes={
            "battle_outcome": "favorable",
            "map_control": "gained",
        },
    ),
    StrategicConcept(
        concept_id="counter_composition",
        name="Counter Unit Composition",
        description="Build units that counter the opponent's army composition. "
                    "Observe what the enemy builds and respond accordingly.",
        abstraction_level=AbstractionLevel.GENRE_SPECIFIC.name,
        preconditions={
            "enemy_composition": "observed",
            "resources": "available",
            "production_capacity": "sufficient",
        },
        expected_outcomes={
            "combat_efficiency": "improved",
            "resource_efficiency": "improved",
        },
    ),
    StrategicConcept(
        concept_id="map_control",
        name="Map Control",
        description="Spread units across the map to control key positions. "
                    "Deny opponent access to resources and information.",
        abstraction_level=AbstractionLevel.UNIVERSAL.name,
        preconditions={
            "army_size": "moderate",
            "map_vision": "any",
        },
        expected_outcomes={
            "information_advantage": "gained",
            "resource_control": "expanded",
            "strategic_options": "increased",
        },
    ),
    StrategicConcept(
        concept_id="defend_and_counter",
        name="Defensive Counter",
        description="Defend against enemy attacks efficiently, then counter-attack "
                    "when the enemy is weakened from their failed assault.",
        abstraction_level=AbstractionLevel.UNIVERSAL.name,
        preconditions={
            "enemy_aggression": "high",
            "defensive_position": "strong",
        },
        expected_outcomes={
            "enemy_army": "weakened",
            "counter_attack_opportunity": "created",
        },
    ),
]


class TransferBridge:
    """
    Bridge for transferring learned knowledge between games.

    Flow:
    1. Learn patterns and knowledge from MicroRTS
    2. Extract abstract strategic concepts
    3. Map concepts to a new game's mechanics
    4. Provide strategy recommendations for the new game
    """

    def __init__(self, knowledge_store: KnowledgeStore,
                 concepts_path: str = "strategic_concepts"):
        self.knowledge_store = knowledge_store
        self.concepts_path = concepts_path
        self.concepts: Dict[str, StrategicConcept] = {}

        os.makedirs(concepts_path, exist_ok=True)
        self._init_universal_concepts()
        self._load_concepts()

    def _init_universal_concepts(self):
        """Initialize with universal concepts."""
        for concept in UNIVERSAL_CONCEPTS:
            if concept.concept_id not in self.concepts:
                self.concepts[concept.concept_id] = concept

    def extract_concepts_from_training(self, pattern_memory: DualMemory,
                                       game_name: str = "micrortsai"
                                       ) -> List[StrategicConcept]:
        """
        Extract strategic concepts from training results.

        Analyzes learned patterns and knowledge to identify
        which universal concepts the agent has discovered.
        """
        extracted = []

        # Get all knowledge from the game
        game_knowledge = self.knowledge_store.query_knowledge(
            game_source=game_name,
            min_confidence=0.3,
            limit=50,
        )

        # Analyze knowledge to map to concepts
        for k in game_knowledge:
            matched_concept = self._match_to_concept(k)
            if matched_concept:
                # Update concept with game-specific data
                matched_concept.effectiveness_by_game[game_name] = k.effectiveness
                matched_concept.source_knowledge.append(k.knowledge_id)
                extracted.append(matched_concept)

        # Also create new concepts from high-performing patterns
        rts_patterns = pattern_memory.patterns.get_by_domain("rts_strategy")
        high_perf = [p for p in rts_patterns if p.confidence > 0.7]

        for pattern in high_perf:
            concept = self._pattern_to_concept(pattern, game_name)
            if concept and concept.concept_id not in self.concepts:
                self.concepts[concept.concept_id] = concept
                extracted.append(concept)

        return extracted

    def _match_to_concept(self, knowledge: StrategicKnowledge
                          ) -> Optional[StrategicConcept]:
        """Match a knowledge entry to a universal concept."""
        actions = knowledge.actions
        conditions = knowledge.conditions

        aggression = actions.get('aggression', 0.5)
        economy = actions.get('economy_focus', 0.5)
        game_phase = conditions.get('game_phase', 0.5)

        # Match patterns
        if aggression > 0.7 and game_phase < 0.3:
            return self.concepts.get("rush")
        elif economy > 0.7 and aggression < 0.3:
            return self.concepts.get("boom")
        elif aggression > 0.6 and 0.3 <= game_phase <= 0.6:
            return self.concepts.get("timing_attack")
        elif aggression < 0.3 and game_phase < 0.5:
            return self.concepts.get("defend_and_counter")

        return None

    def _pattern_to_concept(self, pattern: Pattern,
                            game_name: str) -> Optional[StrategicConcept]:
        """Create a new concept from a high-performing pattern."""
        sig = pattern.signature
        if len(sig) < 3:
            return None

        concept_id = f"learned_{pattern.id[:8]}"

        return StrategicConcept(
            concept_id=concept_id,
            name=f"Learned Strategy {pattern.id[:8]}",
            description=(
                f"Emergent strategy discovered in {game_name} training. "
                f"Pattern confidence: {pattern.confidence:.2f}, "
                f"activations: {pattern.activation_count}"
            ),
            abstraction_level=AbstractionLevel.GENRE_SPECIFIC.name,
            preconditions={
                "game_phase": "variable",
                "derived_from": game_name,
            },
            expected_outcomes={
                "effectiveness": str(round(pattern.confidence, 2)),
            },
            effectiveness_by_game={game_name: pattern.confidence},
            source_patterns=[pattern.id],
        )

    def get_recommendations(self, game_name: str,
                            current_conditions: Dict[str, str],
                            top_k: int = 3) -> List[Dict]:
        """
        Get strategy recommendations for a game given current conditions.

        This is how knowledge transfers to new games: by recommending
        abstract strategies that have worked in similar conditions.
        """
        scored_concepts = []

        for concept in self.concepts.values():
            # Score based on condition match
            match_score = self._score_condition_match(
                current_conditions, concept.preconditions
            )

            # Weight by effectiveness in this or similar games
            effectiveness = concept.effectiveness_by_game.get(game_name, 0.5)

            # Universal concepts get a bonus
            if concept.abstraction_level == AbstractionLevel.UNIVERSAL.name:
                match_score *= 1.2
            elif concept.abstraction_level == AbstractionLevel.GENRE_SPECIFIC.name:
                match_score *= 1.1

            total_score = match_score * effectiveness
            scored_concepts.append((concept, total_score))

        scored_concepts.sort(key=lambda x: x[1], reverse=True)

        recommendations = []
        for concept, score in scored_concepts[:top_k]:
            recommendations.append({
                'concept': concept.name,
                'description': concept.description,
                'score': score,
                'abstraction_level': concept.abstraction_level,
                'expected_outcomes': concept.expected_outcomes,
                'effectiveness_history': concept.effectiveness_by_game,
            })

        return recommendations

    def _score_condition_match(self, current: Dict[str, str],
                               required: Dict[str, str]) -> float:
        """Score how well current conditions match concept preconditions."""
        if not required:
            return 0.5

        matches = 0
        total = len(required)

        for key, required_val in required.items():
            if required_val in ("any", "variable"):
                matches += 0.5
            elif key in current:
                if current[key] == required_val:
                    matches += 1.0
                else:
                    matches += 0.2  # Partial match
            else:
                matches += 0.3  # Unknown condition

        return matches / max(total, 1)

    def generate_game_adapter(self, source_game: str,
                              target_game: str,
                              target_game_info: Dict) -> Dict:
        """
        Generate an adapter configuration for transferring knowledge
        to a new game.

        target_game_info should describe the target game's:
        - unit_types: available unit types
        - resources: types of resources
        - buildings: available buildings
        - map_features: map characteristics
        """
        adapter = {
            'source_game': source_game,
            'target_game': target_game,
            'concept_mappings': {},
            'unit_mappings': {},
            'resource_mappings': {},
            'recommendations': [],
        }

        # Map concepts to target game mechanics
        for concept in self.concepts.values():
            eff = concept.effectiveness_by_game.get(source_game, 0)
            if eff > 0.3:
                adapter['concept_mappings'][concept.concept_id] = {
                    'name': concept.name,
                    'source_effectiveness': eff,
                    'applicable': concept.abstraction_level != AbstractionLevel.GAME_SPECIFIC.name,
                    'adaptation_notes': self._generate_adaptation_notes(
                        concept, target_game_info
                    ),
                }

        # Suggest unit mappings (MicroRTS -> target game)
        if 'unit_types' in target_game_info:
            adapter['unit_mappings'] = {
                'worker': self._find_closest_unit('gatherer', target_game_info['unit_types']),
                'light_combat': self._find_closest_unit('fast_melee', target_game_info['unit_types']),
                'heavy_combat': self._find_closest_unit('slow_tank', target_game_info['unit_types']),
                'ranged': self._find_closest_unit('ranged', target_game_info['unit_types']),
            }

        # Top strategy recommendations
        adapter['recommendations'] = self.get_recommendations(
            target_game,
            current_conditions={"game_phase": "early"},
            top_k=5,
        )

        return adapter

    def _generate_adaptation_notes(self, concept: StrategicConcept,
                                   game_info: Dict) -> str:
        """Generate notes on how to adapt a concept to a new game."""
        if concept.concept_id == "rush":
            return "Identify the fastest military unit and attack before defenses are established."
        elif concept.concept_id == "boom":
            return "Maximize worker/gatherer count and resource income before building military."
        elif concept.concept_id == "timing_attack":
            return "Find key technology or unit thresholds and attack when reached."
        return "Apply the core principle in the context of the new game's mechanics."

    def _find_closest_unit(self, role: str, available_units: List[str]) -> Optional[str]:
        """Find the closest matching unit in the target game."""
        # Simple keyword matching
        role_keywords = {
            'gatherer': ['worker', 'villager', 'peasant', 'probe', 'scv', 'drone'],
            'fast_melee': ['knight', 'cavalry', 'zealot', 'zergling', 'scout'],
            'slow_tank': ['tank', 'siege', 'ultralisk', 'elephant', 'champion'],
            'ranged': ['archer', 'marine', 'ranger', 'hydralisk', 'crossbow'],
        }
        keywords = role_keywords.get(role, [])
        for unit in available_units:
            unit_lower = unit.lower()
            for kw in keywords:
                if kw in unit_lower:
                    return unit
        return available_units[0] if available_units else None

    def save(self):
        """Save concepts to disk."""
        data = {cid: c.to_dict() for cid, c in self.concepts.items()}
        filepath = os.path.join(self.concepts_path, 'concepts.json')
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_concepts(self):
        """Load concepts from disk."""
        filepath = os.path.join(self.concepts_path, 'concepts.json')
        if not os.path.exists(filepath):
            return
        with open(filepath, 'r') as f:
            data = json.load(f)
        for cid, cdata in data.items():
            self.concepts[cid] = StrategicConcept.from_dict(cdata)

    def get_stats(self) -> Dict:
        """Get transfer bridge statistics."""
        return {
            'total_concepts': len(self.concepts),
            'universal': sum(1 for c in self.concepts.values()
                             if c.abstraction_level == AbstractionLevel.UNIVERSAL.name),
            'genre_specific': sum(1 for c in self.concepts.values()
                                  if c.abstraction_level == AbstractionLevel.GENRE_SPECIFIC.name),
            'game_specific': sum(1 for c in self.concepts.values()
                                 if c.abstraction_level == AbstractionLevel.GAME_SPECIFIC.name),
            'games_with_data': list(set(
                game for c in self.concepts.values()
                for game in c.effectiveness_by_game.keys()
            )),
        }
