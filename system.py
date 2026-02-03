"""
Recursive Learning AI System - The Complete Self-Improving Intelligence

This integrates all components into a unified system that embodies the research principles:

THE FOUR PILLARS:
1. Feedback Loops - Learning from outcomes
2. Approximability - Continuous refinement
3. Composability - Building blocks that combine
4. Exploration - Discovering new patterns

THE ARCHITECTURE:
- Bottom-up comparison (not top-down rules)
- Dual memory (Patterns as code, State as data)
- Recursive composition (patterns reference sub-patterns)
- Cross-domain transfer (patterns apply across domains)
- Prediction as core operation
"""

import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import math

from core.pattern import Pattern, PatternType
from core.memory import DualMemory
from core.engine import PatternEngine
from core.feedback import FeedbackLoop, ExplorationStrategy, FeedbackType


@dataclass
class LearningEpisode:
    """A single learning episode with input, processing, and outcome"""
    episode_id: str
    input_data: List[float]
    domain: str
    discovered_patterns: List[str]
    predictions: List[Dict[str, Any]]
    feedback_received: Optional[float] = None


class RecursiveLearningAI:
    """
    The complete self-improving intelligence system

    Key insight from research: Intelligence is finding patterns, not following rules.
    This system discovers, refines, and composes patterns through experience.
    """

    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}

        # Core components
        self.memory = DualMemory(
            max_patterns=config.get('max_patterns', 10000),
            max_state=config.get('max_state', 1000)
        )
        self.engine = PatternEngine(self.memory)
        self.feedback_loop = FeedbackLoop(
            learning_rate=config.get('learning_rate', 0.1),
            discount_factor=config.get('discount_factor', 0.95)
        )
        self.explorer = ExplorationStrategy(
            exploration_rate=config.get('exploration_rate', 0.2)
        )

        # Episode tracking
        self.episodes: List[LearningEpisode] = []
        self.current_predictions: Dict[str, Dict[str, Any]] = {}

        # Cross-domain tracking
        self.domain_bridges: Dict[str, Dict[str, float]] = {}

    def perceive(self, input_data: List[float], domain: str = "general",
                context_key: str = None) -> Dict[str, Any]:
        """
        Perceive new input - the main entry point

        This triggers the full processing pipeline:
        1. Bottom-up pattern extraction
        2. Pattern matching and composition
        3. Prediction generation
        4. Exploration decisions
        """
        episode_id = str(uuid.uuid4())[:8]

        # Process through engine (bottom-up)
        result = self.engine.process(input_data, domain)

        # Store context in state memory
        if context_key:
            pattern_ids = [p.id for p in result['discovered_patterns']]
            self.memory.state.store(context_key, input_data, pattern_ids)

        # Register predictions for later evaluation
        for pred in result['predictions']:
            pred_id = str(uuid.uuid4())[:8]
            self.feedback_loop.register_prediction(
                prediction_id=pred_id,
                predicted_pattern_id=pred['predicted_pattern'],
                source_pattern_ids=[pred['source_pattern']],
                confidence=pred['confidence']
            )
            self.current_predictions[pred_id] = pred

        # Exploration: try novel combinations
        if self.explorer.should_explore():
            patterns = list(self.memory.patterns.patterns.values())
            to_combine = self.explorer.suggest_combination(patterns)
            if len(to_combine) >= 2:
                new_composite = Pattern.create_composite(to_combine, domain)
                self.memory.patterns.store(new_composite)
                result['explored_pattern'] = new_composite.to_dict()

        # Apply intrinsic feedback (curiosity reward)
        for pattern in result['discovered_patterns']:
            intrinsic = self.feedback_loop.generate_intrinsic_feedback(
                pattern,
                list(self.memory.patterns.patterns.values())
            )
            self.feedback_loop.apply_feedback(intrinsic, self.memory.patterns.patterns)

        # Record episode
        episode = LearningEpisode(
            episode_id=episode_id,
            input_data=input_data,
            domain=domain,
            discovered_patterns=[p.id for p in result['discovered_patterns']],
            predictions=result['predictions']
        )
        self.episodes.append(episode)

        return {
            'episode_id': episode_id,
            'patterns_discovered': len(result['discovered_patterns']),
            'patterns': [p.to_dict() for p in result['discovered_patterns']],
            'predictions': result['predictions'][:5],  # Top 5
            'representation': result['final_representation'],
            'explored': result.get('explored_pattern')
        }

    def receive_feedback(self, episode_id: str, feedback_value: float,
                        actual_outcome: List[float] = None):
        """
        Receive external feedback on an episode

        This closes the feedback loop and drives learning
        """
        # Find the episode
        episode = next((e for e in self.episodes if e.episode_id == episode_id), None)
        if not episode:
            return {'error': 'Episode not found'}

        episode.feedback_received = feedback_value

        # If actual outcome provided, evaluate predictions
        if actual_outcome:
            actual_pattern = Pattern.create_atomic(actual_outcome, episode.domain)
            actual_pattern = self._find_or_store_pattern(actual_pattern)

            # Evaluate all predictions from this episode
            for pred_id in self.current_predictions:
                self.feedback_loop.evaluate_prediction(
                    pred_id,
                    actual_pattern.id,
                    self.memory.patterns.patterns
                )

        # Apply direct feedback to discovered patterns
        for pattern_id in episode.discovered_patterns:
            pattern = self.memory.patterns.retrieve(pattern_id)
            if pattern:
                pattern.refine(feedback=(feedback_value + 1) / 2)

        return {
            'episode_id': episode_id,
            'feedback_applied': True,
            'accuracy': self.feedback_loop.get_prediction_accuracy()
        }

    def _find_or_store_pattern(self, pattern: Pattern) -> Pattern:
        """Find existing similar pattern or store new one"""
        similar = self.memory.patterns.find_similar(pattern, threshold=0.9, limit=1)
        if similar:
            return similar[0][0]
        self.memory.patterns.store(pattern)
        return pattern

    def query(self, query_input: List[float], domain: str = None) -> Dict[str, Any]:
        """
        Query the system - find matching patterns and get predictions
        """
        return self.engine.query(query_input, domain)

    def transfer_learning(self, source_domain: str, target_domain: str,
                         mapping_examples: List[Tuple[List[float], List[float]]]) -> Dict[str, Any]:
        """
        Cross-domain transfer learning

        From the research: Stored patterns apply to any input.
        The pattern of "Addition" applies to numbers, audio signals, or abstract ideas.
        """
        transferred_count = 0
        bridge_patterns = []

        # Get patterns from source domain
        source_patterns = self.memory.patterns.get_by_domain(source_domain)

        for source_input, target_input in mapping_examples:
            # Process both through engine
            source_result = self.engine.process(source_input, source_domain)
            target_result = self.engine.process(target_input, target_domain)

            # Find corresponding patterns
            for sp in source_result['discovered_patterns']:
                for tp in target_result['discovered_patterns']:
                    similarity = sp.similarity(tp)
                    if similarity > 0.5:
                        # Create a bridge pattern that spans both domains
                        bridge = Pattern.create_composite([sp, tp], f"{source_domain}_{target_domain}")
                        bridge.domains.add(source_domain)
                        bridge.domains.add(target_domain)
                        self.memory.patterns.store(bridge)
                        bridge_patterns.append(bridge)
                        transferred_count += 1

        # Record domain bridge
        bridge_key = f"{source_domain}->{target_domain}"
        self.domain_bridges[bridge_key] = {
            'patterns_transferred': transferred_count,
            'bridge_patterns': [b.id for b in bridge_patterns]
        }

        return {
            'source_domain': source_domain,
            'target_domain': target_domain,
            'patterns_transferred': transferred_count,
            'bridge_patterns': len(bridge_patterns)
        }

    def learn_sequence(self, sequence: List[List[float]], domain: str = "general") -> Dict[str, Any]:
        """
        Learn temporal patterns from a sequence

        This builds patterns that span time (like phonemes -> words -> sentences)
        """
        return self.engine.learn_sequence(sequence, domain)

    def get_hierarchy(self, pattern_id: str) -> Dict[str, Any]:
        """
        Get the full hierarchy of a pattern

        This reveals the recursive composition structure:
        Atomic -> Sub-Patterns -> High-Level Abstractions
        """
        return self.memory.patterns.get_hierarchy(pattern_id)

    def introspect(self) -> Dict[str, Any]:
        """
        Introspection - understand the system's current state

        Returns statistics about patterns, memory, predictions, etc.
        """
        return {
            'engine_stats': self.engine.get_statistics(),
            'prediction_accuracy': self.feedback_loop.get_prediction_accuracy(),
            'exploration_rate': self.explorer.exploration_rate,
            'total_episodes': len(self.episodes),
            'domain_bridges': self.domain_bridges,
            'pattern_types': {
                'atomic': len(self.memory.patterns.atomic_patterns),
                'composite': len(self.memory.patterns.composite_patterns),
                'abstract': len(self.memory.patterns.abstract_patterns)
            }
        }

    def save_state(self, filepath: str):
        """Save the system state to file"""
        state = {
            'patterns': {pid: p.to_dict() for pid, p in self.memory.patterns.patterns.items()},
            'episodes': [
                {
                    'episode_id': e.episode_id,
                    'domain': e.domain,
                    'discovered_patterns': e.discovered_patterns,
                    'feedback_received': e.feedback_received
                }
                for e in self.episodes
            ],
            'domain_bridges': self.domain_bridges,
            'exploration_rate': self.explorer.exploration_rate
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str):
        """Load system state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)

        # Restore patterns
        for pid, pdata in state.get('patterns', {}).items():
            pattern = Pattern.from_dict(pdata)
            self.memory.patterns.store(pattern)

        self.domain_bridges = state.get('domain_bridges', {})
        self.explorer.exploration_rate = state.get('exploration_rate', 0.2)


def demo():
    """Demonstrate the Recursive Learning AI system"""
    print("=" * 60)
    print("RECURSIVE LEARNING AI - Demo")
    print("Based on: 'Intelligence Is Finding Patterns, Not Following Rules'")
    print("=" * 60)

    # Create system
    ai = RecursiveLearningAI({
        'learning_rate': 0.15,
        'exploration_rate': 0.3
    })

    # Demo 1: Pattern Discovery
    print("\n1. PATTERN DISCOVERY")
    print("-" * 40)

    # Simulate a simple wave pattern
    import math
    wave1 = [math.sin(x * 0.5) for x in range(20)]
    wave2 = [math.sin(x * 0.5 + 0.1) for x in range(20)]  # Similar wave
    wave3 = [math.cos(x * 0.3) for x in range(20)]  # Different wave

    result1 = ai.perceive(wave1, domain="audio")
    print(f"Wave 1: Discovered {result1['patterns_discovered']} patterns")

    result2 = ai.perceive(wave2, domain="audio")
    print(f"Wave 2: Discovered {result2['patterns_discovered']} patterns")

    result3 = ai.perceive(wave3, domain="audio")
    print(f"Wave 3: Discovered {result3['patterns_discovered']} patterns")

    # Demo 2: Feedback Loop
    print("\n2. FEEDBACK LOOP")
    print("-" * 40)

    # Provide positive feedback for wave1
    feedback_result = ai.receive_feedback(result1['episode_id'], feedback_value=0.8)
    print(f"Applied positive feedback: accuracy = {feedback_result['accuracy']['accuracy']:.2%}")

    # Demo 3: Pattern Composition
    print("\n3. PATTERN COMPOSITION (Hierarchy)")
    print("-" * 40)

    stats = ai.introspect()
    print(f"Atomic patterns: {stats['pattern_types']['atomic']}")
    print(f"Composite patterns: {stats['pattern_types']['composite']}")

    # Demo 4: Cross-Domain Transfer
    print("\n4. CROSS-DOMAIN TRANSFER")
    print("-" * 40)

    # Create patterns in 'visual' domain similar to audio patterns
    visual_wave = [math.sin(x * 0.5) * 255 for x in range(20)]  # Same pattern, different scale
    transfer_result = ai.transfer_learning(
        source_domain="audio",
        target_domain="visual",
        mapping_examples=[(wave1, visual_wave)]
    )
    print(f"Transferred {transfer_result['patterns_transferred']} patterns from audio to visual")

    # Demo 5: Query
    print("\n5. QUERY SYSTEM")
    print("-" * 40)

    query_result = ai.query([math.sin(x * 0.5) for x in range(20)])
    print(f"Found {len(query_result['matches'])} matching patterns")
    if query_result['predictions']:
        print(f"Top prediction confidence: {query_result['predictions'][0]['confidence']:.2%}")

    # Final statistics
    print("\n" + "=" * 60)
    print("FINAL SYSTEM STATE")
    print("=" * 60)
    final_stats = ai.introspect()
    print(f"Total episodes: {final_stats['total_episodes']}")
    print(f"Total patterns: {sum(final_stats['pattern_types'].values())}")
    print(f"Exploration rate: {final_stats['exploration_rate']:.1%}")
    print(f"Memory stats: {final_stats['engine_stats']['memory']['pattern_memory']}")

    return ai


if __name__ == "__main__":
    demo()
