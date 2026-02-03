"""
Feedback Loop System - The First Pillar of Learning

From the research:
- Feedback Loops: The basic Stimulus/Response mechanism
- Learning from immediate or long-term signals
- The survival stake creates feedback loops necessary for true intelligence
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional
from enum import Enum
import time
import math

from .pattern import Pattern


class FeedbackType(Enum):
    IMMEDIATE = "immediate"    # Instant reward/penalty
    DELAYED = "delayed"        # Reward comes later
    INTRINSIC = "intrinsic"    # Self-generated (curiosity, novelty)
    EXTRINSIC = "extrinsic"    # External reward


@dataclass
class FeedbackSignal:
    """A feedback signal that drives learning"""
    signal_type: FeedbackType
    value: float  # -1.0 to 1.0 (negative = bad, positive = good)
    target_pattern_ids: List[str]
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class PredictionOutcome:
    """Tracks the outcome of a prediction for feedback"""
    prediction_id: str
    predicted_pattern_id: str
    actual_pattern_id: Optional[str]
    confidence: float
    was_correct: bool
    timestamp: float = field(default_factory=time.time)


class FeedbackLoop:
    """
    Implements continuous feedback-driven learning

    Core mechanisms:
    1. Prediction tracking - monitor what we predicted vs what happened
    2. Reward propagation - spread feedback to contributing patterns
    3. Temporal credit assignment - handle delayed feedback
    """

    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor  # For temporal credit assignment

        # History tracking
        self.predictions: Dict[str, Dict[str, Any]] = {}
        self.outcomes: List[PredictionOutcome] = []
        self.feedback_history: List[FeedbackSignal] = []

        # Pattern contribution tracking (for credit assignment)
        self.pattern_contributions: Dict[str, List[Dict[str, Any]]] = {}

    def register_prediction(self, prediction_id: str, predicted_pattern_id: str,
                          source_pattern_ids: List[str], confidence: float) -> str:
        """
        Register a prediction for later evaluation

        This is how we track what the system predicted
        """
        self.predictions[prediction_id] = {
            'predicted_pattern_id': predicted_pattern_id,
            'source_pattern_ids': source_pattern_ids,
            'confidence': confidence,
            'timestamp': time.time(),
            'evaluated': False
        }

        # Track which patterns contributed to this prediction
        for pid in source_pattern_ids:
            if pid not in self.pattern_contributions:
                self.pattern_contributions[pid] = []
            self.pattern_contributions[pid].append({
                'prediction_id': prediction_id,
                'contribution_weight': 1.0 / len(source_pattern_ids)
            })

        return prediction_id

    def evaluate_prediction(self, prediction_id: str, actual_pattern_id: Optional[str],
                           patterns_dict: Dict[str, Pattern]) -> FeedbackSignal:
        """
        Evaluate a prediction against what actually happened

        This closes the feedback loop
        """
        if prediction_id not in self.predictions:
            return None

        pred = self.predictions[prediction_id]
        if pred['evaluated']:
            return None

        pred['evaluated'] = True

        # Calculate correctness
        predicted = patterns_dict.get(pred['predicted_pattern_id'])
        actual = patterns_dict.get(actual_pattern_id) if actual_pattern_id else None

        if predicted and actual:
            # Use similarity as a continuous measure of correctness
            similarity = predicted.similarity(actual)
            was_correct = similarity > 0.7
            feedback_value = similarity * 2 - 1  # Map [0,1] to [-1,1]
        elif predicted and not actual:
            # Prediction was made but nothing happened
            was_correct = False
            feedback_value = -0.5
        else:
            was_correct = False
            feedback_value = -0.3

        # Record outcome
        outcome = PredictionOutcome(
            prediction_id=prediction_id,
            predicted_pattern_id=pred['predicted_pattern_id'],
            actual_pattern_id=actual_pattern_id,
            confidence=pred['confidence'],
            was_correct=was_correct
        )
        self.outcomes.append(outcome)

        # Create feedback signal
        feedback = FeedbackSignal(
            signal_type=FeedbackType.IMMEDIATE,
            value=feedback_value,
            target_pattern_ids=pred['source_pattern_ids'],
            context={
                'prediction_id': prediction_id,
                'was_correct': was_correct,
                'similarity': (feedback_value + 1) / 2
            }
        )
        self.feedback_history.append(feedback)

        return feedback

    def apply_feedback(self, feedback: FeedbackSignal, patterns_dict: Dict[str, Pattern]):
        """
        Apply feedback to patterns (update confidence/weights)

        This is the learning step - patterns are refined based on feedback
        """
        for pattern_id in feedback.target_pattern_ids:
            pattern = patterns_dict.get(pattern_id)
            if pattern:
                # Apply feedback using pattern's refine method
                pattern.refine(
                    feedback=(feedback.value + 1) / 2,  # Convert to [0,1]
                    new_evidence=None
                )

    def propagate_delayed_feedback(self, final_feedback: float,
                                   patterns_dict: Dict[str, Pattern]):
        """
        Propagate delayed feedback backward through time

        Uses temporal difference learning (discount factor)
        This handles the case where reward comes much later than the action
        """
        # Get recent predictions (not yet given feedback)
        recent_predictions = [
            (pid, pred) for pid, pred in self.predictions.items()
            if not pred['evaluated']
        ]

        # Sort by time (most recent first)
        recent_predictions.sort(key=lambda x: x[1]['timestamp'], reverse=True)

        current_value = final_feedback
        for prediction_id, pred in recent_predictions:
            # Apply discounted feedback
            discounted_feedback = FeedbackSignal(
                signal_type=FeedbackType.DELAYED,
                value=current_value,
                target_pattern_ids=pred['source_pattern_ids'],
                context={'discounted_from': final_feedback}
            )
            self.apply_feedback(discounted_feedback, patterns_dict)

            # Discount for next iteration
            current_value *= self.discount_factor

    def generate_intrinsic_feedback(self, pattern: Pattern,
                                    existing_patterns: List[Pattern]) -> FeedbackSignal:
        """
        Generate intrinsic (curiosity-driven) feedback

        Novelty is rewarding! This drives exploration.
        """
        if not existing_patterns:
            # First pattern is maximally novel
            novelty = 1.0
        else:
            # Calculate average similarity to existing patterns
            similarities = [pattern.similarity(ep) for ep in existing_patterns]
            avg_similarity = sum(similarities) / len(similarities)
            novelty = 1.0 - avg_similarity

        # Novelty is rewarding (but not too much - we want some familiarity)
        # Optimal novelty is around 0.5-0.7 (not too boring, not too confusing)
        optimal_novelty = 0.6
        feedback_value = 1.0 - abs(novelty - optimal_novelty) * 2

        return FeedbackSignal(
            signal_type=FeedbackType.INTRINSIC,
            value=feedback_value,
            target_pattern_ids=[pattern.id],
            context={'novelty': novelty}
        )

    def get_prediction_accuracy(self) -> Dict[str, float]:
        """Get overall prediction accuracy statistics"""
        if not self.outcomes:
            return {'accuracy': 0.0, 'total': 0}

        correct = sum(1 for o in self.outcomes if o.was_correct)
        total = len(self.outcomes)

        return {
            'accuracy': correct / total,
            'total': total,
            'correct': correct,
            'recent_accuracy': self._recent_accuracy(20)
        }

    def _recent_accuracy(self, n: int) -> float:
        """Get accuracy for last n predictions"""
        recent = self.outcomes[-n:] if len(self.outcomes) >= n else self.outcomes
        if not recent:
            return 0.0
        correct = sum(1 for o in recent if o.was_correct)
        return correct / len(recent)

    def get_pattern_performance(self, pattern_id: str) -> Dict[str, Any]:
        """Get performance metrics for a specific pattern"""
        contributions = self.pattern_contributions.get(pattern_id, [])
        if not contributions:
            return {'predictions': 0, 'contribution_score': 0.0}

        # Find outcomes for predictions this pattern contributed to
        prediction_ids = {c['prediction_id'] for c in contributions}
        relevant_outcomes = [o for o in self.outcomes if o.prediction_id in prediction_ids]

        correct = sum(1 for o in relevant_outcomes if o.was_correct)
        total = len(relevant_outcomes)

        return {
            'predictions': total,
            'correct': correct,
            'accuracy': correct / total if total > 0 else 0.0,
            'contribution_score': sum(c['contribution_weight'] for c in contributions)
        }


class ExplorationStrategy:
    """
    Exploration mechanism - The Fourth Pillar of Learning

    Building new patterns from scratch or by combining existing ones in novel ways
    """

    def __init__(self, exploration_rate: float = 0.2):
        self.exploration_rate = exploration_rate
        self.explored_combinations: set = set()

    def should_explore(self) -> bool:
        """Decide whether to explore or exploit"""
        import random
        return random.random() < self.exploration_rate

    def suggest_combination(self, patterns: List[Pattern],
                           max_combine: int = 3) -> List[Pattern]:
        """
        Suggest novel pattern combinations to try

        This is how we discover new higher-level patterns
        """
        import random

        if len(patterns) < 2:
            return []

        # Try to find a combination we haven't explored
        for _ in range(10):  # Max attempts
            n = random.randint(2, min(max_combine, len(patterns)))
            selected = random.sample(patterns, n)
            combination_key = tuple(sorted(p.id for p in selected))

            if combination_key not in self.explored_combinations:
                self.explored_combinations.add(combination_key)
                return selected

        return []

    def generate_variation(self, pattern: Pattern, variation_strength: float = 0.1) -> Pattern:
        """
        Generate a variation of an existing pattern

        This explores the space around known good patterns
        """
        import random

        # Add random noise to signature
        varied_sig = [
            v + random.gauss(0, variation_strength)
            for v in pattern.signature
        ]

        # Create new pattern with varied signature
        return Pattern.create_atomic(varied_sig, list(pattern.domains)[0] if pattern.domains else "general")

    def decay_exploration(self, decay_rate: float = 0.99):
        """
        Decay exploration rate over time (shift towards exploitation)
        """
        self.exploration_rate *= decay_rate
        self.exploration_rate = max(0.05, self.exploration_rate)  # Minimum exploration
