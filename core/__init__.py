"""
Recursive Learning AI - Core Components

A self-improving intelligence system based on the principles:
1. Feedback Loops - Learning from stimulus/response
2. Approximability - Iterative refinement
3. Composability - Reusing patterns in new contexts
4. Exploration - Building new patterns

The system implements:
- Pattern-based learning (not rule-based)
- Bottom-up comparison architecture
- Dual memory (Patterns + State)
- Recursive composition
- Cross-domain transfer
"""

from .pattern import Pattern, PatternType
from .memory import DualMemory, PatternMemory, StateMemory
from .engine import PatternEngine, ProcessingLevel, ProcessingResult
from .feedback import FeedbackLoop, FeedbackSignal, FeedbackType, ExplorationStrategy

__all__ = [
    'Pattern',
    'PatternType',
    'DualMemory',
    'PatternMemory',
    'StateMemory',
    'PatternEngine',
    'ProcessingLevel',
    'ProcessingResult',
    'FeedbackLoop',
    'FeedbackSignal',
    'FeedbackType',
    'ExplorationStrategy'
]
