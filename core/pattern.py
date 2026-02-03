"""
Pattern - The atomic unit of learning in Recursive Learning AI

Based on the research principles:
- Patterns are recursive features that behave like building blocks
- They can reference themselves and form hierarchies
- They apply across domains (cross-domain application)
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set
from enum import Enum
import math


class PatternType(Enum):
    ATOMIC = "atomic"        # Base-level pattern (like phonemes)
    COMPOSITE = "composite"  # Built from sub-patterns
    ABSTRACT = "abstract"    # High-level abstraction


@dataclass
class Pattern:
    """
    A Pattern represents a learned feature that can be:
    1. Atomic - discovered directly from raw data
    2. Composite - built by combining sub-patterns
    3. Abstract - a high-level concept spanning multiple domains
    """
    id: str
    pattern_type: PatternType
    signature: List[float]  # The "shape" of this pattern
    sub_patterns: List[str] = field(default_factory=list)  # IDs of component patterns

    # Metadata
    confidence: float = 0.5  # How reliable this pattern is (0-1)
    activation_count: int = 0  # How often this pattern has been used
    domains: Set[str] = field(default_factory=set)  # Domains where this pattern applies

    # Approximability - patterns can be refined over time
    version: int = 1
    history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.domains, list):
            self.domains = set(self.domains)

    @classmethod
    def create_atomic(cls, signature: List[float], domain: str = "general") -> 'Pattern':
        """Create a base-level atomic pattern from raw signal"""
        pattern_id = cls._generate_id(signature)
        return cls(
            id=pattern_id,
            pattern_type=PatternType.ATOMIC,
            signature=signature,
            domains={domain}
        )

    @classmethod
    def create_composite(cls, sub_patterns: List['Pattern'], domain: str = "general") -> 'Pattern':
        """
        Create a composite pattern from sub-patterns
        This is the recursive composition mechanism
        """
        # Combine signatures using weighted aggregation
        combined_signature = cls._aggregate_signatures([p.signature for p in sub_patterns])
        pattern_id = cls._generate_id(combined_signature, [p.id for p in sub_patterns])

        # Inherit domains from sub-patterns (cross-domain application)
        all_domains = {domain}
        for p in sub_patterns:
            all_domains.update(p.domains)

        return cls(
            id=pattern_id,
            pattern_type=PatternType.COMPOSITE,
            signature=combined_signature,
            sub_patterns=[p.id for p in sub_patterns],
            domains=all_domains
        )

    @staticmethod
    def _generate_id(signature: List[float], sub_ids: List[str] = None) -> str:
        """Generate a unique ID based on pattern content"""
        content = json.dumps({
            'sig': [round(s, 4) for s in signature],
            'subs': sub_ids or []
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @staticmethod
    def _aggregate_signatures(signatures: List[List[float]]) -> List[float]:
        """
        Aggregate multiple signatures into one
        This is the 'aggregation' mechanism from the research
        """
        if not signatures:
            return []

        max_len = max(len(s) for s in signatures)
        result = [0.0] * max_len

        for sig in signatures:
            for i, val in enumerate(sig):
                result[i] += val

        # Normalize
        n = len(signatures)
        return [v / n for v in result]

    def similarity(self, other: 'Pattern') -> float:
        """
        Compare this pattern to another using cosine similarity
        This is the core 'comparison' operation
        """
        # Pad signatures to same length
        len_self = len(self.signature)
        len_other = len(other.signature)
        max_len = max(len_self, len_other)

        sig1 = self.signature + [0.0] * (max_len - len_self)
        sig2 = other.signature + [0.0] * (max_len - len_other)

        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(sig1, sig2))
        norm1 = math.sqrt(sum(a * a for a in sig1))
        norm2 = math.sqrt(sum(b * b for b in sig2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def refine(self, feedback: float, new_evidence: List[float] = None):
        """
        Approximability: Refine pattern based on feedback
        The pattern improves over time through iteration
        """
        # Store history for tracking evolution
        self.history.append({
            'version': self.version,
            'confidence': self.confidence,
            'signature': self.signature.copy()
        })

        # Update confidence based on feedback
        learning_rate = 0.1
        self.confidence = self.confidence + learning_rate * (feedback - self.confidence)
        self.confidence = max(0.0, min(1.0, self.confidence))

        # If new evidence provided, blend into signature
        if new_evidence:
            blend_factor = 0.1
            for i in range(min(len(self.signature), len(new_evidence))):
                self.signature[i] = (1 - blend_factor) * self.signature[i] + blend_factor * new_evidence[i]

        self.version += 1
        self.activation_count += 1

    def activate(self):
        """Record that this pattern was used"""
        self.activation_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize pattern for storage"""
        return {
            'id': self.id,
            'type': self.pattern_type.value,
            'signature': self.signature,
            'sub_patterns': self.sub_patterns,
            'confidence': self.confidence,
            'activation_count': self.activation_count,
            'domains': list(self.domains),
            'version': self.version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Pattern':
        """Deserialize pattern from storage"""
        return cls(
            id=data['id'],
            pattern_type=PatternType(data['type']),
            signature=data['signature'],
            sub_patterns=data.get('sub_patterns', []),
            confidence=data.get('confidence', 0.5),
            activation_count=data.get('activation_count', 0),
            domains=set(data.get('domains', ['general'])),
            version=data.get('version', 1)
        )

    def __repr__(self):
        return f"Pattern({self.id[:8]}..., type={self.pattern_type.value}, conf={self.confidence:.2f})"
