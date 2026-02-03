"""
Memory System - Dual-type memory as described in the research

Type A: Patterns (The Code) - Highly compressed abstractions, static and optimized
Type B: State (The Data) - Context and instance data, dynamic

The Archival Mechanism: Active memory holds 'Hot' context, while 'Cold' patterns
are stored in compressed formats.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import OrderedDict
import time
import json

from .pattern import Pattern, PatternType


@dataclass
class StateEntry:
    """A single entry in state memory - dynamic, contextual"""
    data: Any
    timestamp: float
    relevance: float = 1.0
    access_count: int = 0
    linked_patterns: List[str] = field(default_factory=list)


class PatternMemory:
    """
    Type A Memory: Pattern Storage
    - Stores compressed abstractions
    - Hierarchical organization (atomic -> composite -> abstract)
    - Optimized for retrieval by similarity
    """

    def __init__(self, max_patterns: int = 10000):
        self.patterns: Dict[str, Pattern] = {}
        self.max_patterns = max_patterns

        # Index by hierarchy level
        self.atomic_patterns: Dict[str, Pattern] = {}
        self.composite_patterns: Dict[str, Pattern] = {}
        self.abstract_patterns: Dict[str, Pattern] = {}

        # Domain index for cross-domain retrieval
        self.domain_index: Dict[str, List[str]] = {}

    def store(self, pattern: Pattern) -> bool:
        """Store a pattern in memory"""
        if len(self.patterns) >= self.max_patterns:
            self._archive_cold_patterns()

        self.patterns[pattern.id] = pattern

        # Update hierarchy index
        if pattern.pattern_type == PatternType.ATOMIC:
            self.atomic_patterns[pattern.id] = pattern
        elif pattern.pattern_type == PatternType.COMPOSITE:
            self.composite_patterns[pattern.id] = pattern
        else:
            self.abstract_patterns[pattern.id] = pattern

        # Update domain index
        for domain in pattern.domains:
            if domain not in self.domain_index:
                self.domain_index[domain] = []
            if pattern.id not in self.domain_index[domain]:
                self.domain_index[domain].append(pattern.id)

        return True

    def retrieve(self, pattern_id: str) -> Optional[Pattern]:
        """Retrieve a specific pattern by ID"""
        pattern = self.patterns.get(pattern_id)
        if pattern:
            pattern.activate()
        return pattern

    def find_similar(self, query_pattern: Pattern, threshold: float = 0.7,
                     limit: int = 10, domain: str = None) -> List[Tuple[Pattern, float]]:
        """
        Find patterns similar to query - the core comparison operation
        Returns list of (pattern, similarity_score) tuples
        """
        results = []

        # If domain specified, search only that domain
        search_ids = self.domain_index.get(domain, list(self.patterns.keys())) if domain else list(self.patterns.keys())

        for pid in search_ids:
            pattern = self.patterns.get(pid)
            if pattern and pattern.id != query_pattern.id:
                similarity = query_pattern.similarity(pattern)
                if similarity >= threshold:
                    results.append((pattern, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def get_by_domain(self, domain: str) -> List[Pattern]:
        """Cross-domain application: get all patterns for a domain"""
        pattern_ids = self.domain_index.get(domain, [])
        return [self.patterns[pid] for pid in pattern_ids if pid in self.patterns]

    def get_hierarchy(self, pattern_id: str) -> Dict[str, Any]:
        """
        Get the full hierarchy tree for a pattern
        This reveals the recursive composition structure
        """
        pattern = self.patterns.get(pattern_id)
        if not pattern:
            return {}

        result = {
            'pattern': pattern.to_dict(),
            'children': []
        }

        for sub_id in pattern.sub_patterns:
            child_hierarchy = self.get_hierarchy(sub_id)
            if child_hierarchy:
                result['children'].append(child_hierarchy)

        return result

    def _archive_cold_patterns(self):
        """
        The Archival Mechanism: Move rarely-used patterns to cold storage
        Preserving storylines rather than full raw data
        """
        # Sort by activation count and age
        patterns_by_usage = sorted(
            self.patterns.values(),
            key=lambda p: (p.activation_count, p.confidence)
        )

        # Archive bottom 10%
        num_to_archive = len(patterns_by_usage) // 10
        for pattern in patterns_by_usage[:num_to_archive]:
            self._remove_pattern(pattern.id)

    def _remove_pattern(self, pattern_id: str):
        """Remove a pattern from all indices"""
        if pattern_id in self.patterns:
            pattern = self.patterns.pop(pattern_id)

            # Remove from hierarchy indices
            self.atomic_patterns.pop(pattern_id, None)
            self.composite_patterns.pop(pattern_id, None)
            self.abstract_patterns.pop(pattern_id, None)

            # Remove from domain indices
            for domain in pattern.domains:
                if domain in self.domain_index and pattern_id in self.domain_index[domain]:
                    self.domain_index[domain].remove(pattern_id)

    def stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            'total_patterns': len(self.patterns),
            'atomic': len(self.atomic_patterns),
            'composite': len(self.composite_patterns),
            'abstract': len(self.abstract_patterns),
            'domains': list(self.domain_index.keys()),
            'avg_confidence': sum(p.confidence for p in self.patterns.values()) / max(1, len(self.patterns))
        }


class StateMemory:
    """
    Type B Memory: State Storage
    - Holds context and instance data
    - Dynamic and temporal
    - Fades over time (recency bias)
    """

    def __init__(self, max_entries: int = 1000, decay_rate: float = 0.95):
        self.entries: OrderedDict[str, StateEntry] = OrderedDict()
        self.max_entries = max_entries
        self.decay_rate = decay_rate

    def store(self, key: str, data: Any, linked_patterns: List[str] = None):
        """Store state data with optional pattern links"""
        if len(self.entries) >= self.max_entries:
            self._evict_old_entries()

        self.entries[key] = StateEntry(
            data=data,
            timestamp=time.time(),
            relevance=1.0,
            linked_patterns=linked_patterns or []
        )

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve state data"""
        entry = self.entries.get(key)
        if entry:
            entry.access_count += 1
            entry.relevance = min(1.0, entry.relevance + 0.1)
            return entry.data
        return None

    def get_context(self, pattern_ids: List[str], limit: int = 10) -> List[Tuple[str, Any]]:
        """Get state entries linked to specific patterns"""
        results = []

        for key, entry in self.entries.items():
            if any(pid in entry.linked_patterns for pid in pattern_ids):
                results.append((key, entry.data, entry.relevance))

        # Sort by relevance
        results.sort(key=lambda x: x[2], reverse=True)
        return [(k, d) for k, d, _ in results[:limit]]

    def decay_all(self):
        """Apply temporal decay to all entries"""
        for entry in self.entries.values():
            entry.relevance *= self.decay_rate

    def _evict_old_entries(self):
        """Remove lowest relevance entries"""
        sorted_entries = sorted(
            self.entries.items(),
            key=lambda x: x[1].relevance
        )

        # Remove bottom 20%
        num_to_remove = len(sorted_entries) // 5
        for key, _ in sorted_entries[:num_to_remove]:
            del self.entries[key]

    def stats(self) -> Dict[str, Any]:
        """Get state memory statistics"""
        return {
            'total_entries': len(self.entries),
            'avg_relevance': sum(e.relevance for e in self.entries.values()) / max(1, len(self.entries)),
            'avg_access_count': sum(e.access_count for e in self.entries.values()) / max(1, len(self.entries))
        }


class DualMemory:
    """
    Combined dual-memory system
    Implements the research principle of Pattern (Code) + State (Data) memory
    """

    def __init__(self, max_patterns: int = 10000, max_state: int = 1000):
        self.patterns = PatternMemory(max_patterns)
        self.state = StateMemory(max_state)

    def process_experience(self, input_data: Any, discovered_patterns: List[Pattern],
                          context_key: str = None):
        """
        Process a new experience:
        1. Store discovered patterns in pattern memory
        2. Store context in state memory with pattern links
        """
        pattern_ids = []

        for pattern in discovered_patterns:
            self.patterns.store(pattern)
            pattern_ids.append(pattern.id)

        if context_key:
            self.state.store(
                key=context_key,
                data=input_data,
                linked_patterns=pattern_ids
            )

        return pattern_ids

    def recall(self, query_pattern: Pattern, include_context: bool = True) -> Dict[str, Any]:
        """
        Recall related patterns and optionally their associated context
        """
        similar = self.patterns.find_similar(query_pattern)

        result = {
            'patterns': [(p.to_dict(), score) for p, score in similar],
            'context': []
        }

        if include_context and similar:
            pattern_ids = [p.id for p, _ in similar]
            result['context'] = self.state.get_context(pattern_ids)

        return result

    def stats(self) -> Dict[str, Any]:
        """Get combined memory statistics"""
        return {
            'pattern_memory': self.patterns.stats(),
            'state_memory': self.state.stats()
        }
