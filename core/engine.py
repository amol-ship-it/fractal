"""
Pattern Recognition Engine - The core intelligence system

Based on the research principles:
- Comparison-first architecture (not top-down backpropagation)
- Bottom-up processing: digitization -> edge detection -> clustering -> aggregation
- Uses subtraction for edge detection, division for clustering
- Output of one level becomes input for the next (aggregation)
"""

import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random

from .pattern import Pattern, PatternType
from .memory import DualMemory


class ProcessingLevel(Enum):
    RAW = 0           # Raw input (pixels, samples)
    EDGE = 1          # Edge/gradient detection
    CLUSTER = 2       # Pattern clusters
    COMPOSITE = 3     # Composite patterns
    ABSTRACT = 4      # High-level abstractions


@dataclass
class ProcessingResult:
    """Result of processing at a single level"""
    level: ProcessingLevel
    features: List[List[float]]  # Extracted features
    patterns: List[Pattern]       # Discovered patterns
    predictions: List[Dict[str, Any]]  # What this level predicts


class PatternEngine:
    """
    The core pattern recognition engine implementing:
    1. Bottom-up comparison-first processing
    2. Hierarchical aggregation (pixels to eternity)
    3. Pattern discovery through statistical deviation
    4. Prediction as the core operation
    """

    def __init__(self, memory: DualMemory = None):
        self.memory = memory or DualMemory()
        self.processing_history: List[ProcessingResult] = []

        # Thresholds for pattern discovery
        self.edge_threshold = 0.1       # Minimum gradient for edge
        self.cluster_threshold = 0.7    # Similarity threshold for clustering
        self.novelty_threshold = 0.3    # Below this, it's a new pattern

    def process(self, raw_input: List[float], domain: str = "general") -> Dict[str, Any]:
        """
        Main processing pipeline - bottom-up through all levels
        This is the 'aggregation' mechanism: output of each level becomes input for next
        """
        results = {
            'levels': [],
            'discovered_patterns': [],
            'predictions': [],
            'final_representation': None
        }

        # Level 0: Raw digitization (input is already digital)
        current_features = [raw_input]

        # Level 1: Edge Detection (using subtraction)
        edge_result = self._detect_edges(current_features[0])
        results['levels'].append({
            'name': 'edge_detection',
            'method': 'subtraction',
            'features': edge_result
        })

        # Level 2: Clustering (using division/ratios)
        cluster_result = self._cluster_features(edge_result)
        results['levels'].append({
            'name': 'clustering',
            'method': 'division',
            'features': cluster_result
        })

        # Level 3: Pattern Discovery
        patterns = self._discover_patterns(cluster_result, domain)
        results['discovered_patterns'] = patterns

        # Store in memory
        for pattern in patterns:
            self.memory.patterns.store(pattern)

        # Level 4: Composition (try to form higher-level patterns)
        composite = self._compose_patterns(patterns, domain)
        if composite:
            results['discovered_patterns'].extend(composite)
            for cp in composite:
                self.memory.patterns.store(cp)

        # Generate predictions based on discovered patterns
        predictions = self._generate_predictions(patterns + composite)
        results['predictions'] = predictions

        # Final representation is the highest-level pattern signature
        all_patterns = patterns + composite
        if all_patterns:
            # Aggregate all pattern signatures
            final_sig = Pattern._aggregate_signatures([p.signature for p in all_patterns])
            results['final_representation'] = final_sig

        return results

    def _detect_edges(self, signal: List[float]) -> List[float]:
        """
        Edge detection using SUBTRACTION
        Compare adjacent values - if difference exceeds threshold, it's an edge

        This is the first comparison operation from the research
        """
        if len(signal) < 2:
            return signal

        edges = []
        for i in range(len(signal) - 1):
            # Subtraction: compare adjacent pixels/samples
            diff = signal[i + 1] - signal[i]

            # The result encodes:
            # - Magnitude: strength of the edge
            # - Sign: direction of change (positive = increasing, negative = decreasing)
            if abs(diff) >= self.edge_threshold:
                edges.append(diff)
            else:
                edges.append(0.0)

        return edges

    def _cluster_features(self, features: List[float]) -> List[List[float]]:
        """
        Clustering using DIVISION (ratios)
        Group similar features together based on their relationships

        This is the second comparison operation from the research
        """
        if not features:
            return []

        clusters = []
        current_cluster = []

        # Average for comparison
        avg = sum(abs(f) for f in features) / max(1, len(features))
        if avg == 0:
            avg = 1.0

        for f in features:
            # Division: compare to average to determine relationship
            ratio = f / avg if avg != 0 else 0

            # Positive pattern: stands out from average
            is_positive = abs(ratio) > 1.0

            if is_positive:
                current_cluster.append(f)
            elif current_cluster:
                clusters.append(current_cluster)
                current_cluster = []

        if current_cluster:
            clusters.append(current_cluster)

        # Return cluster signatures (mean of each cluster)
        return [[sum(c) / len(c)] if c else [0.0] for c in clusters] if clusters else [[0.0]]

    def _discover_patterns(self, clusters: List[List[float]], domain: str) -> List[Pattern]:
        """
        Discover atomic patterns from clusters

        A pattern is discovered when:
        1. It's significantly different from existing patterns (novelty)
        2. OR it reinforces an existing pattern (adds evidence)
        """
        discovered = []

        for cluster in clusters:
            if not cluster:
                continue

            # Create a candidate pattern
            candidate = Pattern.create_atomic(cluster, domain)

            # Check if similar pattern exists
            similar = self.memory.patterns.find_similar(
                candidate,
                threshold=self.cluster_threshold,
                limit=1,
                domain=domain
            )

            if similar:
                # Reinforce existing pattern (approximability)
                existing_pattern, similarity = similar[0]
                existing_pattern.refine(feedback=similarity, new_evidence=cluster)
                discovered.append(existing_pattern)
            else:
                # New pattern discovered (exploration)
                discovered.append(candidate)

        return discovered

    def _compose_patterns(self, patterns: List[Pattern], domain: str) -> List[Pattern]:
        """
        Recursive composition: combine patterns into higher-level patterns

        This implements the 'building block hierarchy' from the research:
        Atomic Patterns -> Sub-Patterns -> High-Level Abstractions
        """
        if len(patterns) < 2:
            return []

        composites = []

        # Try combining adjacent patterns
        for i in range(len(patterns) - 1):
            p1, p2 = patterns[i], patterns[i + 1]

            # Check if they have complementary signatures
            similarity = p1.similarity(p2)

            # Compose if they're related but not identical
            if 0.3 <= similarity <= 0.8:
                composite = Pattern.create_composite([p1, p2], domain)

                # Check if this composite already exists
                existing = self.memory.patterns.find_similar(
                    composite,
                    threshold=0.9,
                    limit=1
                )

                if existing:
                    existing[0][0].refine(feedback=similarity)
                else:
                    composites.append(composite)

        return composites

    def _generate_predictions(self, patterns: List[Pattern]) -> List[Dict[str, Any]]:
        """
        Generate predictions based on discovered patterns

        From the research: Intelligence is the search for reward by predicting what's next
        """
        predictions = []

        for pattern in patterns:
            # Find patterns that historically follow this one
            similar = self.memory.patterns.find_similar(pattern, threshold=0.5, limit=5)

            for similar_pattern, similarity in similar:
                predictions.append({
                    'source_pattern': pattern.id,
                    'predicted_pattern': similar_pattern.id,
                    'confidence': similarity * pattern.confidence * similar_pattern.confidence,
                    'domains': list(similar_pattern.domains)
                })

        # Sort by confidence
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        return predictions[:10]  # Top 10 predictions

    def query(self, query_input: List[float], domain: str = None) -> Dict[str, Any]:
        """
        Query the system with new input
        Returns matched patterns and predictions
        """
        # Create query pattern
        query_pattern = Pattern.create_atomic(query_input, domain or "query")

        # Find similar patterns
        matches = self.memory.patterns.find_similar(
            query_pattern,
            threshold=0.5,
            limit=10,
            domain=domain
        )

        # Get context for matched patterns
        pattern_ids = [p.id for p, _ in matches]
        context = self.memory.state.get_context(pattern_ids)

        # Generate predictions
        predictions = self._generate_predictions([p for p, _ in matches])

        return {
            'query_signature': query_input,
            'matches': [(p.to_dict(), score) for p, score in matches],
            'context': context,
            'predictions': predictions
        }

    def learn_sequence(self, sequence: List[List[float]], domain: str = "general") -> Dict[str, Any]:
        """
        Learn from a sequence of inputs
        This builds temporal patterns (patterns that span time)
        """
        all_patterns = []
        temporal_connections = []

        prev_patterns = None
        for i, item in enumerate(sequence):
            result = self.process(item, domain)
            current_patterns = result['discovered_patterns']
            all_patterns.extend(current_patterns)

            # Build temporal connections
            if prev_patterns:
                for pp in prev_patterns:
                    for cp in current_patterns:
                        temporal_connections.append({
                            'from': pp.id,
                            'to': cp.id,
                            'position': i
                        })

            prev_patterns = current_patterns

        return {
            'total_patterns': len(all_patterns),
            'temporal_connections': temporal_connections,
            'unique_patterns': len(set(p.id for p in all_patterns))
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            'memory': self.memory.stats(),
            'thresholds': {
                'edge': self.edge_threshold,
                'cluster': self.cluster_threshold,
                'novelty': self.novelty_threshold
            }
        }
