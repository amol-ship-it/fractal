"""
Distributed Processing Workers

Implements parallel pattern processing using Ray actors.
Each worker is an independent entity that:
- Processes input data through the pattern pipeline
- Discovers and refines patterns
- Communicates with the distributed store
- Reports back to the coordinator

This embodies the "Swarm Intelligence" principle:
Independent entities that self-organize to create global coherence.
"""

import asyncio
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import math

# Attempt Ray import with fallback for testing
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    # Create mock decorator
    class ray:
        @staticmethod
        def remote(*args, **kwargs):
            def decorator(cls):
                return cls
            return decorator if not args else decorator(args[0])

        @staticmethod
        def init(*args, **kwargs):
            pass

        @staticmethod
        def get(refs):
            return refs

import sys
sys.path.insert(0, '..')
from core.pattern import Pattern, PatternType

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of processing tasks"""
    PROCESS_INPUT = "process_input"
    FIND_SIMILAR = "find_similar"
    COMPOSE_PATTERNS = "compose_patterns"
    APPLY_FEEDBACK = "apply_feedback"
    TRANSFER_LEARNING = "transfer_learning"
    BATCH_PROCESS = "batch_process"


@dataclass
class ProcessingTask:
    """A task to be processed by a worker"""
    task_id: str
    task_type: TaskType
    data: Any
    domain: str = "general"
    priority: int = 5  # 1-10, higher = more urgent
    created_at: float = field(default_factory=time.time)
    timeout: float = 300.0  # seconds

    @classmethod
    def create(cls, task_type: TaskType, data: Any,
               domain: str = "general", priority: int = 5) -> 'ProcessingTask':
        return cls(
            task_id=str(uuid.uuid4())[:12],
            task_type=task_type,
            data=data,
            domain=domain,
            priority=priority
        )


@dataclass
class ProcessingResult:
    """Result from a worker"""
    task_id: str
    worker_id: str
    success: bool
    patterns: List[Dict[str, Any]] = field(default_factory=list)
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    processing_time: float = 0.0


@ray.remote
class PatternWorker:
    """
    A distributed pattern processing worker.

    Each worker runs independently and:
    1. Pulls tasks from the queue
    2. Processes data through the pattern pipeline
    3. Stores discovered patterns in distributed storage
    4. Reports results back to coordinator
    """

    def __init__(self, worker_id: str, config: Dict[str, Any] = None):
        self.worker_id = worker_id
        self.config = config or {}

        # Processing parameters
        self.edge_threshold = self.config.get('edge_threshold', 0.1)
        self.cluster_threshold = self.config.get('cluster_threshold', 0.7)

        # Local pattern cache (hot patterns for fast comparison)
        self.local_patterns: Dict[str, Pattern] = {}
        self.max_local_patterns = self.config.get('max_local_patterns', 1000)

        # Statistics
        self.tasks_processed = 0
        self.patterns_discovered = 0
        self.total_processing_time = 0.0
        self.start_time = time.time()

        logger.info(f"Worker {worker_id} initialized")

    def get_status(self) -> Dict[str, Any]:
        """Get worker status"""
        uptime = time.time() - self.start_time
        return {
            'worker_id': self.worker_id,
            'tasks_processed': self.tasks_processed,
            'patterns_discovered': self.patterns_discovered,
            'avg_processing_time': (
                self.total_processing_time / self.tasks_processed
                if self.tasks_processed > 0 else 0
            ),
            'uptime_seconds': uptime,
            'local_patterns': len(self.local_patterns)
        }

    async def process_task(self, task: ProcessingTask) -> ProcessingResult:
        """Process a single task"""
        start_time = time.time()

        try:
            if task.task_type == TaskType.PROCESS_INPUT:
                result = await self._process_input(task)
            elif task.task_type == TaskType.FIND_SIMILAR:
                result = await self._find_similar(task)
            elif task.task_type == TaskType.COMPOSE_PATTERNS:
                result = await self._compose_patterns(task)
            elif task.task_type == TaskType.APPLY_FEEDBACK:
                result = await self._apply_feedback(task)
            elif task.task_type == TaskType.BATCH_PROCESS:
                result = await self._batch_process(task)
            else:
                result = ProcessingResult(
                    task_id=task.task_id,
                    worker_id=self.worker_id,
                    success=False,
                    error=f"Unknown task type: {task.task_type}"
                )

            processing_time = time.time() - start_time
            result.processing_time = processing_time

            self.tasks_processed += 1
            self.total_processing_time += processing_time

            return result

        except Exception as e:
            logger.error(f"Worker {self.worker_id} error: {e}")
            return ProcessingResult(
                task_id=task.task_id,
                worker_id=self.worker_id,
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )

    async def _process_input(self, task: ProcessingTask) -> ProcessingResult:
        """Process raw input through the pattern pipeline"""
        input_data = task.data
        domain = task.domain

        # Step 1: Edge detection (subtraction)
        edges = self._detect_edges(input_data)

        # Step 2: Clustering (division)
        clusters = self._cluster_features(edges)

        # Step 3: Pattern discovery
        patterns = []
        for cluster in clusters:
            pattern = self._discover_pattern(cluster, domain)
            patterns.append(pattern)
            self._cache_pattern(pattern)

        # Step 4: Try composition
        composites = self._compose_adjacent(patterns, domain)
        patterns.extend(composites)

        # Step 5: Generate predictions
        predictions = self._generate_predictions(patterns)

        self.patterns_discovered += len(patterns)

        return ProcessingResult(
            task_id=task.task_id,
            worker_id=self.worker_id,
            success=True,
            patterns=[p.to_dict() for p in patterns],
            predictions=predictions,
            metrics={
                'edges_detected': len([e for e in edges if e != 0]),
                'clusters_found': len(clusters),
                'patterns_discovered': len(patterns),
                'composites_created': len(composites)
            }
        )

    def _detect_edges(self, signal: List[float]) -> List[float]:
        """Edge detection using subtraction"""
        if len(signal) < 2:
            return signal

        edges = []
        for i in range(len(signal) - 1):
            diff = signal[i + 1] - signal[i]
            if abs(diff) >= self.edge_threshold:
                edges.append(diff)
            else:
                edges.append(0.0)

        return edges

    def _cluster_features(self, features: List[float]) -> List[List[float]]:
        """Cluster features using division (ratio to average)"""
        if not features:
            return []

        avg = sum(abs(f) for f in features) / max(1, len(features))
        if avg == 0:
            avg = 1.0

        clusters = []
        current_cluster = []

        for f in features:
            ratio = f / avg if avg != 0 else 0
            is_significant = abs(ratio) > 1.0

            if is_significant:
                current_cluster.append(f)
            elif current_cluster:
                clusters.append(current_cluster)
                current_cluster = []

        if current_cluster:
            clusters.append(current_cluster)

        return clusters if clusters else [[0.0]]

    def _discover_pattern(self, cluster: List[float], domain: str) -> Pattern:
        """Discover a pattern from a cluster"""
        signature = [sum(cluster) / len(cluster)] if cluster else [0.0]

        # Check for similar existing pattern
        candidate = Pattern.create_atomic(signature, domain)

        for existing in self.local_patterns.values():
            if candidate.similarity(existing) > self.cluster_threshold:
                # Reinforce existing pattern
                existing.refine(
                    feedback=0.6,
                    new_evidence=signature
                )
                return existing

        # New pattern
        return candidate

    def _compose_adjacent(self, patterns: List[Pattern],
                         domain: str) -> List[Pattern]:
        """Try to compose adjacent patterns"""
        composites = []

        for i in range(len(patterns) - 1):
            p1, p2 = patterns[i], patterns[i + 1]
            similarity = p1.similarity(p2)

            if 0.3 <= similarity <= 0.8:
                composite = Pattern.create_composite([p1, p2], domain)
                composites.append(composite)
                self._cache_pattern(composite)

        return composites

    def _generate_predictions(self, patterns: List[Pattern]) -> List[Dict[str, Any]]:
        """Generate predictions based on discovered patterns"""
        predictions = []

        for pattern in patterns:
            for cached_id, cached in self.local_patterns.items():
                if cached.id != pattern.id:
                    similarity = pattern.similarity(cached)
                    if similarity > 0.5:
                        predictions.append({
                            'source_pattern': pattern.id,
                            'predicted_pattern': cached.id,
                            'confidence': similarity * pattern.confidence * cached.confidence
                        })

        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        return predictions[:10]

    def _cache_pattern(self, pattern: Pattern):
        """Cache a pattern locally"""
        if len(self.local_patterns) >= self.max_local_patterns:
            # Remove least confident pattern
            min_conf_id = min(
                self.local_patterns.keys(),
                key=lambda k: self.local_patterns[k].confidence
            )
            del self.local_patterns[min_conf_id]

        self.local_patterns[pattern.id] = pattern

    async def _find_similar(self, task: ProcessingTask) -> ProcessingResult:
        """Find patterns similar to query"""
        query_data = task.data.get('query', [])
        threshold = task.data.get('threshold', 0.7)
        limit = task.data.get('limit', 10)

        query_pattern = Pattern.create_atomic(query_data, task.domain)

        results = []
        for pattern in self.local_patterns.values():
            similarity = query_pattern.similarity(pattern)
            if similarity >= threshold:
                results.append({
                    'pattern': pattern.to_dict(),
                    'similarity': similarity
                })

        results.sort(key=lambda x: x['similarity'], reverse=True)

        return ProcessingResult(
            task_id=task.task_id,
            worker_id=self.worker_id,
            success=True,
            patterns=[r['pattern'] for r in results[:limit]],
            metrics={'matches_found': len(results)}
        )

    async def _compose_patterns(self, task: ProcessingTask) -> ProcessingResult:
        """Compose patterns into higher-level patterns"""
        pattern_ids = task.data.get('pattern_ids', [])
        patterns = [
            self.local_patterns[pid]
            for pid in pattern_ids
            if pid in self.local_patterns
        ]

        if len(patterns) >= 2:
            composite = Pattern.create_composite(patterns, task.domain)
            self._cache_pattern(composite)

            return ProcessingResult(
                task_id=task.task_id,
                worker_id=self.worker_id,
                success=True,
                patterns=[composite.to_dict()],
                metrics={'composed_from': len(patterns)}
            )

        return ProcessingResult(
            task_id=task.task_id,
            worker_id=self.worker_id,
            success=False,
            error="Not enough patterns to compose"
        )

    async def _apply_feedback(self, task: ProcessingTask) -> ProcessingResult:
        """Apply feedback to patterns"""
        pattern_id = task.data.get('pattern_id')
        feedback_value = task.data.get('feedback', 0.5)

        if pattern_id in self.local_patterns:
            pattern = self.local_patterns[pattern_id]
            pattern.refine(feedback=feedback_value)

            return ProcessingResult(
                task_id=task.task_id,
                worker_id=self.worker_id,
                success=True,
                patterns=[pattern.to_dict()],
                metrics={
                    'new_confidence': pattern.confidence,
                    'version': pattern.version
                }
            )

        return ProcessingResult(
            task_id=task.task_id,
            worker_id=self.worker_id,
            success=False,
            error=f"Pattern {pattern_id} not found locally"
        )

    async def _batch_process(self, task: ProcessingTask) -> ProcessingResult:
        """Process a batch of inputs"""
        inputs = task.data.get('inputs', [])
        all_patterns = []
        all_predictions = []

        for input_data in inputs:
            sub_task = ProcessingTask.create(
                TaskType.PROCESS_INPUT,
                input_data,
                task.domain
            )
            result = await self._process_input(sub_task)
            all_patterns.extend(result.patterns)
            all_predictions.extend(result.predictions)

        return ProcessingResult(
            task_id=task.task_id,
            worker_id=self.worker_id,
            success=True,
            patterns=all_patterns,
            predictions=all_predictions[:20],
            metrics={
                'inputs_processed': len(inputs),
                'total_patterns': len(all_patterns)
            }
        )

    def receive_patterns(self, patterns: List[Dict[str, Any]]):
        """Receive patterns from other workers (for sync)"""
        for pdata in patterns:
            pattern = Pattern.from_dict(pdata)
            self._cache_pattern(pattern)

    def get_hot_patterns(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get most active patterns for sharing"""
        sorted_patterns = sorted(
            self.local_patterns.values(),
            key=lambda p: p.activation_count,
            reverse=True
        )
        return [p.to_dict() for p in sorted_patterns[:limit]]


class WorkerPool:
    """
    Manages a pool of PatternWorkers.
    Handles task distribution, load balancing, and worker lifecycle.
    """

    def __init__(self, num_workers: int = 4, config: Dict[str, Any] = None):
        self.num_workers = num_workers
        self.config = config or {}
        self.workers: List[Any] = []  # Ray actor handles
        self.worker_stats: Dict[str, Dict[str, Any]] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.results: Dict[str, ProcessingResult] = {}

    async def start(self):
        """Start the worker pool"""
        if RAY_AVAILABLE:
            for i in range(self.num_workers):
                worker_id = f"worker_{i}"
                worker = PatternWorker.remote(worker_id, self.config)
                self.workers.append(worker)
                logger.info(f"Started worker {worker_id}")
        else:
            # Fallback to local workers
            for i in range(self.num_workers):
                worker_id = f"worker_{i}"
                worker = PatternWorker(worker_id, self.config)
                self.workers.append(worker)
                logger.info(f"Started local worker {worker_id}")

    async def stop(self):
        """Stop all workers"""
        self.workers.clear()
        logger.info("Worker pool stopped")

    async def submit_task(self, task: ProcessingTask) -> str:
        """Submit a task to the pool"""
        await self.task_queue.put(task)
        return task.task_id

    async def process_task(self, task: ProcessingTask) -> ProcessingResult:
        """Process a task using the pool"""
        # Simple round-robin load balancing
        worker_idx = hash(task.task_id) % len(self.workers)
        worker = self.workers[worker_idx]

        if RAY_AVAILABLE:
            result = await worker.process_task.remote(task)
            return ray.get(result)
        else:
            return await worker.process_task(task)

    async def batch_process(self, tasks: List[ProcessingTask]) -> List[ProcessingResult]:
        """Process multiple tasks in parallel"""
        if RAY_AVAILABLE:
            # Distribute tasks across workers
            refs = []
            for i, task in enumerate(tasks):
                worker = self.workers[i % len(self.workers)]
                refs.append(worker.process_task.remote(task))

            results = ray.get(refs)
            return results
        else:
            # Process sequentially for fallback
            results = []
            for i, task in enumerate(tasks):
                worker = self.workers[i % len(self.workers)]
                result = await worker.process_task(task)
                results.append(result)
            return results

    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get statistics for all workers"""
        stats = {
            'num_workers': len(self.workers),
            'queue_size': self.task_queue.qsize(),
            'workers': []
        }

        for worker in self.workers:
            if RAY_AVAILABLE:
                worker_stats = ray.get(worker.get_status.remote())
            else:
                worker_stats = worker.get_status()
            stats['workers'].append(worker_stats)

        return stats

    async def sync_patterns(self):
        """Synchronize patterns across workers"""
        # Collect hot patterns from each worker
        all_patterns = []

        for worker in self.workers:
            if RAY_AVAILABLE:
                patterns = ray.get(worker.get_hot_patterns.remote(50))
            else:
                patterns = worker.get_hot_patterns(50)
            all_patterns.extend(patterns)

        # Distribute to all workers
        for worker in self.workers:
            if RAY_AVAILABLE:
                worker.receive_patterns.remote(all_patterns)
            else:
                worker.receive_patterns(all_patterns)

        logger.info(f"Synced {len(all_patterns)} patterns across {len(self.workers)} workers")
