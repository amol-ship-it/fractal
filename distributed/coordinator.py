"""
Cluster Coordinator

The central brain of the distributed system that:
- Manages worker lifecycle
- Distributes tasks intelligently
- Handles auto-scaling
- Coordinates pattern synchronization
- Monitors cluster health
- Handles fault tolerance and recovery
"""

import asyncio
import time
import uuid
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging
import json

from .config import ClusterConfig, NodeRole, ScalingStrategy
from .worker import WorkerPool, ProcessingTask, ProcessingResult, TaskType
from .storage import DistributedPatternStore, DistributedStateStore

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Status of a cluster node"""
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    DRAINING = "draining"


@dataclass
class NodeInfo:
    """Information about a cluster node"""
    node_id: str
    role: NodeRole
    address: str
    port: int
    status: NodeStatus = NodeStatus.STARTING
    last_heartbeat: float = field(default_factory=time.time)
    tasks_processed: int = 0
    current_load: float = 0.0
    patterns_stored: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusterMetrics:
    """Aggregated cluster metrics"""
    total_nodes: int = 0
    healthy_nodes: int = 0
    total_workers: int = 0
    tasks_in_queue: int = 0
    tasks_processed: int = 0
    patterns_discovered: int = 0
    avg_latency_ms: float = 0.0
    throughput_per_sec: float = 0.0
    memory_used_gb: float = 0.0
    cpu_utilization: float = 0.0


class LoadBalancer:
    """
    Intelligent load balancing for task distribution.
    Uses adaptive algorithms based on worker performance.
    """

    def __init__(self):
        self.worker_scores: Dict[str, float] = {}
        self.worker_latencies: Dict[str, deque] = {}
        self.task_affinities: Dict[str, str] = {}  # task_type -> preferred_worker

    def update_worker_score(self, worker_id: str, latency: float, success: bool):
        """Update worker score based on performance"""
        if worker_id not in self.worker_latencies:
            self.worker_latencies[worker_id] = deque(maxlen=100)

        self.worker_latencies[worker_id].append(latency)

        # Calculate score: lower latency and higher success = higher score
        avg_latency = sum(self.worker_latencies[worker_id]) / len(self.worker_latencies[worker_id])
        score = (1.0 / (avg_latency + 0.1)) * (1.0 if success else 0.5)
        self.worker_scores[worker_id] = score

    def select_worker(self, workers: List[str], task_type: TaskType = None) -> str:
        """Select best worker for a task"""
        if not workers:
            raise ValueError("No workers available")

        # Check for task affinity
        if task_type and task_type.value in self.task_affinities:
            preferred = self.task_affinities[task_type.value]
            if preferred in workers:
                return preferred

        # Score-based selection with randomization
        scored_workers = [
            (w, self.worker_scores.get(w, 1.0))
            for w in workers
        ]

        # Weighted random selection (better workers more likely)
        total_score = sum(s for _, s in scored_workers)
        if total_score == 0:
            return workers[0]

        import random
        r = random.uniform(0, total_score)
        cumulative = 0
        for worker, score in scored_workers:
            cumulative += score
            if cumulative >= r:
                return worker

        return workers[-1]


class AutoScaler:
    """
    Automatic scaling based on load and queue depth.
    Implements multiple scaling strategies.
    """

    def __init__(self, config: ClusterConfig):
        self.config = config
        self.scaling = config.scaling
        self.last_scale_up = 0
        self.last_scale_down = 0

    def should_scale_up(self, metrics: ClusterMetrics) -> int:
        """Determine if we should add workers"""
        now = time.time()

        # Cooldown check
        if now - self.last_scale_up < self.scaling.scale_up_cooldown:
            return 0

        workers_to_add = 0

        if self.scaling.strategy == ScalingStrategy.THRESHOLD:
            if metrics.cpu_utilization > self.scaling.scale_up_threshold:
                workers_to_add = max(1, metrics.total_workers // 4)

        elif self.scaling.strategy == ScalingStrategy.QUEUE_BASED:
            target_workers = metrics.tasks_in_queue // self.scaling.queue_depth_per_worker
            if target_workers > metrics.total_workers:
                workers_to_add = min(
                    target_workers - metrics.total_workers,
                    self.scaling.max_workers - metrics.total_workers
                )

        elif self.scaling.strategy == ScalingStrategy.PREDICTIVE:
            # Simple prediction based on throughput trend
            if metrics.throughput_per_sec > 0:
                predicted_queue = metrics.tasks_in_queue + (metrics.throughput_per_sec * 60)
                if predicted_queue > metrics.total_workers * 100:
                    workers_to_add = 2

        if workers_to_add > 0:
            self.last_scale_up = now
            # Respect max limit
            workers_to_add = min(
                workers_to_add,
                self.scaling.max_workers - metrics.total_workers
            )

        return workers_to_add

    def should_scale_down(self, metrics: ClusterMetrics) -> int:
        """Determine if we should remove workers"""
        now = time.time()

        # Cooldown check
        if now - self.last_scale_down < self.scaling.scale_down_cooldown:
            return 0

        if metrics.total_workers <= self.scaling.min_workers:
            return 0

        workers_to_remove = 0

        if self.scaling.strategy == ScalingStrategy.THRESHOLD:
            if metrics.cpu_utilization < self.scaling.scale_down_threshold:
                workers_to_remove = max(1, metrics.total_workers // 8)

        elif self.scaling.strategy == ScalingStrategy.QUEUE_BASED:
            target_workers = max(
                self.scaling.min_workers,
                metrics.tasks_in_queue // self.scaling.queue_depth_per_worker
            )
            if target_workers < metrics.total_workers:
                workers_to_remove = min(
                    metrics.total_workers - target_workers,
                    metrics.total_workers - self.scaling.min_workers
                )

        if workers_to_remove > 0:
            self.last_scale_down = now
            # Respect min limit
            workers_to_remove = min(
                workers_to_remove,
                metrics.total_workers - self.scaling.min_workers
            )

        return workers_to_remove


class ClusterCoordinator:
    """
    Main coordinator for the distributed cluster.

    Responsibilities:
    - Node registration and health monitoring
    - Task distribution and load balancing
    - Auto-scaling decisions
    - Pattern synchronization across nodes
    - Fault detection and recovery
    """

    def __init__(self, config: ClusterConfig):
        self.config = config
        self.coordinator_id = f"coord_{uuid.uuid4().hex[:8]}"

        # Node registry
        self.nodes: Dict[str, NodeInfo] = {}

        # Worker pool
        self.worker_pool: Optional[WorkerPool] = None

        # Load balancer and auto-scaler
        self.load_balancer = LoadBalancer()
        self.auto_scaler = AutoScaler(config)

        # Task management
        self.pending_tasks: asyncio.Queue = asyncio.Queue()
        self.active_tasks: Dict[str, ProcessingTask] = {}
        self.completed_tasks: deque = deque(maxlen=10000)

        # Metrics
        self.metrics = ClusterMetrics()
        self.latency_samples: deque = deque(maxlen=1000)

        # Background tasks
        self._running = False
        self._background_tasks: List[asyncio.Task] = []

        logger.info(f"Coordinator {self.coordinator_id} initialized")

    async def start(self):
        """Start the coordinator"""
        self._running = True

        # Initialize worker pool
        self.worker_pool = WorkerPool(
            num_workers=self.config.scaling.min_workers,
            config={
                'edge_threshold': self.config.edge_threshold,
                'cluster_threshold': self.config.cluster_threshold
            }
        )
        await self.worker_pool.start()

        # Start background tasks
        self._background_tasks = [
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._task_processor_loop()),
            asyncio.create_task(self._metrics_collector_loop()),
            asyncio.create_task(self._auto_scale_loop()),
            asyncio.create_task(self._sync_loop()),
        ]

        logger.info(f"Coordinator started with {self.config.scaling.min_workers} workers")

    async def stop(self):
        """Stop the coordinator"""
        self._running = False

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        # Stop worker pool
        if self.worker_pool:
            await self.worker_pool.stop()

        logger.info("Coordinator stopped")

    async def register_node(self, node_info: NodeInfo):
        """Register a new node in the cluster"""
        self.nodes[node_info.node_id] = node_info

        # Add to storage ring if storage node
        if node_info.role in [NodeRole.STORAGE, NodeRole.HYBRID]:
            # Would update distributed storage here
            pass

        logger.info(f"Registered node {node_info.node_id} as {node_info.role.value}")

    async def unregister_node(self, node_id: str):
        """Remove a node from the cluster"""
        if node_id in self.nodes:
            node = self.nodes.pop(node_id)
            logger.info(f"Unregistered node {node_id}")

            # Trigger rebalancing if needed
            await self._rebalance_after_node_removal(node)

    async def submit_task(self, task: ProcessingTask) -> str:
        """Submit a task for processing"""
        await self.pending_tasks.put(task)
        self.active_tasks[task.task_id] = task
        return task.task_id

    async def get_task_result(self, task_id: str,
                             timeout: float = None) -> Optional[ProcessingResult]:
        """Get result for a task (blocking with timeout)"""
        start = time.time()
        while True:
            # Check completed tasks
            for result in self.completed_tasks:
                if result.task_id == task_id:
                    return result

            # Check timeout
            if timeout and (time.time() - start) > timeout:
                return None

            await asyncio.sleep(0.1)

    async def _task_processor_loop(self):
        """Background loop that processes tasks from the queue"""
        while self._running:
            try:
                # Get task from queue with timeout
                try:
                    task = await asyncio.wait_for(
                        self.pending_tasks.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Process task
                start_time = time.time()
                result = await self.worker_pool.process_task(task)
                latency = time.time() - start_time

                # Update metrics
                self.latency_samples.append(latency)
                self.metrics.tasks_processed += 1

                # Update load balancer
                self.load_balancer.update_worker_score(
                    result.worker_id,
                    latency,
                    result.success
                )

                # Store result
                self.completed_tasks.append(result)
                self.active_tasks.pop(task.task_id, None)

                # Track patterns discovered
                self.metrics.patterns_discovered += len(result.patterns)

            except Exception as e:
                logger.error(f"Task processor error: {e}")
                await asyncio.sleep(1)

    async def _health_check_loop(self):
        """Background loop that checks node health"""
        while self._running:
            try:
                now = time.time()
                failed_nodes = []

                for node_id, node in self.nodes.items():
                    # Check heartbeat timeout (30 seconds)
                    if now - node.last_heartbeat > 30:
                        node.status = NodeStatus.FAILED
                        failed_nodes.append(node_id)

                # Handle failed nodes
                for node_id in failed_nodes:
                    await self._handle_node_failure(node_id)

                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(5)

    async def _metrics_collector_loop(self):
        """Background loop that collects cluster metrics"""
        while self._running:
            try:
                # Get worker pool stats
                if self.worker_pool:
                    pool_stats = await self.worker_pool.get_pool_stats()

                    self.metrics.total_workers = pool_stats['num_workers']
                    self.metrics.tasks_in_queue = self.pending_tasks.qsize()

                    # Calculate throughput
                    if self.latency_samples:
                        avg_latency = sum(self.latency_samples) / len(self.latency_samples)
                        self.metrics.avg_latency_ms = avg_latency * 1000
                        self.metrics.throughput_per_sec = (
                            self.metrics.total_workers / avg_latency
                            if avg_latency > 0 else 0
                        )

                # Node metrics
                self.metrics.total_nodes = len(self.nodes)
                self.metrics.healthy_nodes = sum(
                    1 for n in self.nodes.values()
                    if n.status == NodeStatus.HEALTHY
                )

                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Metrics collector error: {e}")
                await asyncio.sleep(5)

    async def _auto_scale_loop(self):
        """Background loop that handles auto-scaling"""
        while self._running:
            try:
                if self.config.scaling.strategy != ScalingStrategy.MANUAL:
                    # Check if we should scale up
                    workers_to_add = self.auto_scaler.should_scale_up(self.metrics)
                    if workers_to_add > 0:
                        await self._scale_up(workers_to_add)

                    # Check if we should scale down
                    workers_to_remove = self.auto_scaler.should_scale_down(self.metrics)
                    if workers_to_remove > 0:
                        await self._scale_down(workers_to_remove)

                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Auto-scale error: {e}")
                await asyncio.sleep(10)

    async def _sync_loop(self):
        """Background loop that syncs patterns across workers"""
        while self._running:
            try:
                if self.worker_pool:
                    await self.worker_pool.sync_patterns()

                await asyncio.sleep(self.config.sync_interval)

            except Exception as e:
                logger.error(f"Sync error: {e}")
                await asyncio.sleep(self.config.sync_interval)

    async def _scale_up(self, count: int):
        """Add workers to the cluster"""
        logger.info(f"Scaling up: adding {count} workers")

        # In a real implementation, this would spawn new Ray actors
        # For now, we track the intent
        self.metrics.total_workers += count

    async def _scale_down(self, count: int):
        """Remove workers from the cluster"""
        logger.info(f"Scaling down: removing {count} workers")

        # In a real implementation, this would gracefully drain and remove workers
        self.metrics.total_workers = max(
            self.config.scaling.min_workers,
            self.metrics.total_workers - count
        )

    async def _handle_node_failure(self, node_id: str):
        """Handle a failed node"""
        logger.warning(f"Handling failure of node {node_id}")

        node = self.nodes.get(node_id)
        if not node:
            return

        # Mark as failed
        node.status = NodeStatus.FAILED

        # If storage node, trigger replication
        if node.role in [NodeRole.STORAGE, NodeRole.HYBRID]:
            await self._recover_patterns_from_node(node_id)

        # Remove from active nodes
        await self.unregister_node(node_id)

    async def _recover_patterns_from_node(self, node_id: str):
        """Recover patterns that were stored on a failed node"""
        logger.info(f"Recovering patterns from failed node {node_id}")
        # In production, would re-replicate patterns from surviving replicas

    async def _rebalance_after_node_removal(self, node: NodeInfo):
        """Rebalance work after a node is removed"""
        logger.info(f"Rebalancing after removal of {node.node_id}")
        # In production, would redistribute patterns and tasks

    def receive_heartbeat(self, node_id: str, load: float = 0.0):
        """Receive a heartbeat from a node"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.last_heartbeat = time.time()
            node.current_load = load
            if node.status != NodeStatus.HEALTHY:
                node.status = NodeStatus.HEALTHY

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status"""
        return {
            'coordinator_id': self.coordinator_id,
            'cluster_name': self.config.cluster_name,
            'metrics': {
                'total_nodes': self.metrics.total_nodes,
                'healthy_nodes': self.metrics.healthy_nodes,
                'total_workers': self.metrics.total_workers,
                'tasks_in_queue': self.metrics.tasks_in_queue,
                'tasks_processed': self.metrics.tasks_processed,
                'patterns_discovered': self.metrics.patterns_discovered,
                'avg_latency_ms': self.metrics.avg_latency_ms,
                'throughput_per_sec': self.metrics.throughput_per_sec
            },
            'nodes': [
                {
                    'id': n.node_id,
                    'role': n.role.value,
                    'status': n.status.value,
                    'load': n.current_load
                }
                for n in self.nodes.values()
            ],
            'scaling': {
                'strategy': self.config.scaling.strategy.value,
                'min_workers': self.config.scaling.min_workers,
                'max_workers': self.config.scaling.max_workers
            }
        }
