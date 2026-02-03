"""
Distributed Node

A single node in the distributed cluster that can play multiple roles:
- Coordinator: Manages the cluster
- Worker: Processes patterns
- Storage: Stores patterns
- Gateway: API entry point
- Hybrid: All of the above (for small deployments)
"""

import asyncio
import time
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
import socket

from .config import ClusterConfig, NodeRole
from .coordinator import ClusterCoordinator, NodeInfo, NodeStatus
from .worker import WorkerPool, ProcessingTask, TaskType
from .storage import DistributedPatternStore, DistributedStateStore

logger = logging.getLogger(__name__)


class DistributedNode:
    """
    A single node in the distributed Recursive Learning AI cluster.

    Each node can serve multiple roles and automatically connects
    to the cluster, registers itself, and begins processing.
    """

    def __init__(self, config: ClusterConfig, role: NodeRole = NodeRole.HYBRID,
                 node_id: str = None):
        self.config = config
        self.role = role
        self.node_id = node_id or f"node_{uuid.uuid4().hex[:8]}"

        # Determine this node's address
        self.address = self._get_local_address()
        self.port = config.network.coordinator_port

        # Components (initialized based on role)
        self.coordinator: Optional[ClusterCoordinator] = None
        self.worker_pool: Optional[WorkerPool] = None
        self.pattern_store: Optional[DistributedPatternStore] = None
        self.state_store: Optional[DistributedStateStore] = None

        # Redis client (would be initialized with actual Redis in production)
        self.redis_client = None

        # State
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None

        logger.info(f"Node {self.node_id} created with role {role.value}")

    def _get_local_address(self) -> str:
        """Get this machine's IP address"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            addr = s.getsockname()[0]
            s.close()
            return addr
        except Exception:
            return "127.0.0.1"

    async def start(self):
        """Start the node"""
        logger.info(f"Starting node {self.node_id} as {self.role.value}")
        self._running = True

        # Initialize components based on role
        if self.role in [NodeRole.COORDINATOR, NodeRole.HYBRID]:
            self.coordinator = ClusterCoordinator(self.config)
            await self.coordinator.start()

        if self.role in [NodeRole.WORKER, NodeRole.HYBRID]:
            self.worker_pool = WorkerPool(
                num_workers=self.config.ray.num_workers_per_node,
                config={
                    'edge_threshold': self.config.edge_threshold,
                    'cluster_threshold': self.config.cluster_threshold
                }
            )
            await self.worker_pool.start()

        if self.role in [NodeRole.STORAGE, NodeRole.HYBRID]:
            self.pattern_store = DistributedPatternStore(
                node_id=self.node_id,
                redis_client=self.redis_client,
                replication_factor=self.config.pattern_replication
            )
            self.state_store = DistributedStateStore(
                node_id=self.node_id,
                redis_client=self.redis_client
            )

        # Start heartbeat
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        # Register with coordinator (if not self)
        if self.coordinator:
            node_info = NodeInfo(
                node_id=self.node_id,
                role=self.role,
                address=self.address,
                port=self.port,
                status=NodeStatus.HEALTHY
            )
            await self.coordinator.register_node(node_info)

        logger.info(f"Node {self.node_id} started successfully")

    async def stop(self):
        """Stop the node"""
        logger.info(f"Stopping node {self.node_id}")
        self._running = False

        # Cancel heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        # Stop components
        if self.coordinator:
            await self.coordinator.stop()

        if self.worker_pool:
            await self.worker_pool.stop()

        logger.info(f"Node {self.node_id} stopped")

    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self._running:
            try:
                if self.coordinator:
                    # Send heartbeat to self (coordinator tracks all nodes)
                    load = await self._calculate_load()
                    self.coordinator.receive_heartbeat(self.node_id, load)

                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(5)

    async def _calculate_load(self) -> float:
        """Calculate current node load (0.0 to 1.0)"""
        # Simple load calculation based on queue size
        if self.worker_pool:
            stats = await self.worker_pool.get_pool_stats()
            queue_size = stats.get('queue_size', 0)
            # Normalize: 100 tasks = 100% load
            return min(1.0, queue_size / 100)
        return 0.0

    async def process(self, input_data: List[float],
                     domain: str = "general") -> Dict[str, Any]:
        """
        Process input data through the distributed system.
        This is the main entry point for using the AI.
        """
        task = ProcessingTask.create(
            TaskType.PROCESS_INPUT,
            input_data,
            domain
        )

        if self.coordinator:
            # Submit to coordinator
            task_id = await self.coordinator.submit_task(task)
            result = await self.coordinator.get_task_result(task_id, timeout=30)

            if result:
                # Store discovered patterns
                if self.pattern_store:
                    for pdata in result.patterns:
                        from core.pattern import Pattern
                        pattern = Pattern.from_dict(pdata)
                        await self.pattern_store.store(pattern)

                return {
                    'success': result.success,
                    'patterns': result.patterns,
                    'predictions': result.predictions,
                    'metrics': result.metrics,
                    'processing_time': result.processing_time
                }

        elif self.worker_pool:
            # Process directly with worker pool
            result = await self.worker_pool.process_task(task)

            return {
                'success': result.success,
                'patterns': result.patterns,
                'predictions': result.predictions,
                'metrics': result.metrics,
                'processing_time': result.processing_time
            }

        return {'success': False, 'error': 'No processing capability available'}

    async def batch_process(self, inputs: List[List[float]],
                           domain: str = "general") -> List[Dict[str, Any]]:
        """Process multiple inputs in parallel"""
        tasks = [
            ProcessingTask.create(TaskType.PROCESS_INPUT, data, domain)
            for data in inputs
        ]

        if self.coordinator:
            # Submit all tasks
            task_ids = await asyncio.gather(*[
                self.coordinator.submit_task(task)
                for task in tasks
            ])

            # Wait for all results
            results = await asyncio.gather(*[
                self.coordinator.get_task_result(tid, timeout=60)
                for tid in task_ids
            ])

            return [
                {
                    'success': r.success if r else False,
                    'patterns': r.patterns if r else [],
                    'predictions': r.predictions if r else [],
                    'metrics': r.metrics if r else {}
                }
                for r in results
            ]

        elif self.worker_pool:
            results = await self.worker_pool.batch_process(tasks)

            return [
                {
                    'success': r.success,
                    'patterns': r.patterns,
                    'predictions': r.predictions,
                    'metrics': r.metrics
                }
                for r in results
            ]

        return [{'success': False, 'error': 'No processing capability'}]

    async def query(self, query_data: List[float],
                   domain: str = None,
                   threshold: float = 0.7,
                   limit: int = 10) -> Dict[str, Any]:
        """Query for similar patterns"""
        if self.pattern_store:
            from core.pattern import Pattern
            query_pattern = Pattern.create_atomic(query_data, domain or "query")

            matches = await self.pattern_store.find_similar(
                query_pattern,
                threshold=threshold,
                limit=limit,
                domain=domain
            )

            return {
                'success': True,
                'matches': [
                    {'pattern': p.to_dict(), 'similarity': s}
                    for p, s in matches
                ]
            }

        return {'success': False, 'error': 'No storage capability'}

    def get_status(self) -> Dict[str, Any]:
        """Get node status"""
        status = {
            'node_id': self.node_id,
            'role': self.role.value,
            'address': self.address,
            'port': self.port,
            'running': self._running
        }

        if self.coordinator:
            status['cluster'] = self.coordinator.get_cluster_status()

        if self.pattern_store:
            status['storage'] = self.pattern_store.get_stats()

        return status
