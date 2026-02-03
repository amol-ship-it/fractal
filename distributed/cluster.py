"""
Recursive Learning Cluster - High-Level API

The main entry point for using the distributed Recursive Learning AI.
Provides a simple interface that hides the complexity of distribution.
"""

import asyncio
import math
from typing import Dict, List, Any, Optional
import logging

from .config import ClusterConfig, NodeRole, ScalingStrategy
from .node import DistributedNode
from .worker import ProcessingTask, TaskType, ProcessingResult

logger = logging.getLogger(__name__)


class RecursiveLearningCluster:
    """
    High-level interface for the Distributed Recursive Learning AI.

    Usage:
        # Single machine (auto-detects CPUs)
        cluster = RecursiveLearningCluster.create_local()
        await cluster.start()

        # Process data
        result = await cluster.perceive([0.1, 0.3, 0.5, 0.7])

        # Multi-machine cluster
        cluster = RecursiveLearningCluster.create_cluster(
            coordinator_host="192.168.1.100",
            redis_hosts=["192.168.1.101", "192.168.1.102"]
        )
    """

    def __init__(self, config: ClusterConfig):
        self.config = config
        self.node: Optional[DistributedNode] = None
        self._started = False

    @classmethod
    def create_local(cls, num_workers: int = None) -> 'RecursiveLearningCluster':
        """Create a cluster that runs on a single machine"""
        config = ClusterConfig.for_single_machine(num_workers)
        return cls(config)

    @classmethod
    def create_cluster(cls, coordinator_host: str,
                      redis_hosts: List[str],
                      num_workers: int = 10) -> 'RecursiveLearningCluster':
        """Create a distributed cluster across multiple machines"""
        config = ClusterConfig.for_multi_machine(
            coordinator_host,
            redis_hosts,
            num_workers
        )
        return cls(config)

    @classmethod
    def create_kubernetes(cls) -> 'RecursiveLearningCluster':
        """Create a cluster configured for Kubernetes"""
        config = ClusterConfig.for_kubernetes()
        return cls(config)

    async def start(self, role: NodeRole = NodeRole.HYBRID):
        """Start the cluster/node"""
        if self._started:
            return

        self.node = DistributedNode(self.config, role)
        await self.node.start()
        self._started = True

        logger.info(f"Cluster started: {self.config.cluster_name}")

    async def stop(self):
        """Stop the cluster/node"""
        if self.node:
            await self.node.stop()
        self._started = False

        logger.info("Cluster stopped")

    async def perceive(self, input_data: List[float],
                      domain: str = "general") -> Dict[str, Any]:
        """
        Process new input through the distributed system.

        This is the main entry point - equivalent to the single-machine
        perceive() but distributed across all available workers.
        """
        if not self._started:
            raise RuntimeError("Cluster not started. Call start() first.")

        return await self.node.process(input_data, domain)

    async def batch_perceive(self, inputs: List[List[float]],
                            domain: str = "general") -> List[Dict[str, Any]]:
        """
        Process multiple inputs in parallel.

        The work is automatically distributed across all workers.
        """
        if not self._started:
            raise RuntimeError("Cluster not started. Call start() first.")

        return await self.node.batch_process(inputs, domain)

    async def query(self, query_data: List[float],
                   domain: str = None,
                   threshold: float = 0.7,
                   limit: int = 10) -> Dict[str, Any]:
        """Query for similar patterns across the cluster"""
        if not self._started:
            raise RuntimeError("Cluster not started. Call start() first.")

        return await self.node.query(query_data, domain, threshold, limit)

    async def learn_sequence(self, sequence: List[List[float]],
                            domain: str = "general") -> Dict[str, Any]:
        """
        Learn temporal patterns from a sequence.
        Processes the sequence and builds connections between patterns.
        """
        results = await self.batch_perceive(sequence, domain)

        # Build temporal connections
        temporal_connections = []
        for i in range(len(results) - 1):
            if results[i]['success'] and results[i + 1]['success']:
                for p1 in results[i]['patterns']:
                    for p2 in results[i + 1]['patterns']:
                        temporal_connections.append({
                            'from': p1['id'],
                            'to': p2['id'],
                            'position': i
                        })

        return {
            'total_patterns': sum(len(r['patterns']) for r in results),
            'temporal_connections': len(temporal_connections),
            'success_rate': sum(1 for r in results if r['success']) / len(results)
        }

    async def transfer_learning(self, source_domain: str,
                               target_domain: str,
                               examples: List[tuple]) -> Dict[str, Any]:
        """
        Transfer patterns from one domain to another.

        examples: List of (source_input, target_input) tuples
        """
        transferred = 0

        for source_input, target_input in examples:
            # Process both inputs
            source_result = await self.perceive(source_input, source_domain)
            target_result = await self.perceive(target_input, target_domain)

            # Bridge patterns would be created automatically in storage
            if source_result['success'] and target_result['success']:
                transferred += min(
                    len(source_result['patterns']),
                    len(target_result['patterns'])
                )

        return {
            'source_domain': source_domain,
            'target_domain': target_domain,
            'patterns_transferred': transferred,
            'examples_processed': len(examples)
        }

    def get_status(self) -> Dict[str, Any]:
        """Get cluster status and metrics"""
        if not self._started or not self.node:
            return {'started': False}

        return {
            'started': True,
            **self.node.get_status()
        }

    async def scale(self, num_workers: int):
        """Manually scale the cluster to a specific number of workers"""
        # This would interact with the auto-scaler to set the target
        logger.info(f"Scaling cluster to {num_workers} workers")
        # In production, this would trigger actual scaling


async def demo_distributed():
    """Demonstrate the distributed cluster"""
    print("=" * 70)
    print("DISTRIBUTED RECURSIVE LEARNING AI - Demo")
    print("Scaling intelligence across multiple machines")
    print("=" * 70)

    # Create a local cluster (simulates distributed on single machine)
    cluster = RecursiveLearningCluster.create_local(num_workers=4)

    try:
        # Start the cluster
        print("\n1. STARTING CLUSTER")
        print("-" * 50)
        await cluster.start()

        status = cluster.get_status()
        print(f"Cluster: {status.get('node_id', 'unknown')}")
        print(f"Role: {status.get('role', 'unknown')}")

        # Process some data
        print("\n2. PROCESSING DATA")
        print("-" * 50)

        # Generate test signals
        signals = [
            [math.sin(x * 0.3) for x in range(30)],
            [math.sin(x * 0.3 + 0.1) for x in range(30)],
            [math.cos(x * 0.2) for x in range(30)],
        ]

        for i, signal in enumerate(signals):
            result = await cluster.perceive(signal, domain="audio")
            print(f"Signal {i + 1}: {result.get('success', False)} - "
                  f"Discovered {len(result.get('patterns', []))} patterns")

        # Batch processing
        print("\n3. BATCH PROCESSING (Parallel)")
        print("-" * 50)

        batch_signals = [
            [math.sin(x * (0.1 + i * 0.05)) for x in range(30)]
            for i in range(10)
        ]

        import time
        start = time.time()
        results = await cluster.batch_perceive(batch_signals, domain="audio")
        elapsed = time.time() - start

        total_patterns = sum(len(r.get('patterns', [])) for r in results)
        print(f"Processed {len(batch_signals)} signals in {elapsed:.3f}s")
        print(f"Discovered {total_patterns} patterns total")
        print(f"Throughput: {len(batch_signals) / elapsed:.1f} signals/sec")

        # Sequence learning
        print("\n4. SEQUENCE LEARNING (Temporal Patterns)")
        print("-" * 50)

        sequence = [
            [math.sin(x * 0.2 + t * 0.5) for x in range(20)]
            for t in range(5)
        ]

        seq_result = await cluster.learn_sequence(sequence, domain="temporal")
        print(f"Total patterns: {seq_result['total_patterns']}")
        print(f"Temporal connections: {seq_result['temporal_connections']}")

        # Query
        print("\n5. QUERYING SIMILAR PATTERNS")
        print("-" * 50)

        query_result = await cluster.query(
            [math.sin(x * 0.3) for x in range(30)],
            domain="audio",
            threshold=0.5
        )
        print(f"Found {len(query_result.get('matches', []))} similar patterns")

        # Final status
        print("\n" + "=" * 70)
        print("CLUSTER STATUS")
        print("=" * 70)

        final_status = cluster.get_status()
        if 'cluster' in final_status:
            metrics = final_status['cluster'].get('metrics', {})
            print(f"Tasks Processed: {metrics.get('tasks_processed', 0)}")
            print(f"Patterns Discovered: {metrics.get('patterns_discovered', 0)}")
            print(f"Avg Latency: {metrics.get('avg_latency_ms', 0):.2f}ms")
            print(f"Throughput: {metrics.get('throughput_per_sec', 0):.1f}/sec")

    finally:
        # Stop the cluster
        await cluster.stop()
        print("\nCluster stopped.")


def main():
    """Run the demo"""
    asyncio.run(demo_distributed())


if __name__ == "__main__":
    main()
