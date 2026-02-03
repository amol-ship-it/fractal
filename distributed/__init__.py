"""
Distributed Recursive Learning AI

A scalable, multi-machine architecture implementing the Swarm Intelligence principle:
"Building independent entities that self-organize to create global coherence"

Architecture:
- Distributed Pattern Storage (Redis Cluster)
- Parallel Processing Workers (Ray Actors)
- Event-Driven Coordination (Pub/Sub)
- Auto-Scaling based on load
- Fault-Tolerant with automatic recovery
"""

from .config import ClusterConfig, NodeRole
from .storage import DistributedPatternStore, DistributedStateStore
from .worker import PatternWorker, ProcessingTask
from .coordinator import ClusterCoordinator
from .node import DistributedNode
from .cluster import RecursiveLearningCluster

__all__ = [
    'ClusterConfig',
    'NodeRole',
    'DistributedPatternStore',
    'DistributedStateStore',
    'PatternWorker',
    'ProcessingTask',
    'ClusterCoordinator',
    'DistributedNode',
    'RecursiveLearningCluster'
]
