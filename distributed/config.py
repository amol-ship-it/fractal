"""
Cluster Configuration - Settings for distributed deployment

Supports multiple deployment modes:
- Single machine (multi-process)
- Multi-machine cluster
- Cloud auto-scaling (AWS, GCP, Azure)
- Kubernetes deployment
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import os
import json


class NodeRole(Enum):
    """Roles a node can have in the cluster"""
    COORDINATOR = "coordinator"      # Manages cluster, distributes work
    WORKER = "worker"                # Processes patterns
    STORAGE = "storage"              # Dedicated storage node
    GATEWAY = "gateway"              # API gateway for external requests
    HYBRID = "hybrid"                # All roles (for small deployments)


class ScalingStrategy(Enum):
    """How the cluster scales"""
    MANUAL = "manual"                # Fixed number of nodes
    THRESHOLD = "threshold"          # Scale at CPU/memory thresholds
    PREDICTIVE = "predictive"        # ML-based prediction of load
    QUEUE_BASED = "queue_based"      # Scale based on task queue depth


@dataclass
class RedisConfig:
    """Redis cluster configuration"""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    cluster_mode: bool = False
    cluster_nodes: List[Dict[str, Any]] = field(default_factory=list)
    max_connections: int = 100
    socket_timeout: float = 5.0
    retry_on_timeout: bool = True

    # Sharding configuration
    num_shards: int = 16
    replication_factor: int = 2

    @classmethod
    def from_env(cls) -> 'RedisConfig':
        """Load from environment variables"""
        return cls(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            password=os.getenv('REDIS_PASSWORD'),
            cluster_mode=os.getenv('REDIS_CLUSTER', 'false').lower() == 'true'
        )


@dataclass
class RayConfig:
    """Ray distributed computing configuration"""
    address: Optional[str] = None  # None = start local cluster
    num_cpus: Optional[int] = None  # None = auto-detect
    num_gpus: Optional[int] = None
    object_store_memory: Optional[int] = None  # Bytes
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8265

    # Worker configuration
    num_workers_per_node: int = 4
    max_task_retries: int = 3
    task_timeout: float = 300.0  # seconds

    @classmethod
    def from_env(cls) -> 'RayConfig':
        """Load from environment variables"""
        return cls(
            address=os.getenv('RAY_ADDRESS'),
            num_cpus=int(os.getenv('RAY_NUM_CPUS')) if os.getenv('RAY_NUM_CPUS') else None,
            num_gpus=int(os.getenv('RAY_NUM_GPUS')) if os.getenv('RAY_NUM_GPUS') else None
        )


@dataclass
class ScalingConfig:
    """Auto-scaling configuration"""
    strategy: ScalingStrategy = ScalingStrategy.THRESHOLD
    min_workers: int = 2
    max_workers: int = 100
    scale_up_threshold: float = 0.8  # CPU utilization
    scale_down_threshold: float = 0.3
    scale_up_cooldown: int = 60  # seconds
    scale_down_cooldown: int = 300
    queue_depth_per_worker: int = 100  # For queue-based scaling


@dataclass
class NetworkConfig:
    """Network configuration for inter-node communication"""
    bind_address: str = "0.0.0.0"
    coordinator_port: int = 8000
    worker_port_range: tuple = (8001, 8100)
    grpc_port: int = 50051
    use_tls: bool = False
    tls_cert_path: Optional[str] = None
    tls_key_path: Optional[str] = None


@dataclass
class ClusterConfig:
    """Master configuration for the distributed cluster"""
    # Cluster identity
    cluster_name: str = "recursive-learning-cluster"
    cluster_id: str = field(default_factory=lambda: os.urandom(8).hex())

    # Component configs
    redis: RedisConfig = field(default_factory=RedisConfig)
    ray: RayConfig = field(default_factory=RayConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)

    # Pattern engine settings
    edge_threshold: float = 0.1
    cluster_threshold: float = 0.7
    learning_rate: float = 0.1
    exploration_rate: float = 0.2

    # Distributed settings
    pattern_replication: int = 2  # How many nodes store each pattern
    batch_size: int = 100  # Patterns processed per batch
    sync_interval: float = 1.0  # Seconds between sync operations

    # Resource limits
    max_patterns_per_node: int = 100000
    max_memory_per_node: int = 4 * 1024 * 1024 * 1024  # 4GB
    max_queue_size: int = 10000

    @classmethod
    def for_single_machine(cls, num_workers: int = None) -> 'ClusterConfig':
        """Configuration for single-machine multi-process deployment"""
        import multiprocessing
        num_workers = num_workers or multiprocessing.cpu_count()

        return cls(
            cluster_name="local-cluster",
            ray=RayConfig(num_workers_per_node=num_workers),
            scaling=ScalingConfig(
                strategy=ScalingStrategy.MANUAL,
                min_workers=num_workers,
                max_workers=num_workers
            )
        )

    @classmethod
    def for_multi_machine(cls, coordinator_host: str,
                         redis_hosts: List[str],
                         num_workers: int = 10) -> 'ClusterConfig':
        """Configuration for multi-machine cluster"""
        return cls(
            cluster_name="distributed-cluster",
            redis=RedisConfig(
                cluster_mode=True,
                cluster_nodes=[{'host': h, 'port': 6379} for h in redis_hosts]
            ),
            ray=RayConfig(
                address=f"{coordinator_host}:6379",
                num_workers_per_node=4
            ),
            scaling=ScalingConfig(
                strategy=ScalingStrategy.THRESHOLD,
                min_workers=num_workers,
                max_workers=num_workers * 5
            )
        )

    @classmethod
    def for_kubernetes(cls) -> 'ClusterConfig':
        """Configuration for Kubernetes deployment"""
        return cls(
            cluster_name=os.getenv('CLUSTER_NAME', 'k8s-rl-cluster'),
            redis=RedisConfig.from_env(),
            ray=RayConfig.from_env(),
            scaling=ScalingConfig(
                strategy=ScalingStrategy.QUEUE_BASED,
                min_workers=int(os.getenv('MIN_WORKERS', 2)),
                max_workers=int(os.getenv('MAX_WORKERS', 50))
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration"""
        return {
            'cluster_name': self.cluster_name,
            'cluster_id': self.cluster_id,
            'redis': {
                'host': self.redis.host,
                'port': self.redis.port,
                'cluster_mode': self.redis.cluster_mode
            },
            'scaling': {
                'strategy': self.scaling.strategy.value,
                'min_workers': self.scaling.min_workers,
                'max_workers': self.scaling.max_workers
            },
            'engine': {
                'edge_threshold': self.edge_threshold,
                'cluster_threshold': self.cluster_threshold,
                'learning_rate': self.learning_rate,
                'exploration_rate': self.exploration_rate
            }
        }

    def save(self, filepath: str):
        """Save configuration to file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'ClusterConfig':
        """Load configuration from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        # Simplified loading - in production would fully deserialize
        config = cls(
            cluster_name=data.get('cluster_name', 'cluster'),
            cluster_id=data.get('cluster_id', os.urandom(8).hex())
        )
        return config
