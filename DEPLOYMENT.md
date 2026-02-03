# Recursive Learning AI - Deployment Guide

This guide covers deploying the Distributed Recursive Learning AI system at various scales, from single-machine development to production Kubernetes clusters.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLUSTER ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│    ┌──────────────┐                                                      │
│    │   Gateway    │  ←── External API requests                          │
│    │   (API)      │                                                      │
│    └──────┬───────┘                                                      │
│           │                                                              │
│           ▼                                                              │
│    ┌──────────────┐         ┌─────────────────────────────────┐         │
│    │  Coordinator │ ◄──────►│        Redis Cluster            │         │
│    │              │         │   (Distributed Pattern Store)   │         │
│    │  • Task Dist │         │   • Patterns (Sharded)          │         │
│    │  • Load Bal  │         │   • State (Replicated)          │         │
│    │  • AutoScale │         │   • Pub/Sub (Events)            │         │
│    └──────┬───────┘         └─────────────────────────────────┘         │
│           │                              ▲                               │
│           │ Task Assignment              │ Pattern Storage               │
│           ▼                              │                               │
│    ┌──────────────────────────────────────────────────────────┐         │
│    │                    WORKER POOL                           │         │
│    │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │         │
│    │  │ Worker  │ │ Worker  │ │ Worker  │ │ Worker  │  ...   │         │
│    │  │  (Ray)  │ │  (Ray)  │ │  (Ray)  │ │  (Ray)  │        │         │
│    │  │         │ │         │ │         │ │         │        │         │
│    │  │ Pattern │ │ Pattern │ │ Pattern │ │ Pattern │        │         │
│    │  │ Engine  │ │ Engine  │ │ Engine  │ │ Engine  │        │         │
│    │  └─────────┘ └─────────┘ └─────────┘ └─────────┘        │         │
│    └──────────────────────────────────────────────────────────┘         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Deployment Options

### 1. Single Machine (Development)

Best for development and testing. Uses all available CPU cores.

```bash
# Install dependencies
pip install -r requirements.txt

# Start with auto-detected workers
python cli.py start

# Or specify worker count
python cli.py start --workers 8

# Run demo
python cli.py demo

# Run benchmark
python cli.py benchmark --signals 1000 --workers 8
```

**Python API:**
```python
from distributed.cluster import RecursiveLearningCluster

# Create local cluster
cluster = RecursiveLearningCluster.create_local(num_workers=8)
await cluster.start()

# Process data
result = await cluster.perceive([0.1, 0.3, 0.5, 0.7], domain="audio")
print(f"Discovered {len(result['patterns'])} patterns")

# Batch processing (parallel)
results = await cluster.batch_perceive(signals, domain="audio")

await cluster.stop()
```

### 2. Multi-Machine Cluster

For teams or larger workloads. Requires Redis for coordination.

**Prerequisites:**
- Redis server (or cluster)
- Network connectivity between machines
- Same Python environment on all nodes

**On the coordinator machine:**
```bash
# Start Redis (if not running)
redis-server

# Start coordinator
python cli.py start --role coordinator --redis localhost
```

**On worker machines:**
```bash
# Join cluster
python cli.py start --role worker \
    --coordinator 192.168.1.100 \
    --redis 192.168.1.100
```

**Python API:**
```python
cluster = RecursiveLearningCluster.create_cluster(
    coordinator_host="192.168.1.100",
    redis_hosts=["192.168.1.101"],
    num_workers=20
)
await cluster.start()
```

### 3. Docker Compose

For consistent environments and easy scaling on a single machine or small cluster.

```bash
# Build and start all services
docker-compose up -d

# Scale workers
docker-compose up -d --scale worker=4

# View logs
docker-compose logs -f coordinator

# Stop
docker-compose down
```

**Services:**
- `redis`: Pattern storage (port 6379)
- `coordinator`: Cluster brain (port 8000, 8265)
- `worker`: Pattern processing (scalable)
- `gateway`: API endpoint (port 8080)

### 4. Kubernetes (Production)

For production workloads with auto-scaling.

```bash
# Create namespace and deploy
kubectl apply -f kubernetes/deployment.yaml

# Check status
kubectl get pods -n recursive-learning-ai

# View coordinator logs
kubectl logs -f deployment/coordinator -n recursive-learning-ai

# Scale workers manually
kubectl scale deployment/worker --replicas=10 -n recursive-learning-ai

# Access API (get external IP)
kubectl get svc api-gateway -n recursive-learning-ai
```

**Components:**
- **Redis StatefulSet**: 3-node cluster for high availability
- **Coordinator Deployment**: Single instance (leader election possible)
- **Worker Deployment**: Auto-scaled by HPA based on CPU/memory
- **HorizontalPodAutoscaler**: Scales 2-20 workers based on load

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NODE_ROLE` | `hybrid` | Node role: coordinator, worker, storage, hybrid |
| `REDIS_HOST` | `localhost` | Redis server hostname |
| `REDIS_PORT` | `6379` | Redis server port |
| `REDIS_CLUSTER` | `false` | Enable Redis cluster mode |
| `COORDINATOR_HOST` | `localhost` | Coordinator address (for workers) |
| `NUM_WORKERS` | `4` | Workers per node |
| `MIN_WORKERS` | `2` | Minimum workers (auto-scaling) |
| `MAX_WORKERS` | `50` | Maximum workers (auto-scaling) |

### Configuration File

Create `config.json`:
```json
{
  "cluster_name": "my-cluster",
  "scaling": {
    "strategy": "queue_based",
    "min_workers": 2,
    "max_workers": 50,
    "queue_depth_per_worker": 100
  },
  "engine": {
    "edge_threshold": 0.1,
    "cluster_threshold": 0.7,
    "learning_rate": 0.1,
    "exploration_rate": 0.2
  }
}
```

## Scaling Guidelines

### Horizontal Scaling (More Workers)

| Workload | Recommended Workers | Notes |
|----------|---------------------|-------|
| Development | 2-4 | Single machine |
| Small team | 8-16 | 2-4 machines |
| Production | 50-100 | Kubernetes with HPA |
| Large scale | 100-500 | Multi-region deployment |

### Vertical Scaling (Bigger Workers)

Each worker benefits from:
- **Memory**: More local pattern cache (faster lookups)
- **CPU**: Faster pattern processing

Recommended per worker:
- **Minimum**: 2 CPU, 2GB RAM
- **Optimal**: 4 CPU, 8GB RAM
- **High Performance**: 8 CPU, 16GB RAM

### Storage Scaling

Redis cluster nodes based on pattern count:
- < 1M patterns: Single Redis node
- 1M-10M patterns: 3-node Redis cluster
- 10M-100M patterns: 6+ node Redis cluster with sharding

## Monitoring

### Built-in Metrics

```bash
# Get cluster status
python cli.py status

# API endpoint (if gateway running)
curl http://localhost:8000/metrics
```

### Key Metrics

- `tasks_processed`: Total tasks completed
- `patterns_discovered`: Patterns found
- `avg_latency_ms`: Processing latency
- `throughput_per_sec`: Tasks per second
- `queue_depth`: Pending tasks
- `worker_count`: Active workers

### Prometheus Integration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'recursive-learning-ai'
    static_configs:
      - targets: ['coordinator:8000']
```

## High Availability

### Coordinator HA

For production, run multiple coordinators with leader election:

```yaml
# Coordinator with leader election
apiVersion: apps/v1
kind: Deployment
metadata:
  name: coordinator
spec:
  replicas: 3  # Multiple replicas
  ...
```

### Storage HA

Use Redis Sentinel or Redis Cluster:

```bash
# Redis Sentinel
redis-sentinel /path/to/sentinel.conf
```

### Worker Recovery

Workers are stateless - failed workers are automatically replaced by Kubernetes HPA.

## Performance Tuning

### Batch Size

Larger batches = higher throughput, higher latency
```python
# High throughput
results = await cluster.batch_perceive(signals, batch_size=1000)

# Low latency
result = await cluster.perceive(signal)
```

### Pattern Synchronization

```python
# More frequent sync = better consistency, more overhead
config.sync_interval = 0.5  # 500ms

# Less frequent sync = better performance, eventual consistency
config.sync_interval = 5.0  # 5 seconds
```

### Cache Tuning

```python
# Larger cache = more memory, faster lookups
config.max_local_patterns = 50000

# Smaller cache = less memory, more Redis calls
config.max_local_patterns = 1000
```

## Troubleshooting

### Common Issues

**Workers not connecting:**
```bash
# Check coordinator is running
curl http://coordinator:8000/health

# Check Redis connectivity
redis-cli -h redis ping
```

**High latency:**
- Increase workers
- Check Redis latency: `redis-cli --latency`
- Enable local caching

**Out of memory:**
- Reduce `max_local_patterns`
- Add more Redis nodes
- Enable pattern archival

### Debug Mode

```bash
# Verbose logging
python cli.py start --verbose

# Or set environment
export LOG_LEVEL=DEBUG
```

## Security

### Redis Authentication

```bash
# Set Redis password
redis-server --requirepass your_password

# Configure in app
export REDIS_PASSWORD=your_password
```

### Network Security

- Use VPC/private networks for inter-node communication
- Enable TLS for Redis connections
- Use Kubernetes NetworkPolicies

```yaml
# NetworkPolicy example
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: worker-policy
spec:
  podSelector:
    matchLabels:
      app: worker
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: coordinator
```
