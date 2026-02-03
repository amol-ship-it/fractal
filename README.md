# Recursive Learning AI

A self-improving, distributed intelligence system that learns by discovering patterns—not following rules.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the demo
python cli.py demo

# 3. Run a benchmark
python cli.py benchmark --signals 100 --workers 4
```

---

## Installation

### Requirements
- Python 3.10+
- Redis (optional, for distributed mode)

### Steps

```bash
# Extract the zip file
unzip recursive_learning_ai.zip
cd recursive_learning_ai

# Install dependencies
pip install -r requirements.txt

# Verify installation
python cli.py demo
```

---

## Usage

### Command Line

```bash
# Run interactive demo
python cli.py demo

# Start cluster with 8 workers
python cli.py start --workers 8

# Run performance benchmark
python cli.py benchmark --signals 1000 --workers 4

# Process data from JSON file
python cli.py process '[0.1, 0.3, 0.5, 0.7]' --domain audio
```

### Python API

```python
import asyncio
import math
from distributed.cluster import RecursiveLearningCluster

async def main():
    # Create local cluster
    cluster = RecursiveLearningCluster.create_local(num_workers=4)
    await cluster.start()

    # Process a signal
    signal = [math.sin(x * 0.3) for x in range(50)]
    result = await cluster.perceive(signal, domain="audio")

    print(f"Patterns discovered: {len(result['patterns'])}")
    print(f"Predictions: {result['predictions']}")

    # Batch process (parallel)
    signals = [[math.sin(x * (0.1 + i*0.05)) for x in range(50)] for i in range(10)]
    results = await cluster.batch_perceive(signals)

    # Query for similar patterns
    matches = await cluster.query([0.1, 0.2, 0.3], threshold=0.6)

    await cluster.stop()

asyncio.run(main())
```

### Standalone (Non-Distributed)

```python
from system import RecursiveLearningAI

ai = RecursiveLearningAI({'learning_rate': 0.1, 'exploration_rate': 0.2})

# Process input
result = ai.perceive([0.1, 0.3, 0.5, 0.7], domain="sensor")

# Provide feedback
ai.receive_feedback(result['episode_id'], feedback_value=0.8)

# Query
matches = ai.query([0.2, 0.4, 0.6])

# Get stats
print(ai.introspect())
```

---

## Deployment Options

### 1. Single Machine
```bash
python cli.py start --workers 8
```

### 2. Docker Compose
```bash
docker-compose up -d
docker-compose up -d --scale worker=8  # Scale workers
```

### 3. Kubernetes
```bash
kubectl apply -f kubernetes/deployment.yaml
```

### 4. Multi-Machine Cluster
```bash
# Start Redis
redis-server

# Machine 1: Coordinator
python cli.py start --role coordinator --redis localhost

# Machine 2+: Workers
python cli.py start --role worker --coordinator 192.168.1.100
```

---

## Architecture

```
INPUT → EDGE DETECTION → CLUSTERING → PATTERN DISCOVERY → COMPOSITION → PREDICTIONS
        (subtraction)    (division)   (similarity match)   (hierarchy)   (forecast)
```

### The Four Pillars

| Pillar | What It Does |
|--------|--------------|
| **Feedback Loops** | Learn from outcomes (rewards/penalties) |
| **Approximability** | Refine patterns over time |
| **Composability** | Combine patterns into hierarchies |
| **Exploration** | Discover novel pattern combinations |

### Dual Memory

- **Pattern Memory**: Compressed abstractions (the "code")
- **State Memory**: Context and instances (the "data")

---

## File Structure

```
recursive_learning_ai/
├── core/                    # Core AI engine
│   ├── pattern.py          # Pattern class
│   ├── memory.py           # Dual memory system
│   ├── engine.py           # Processing pipeline
│   └── feedback.py         # Feedback & exploration
├── distributed/            # Distributed system
│   ├── cluster.py          # High-level API
│   ├── coordinator.py      # Cluster brain
│   ├── worker.py           # Processing workers
│   ├── storage.py          # Distributed storage
│   └── config.py           # Configuration
├── kubernetes/             # K8s deployment
├── cli.py                  # Command line tool
├── system.py               # Standalone system
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_WORKERS` | 4 | Workers per node |
| `REDIS_HOST` | localhost | Redis server |
| `MIN_WORKERS` | 2 | Min workers (auto-scale) |
| `MAX_WORKERS` | 50 | Max workers (auto-scale) |

### Python Config

```python
from distributed.config import ClusterConfig

config = ClusterConfig(
    edge_threshold=0.1,      # Edge detection sensitivity
    cluster_threshold=0.7,   # Clustering similarity
    learning_rate=0.1,       # Pattern update rate
    exploration_rate=0.2,    # Exploration probability
)
```

---

## Examples

### Process Audio
```python
import math
signal = [math.sin(x * 0.3) + math.sin(x * 0.7) * 0.5 for x in range(100)]
result = await cluster.perceive(signal, domain="audio")
```

### Learn Time Series
```python
prices = [get_stock_price(t) for t in range(1000)]
await cluster.learn_sequence(prices, domain="finance")
```

### Transfer Learning
```python
await cluster.transfer_learning(
    source_domain="english",
    target_domain="french",
    examples=[(en_text, fr_text), ...]
)
```

---

## Performance

| Setup | Workers | Throughput |
|-------|---------|------------|
| Local (4 CPU) | 4 | ~100/sec |
| Local (8 CPU) | 8 | ~200/sec |
| Cluster | 20 | ~500/sec |
| Kubernetes | 100 | ~2500/sec |

---

## Troubleshooting

**"No module named 'ray'"**
```bash
pip install "ray[default]"
```

**Redis connection error**
```bash
# Start Redis
redis-server
# Or use Docker
docker run -d -p 6379:6379 redis
```

**Import errors**
```bash
# Make sure you're in the right directory
cd recursive_learning_ai
python -c "from distributed.cluster import RecursiveLearningCluster; print('OK')"
```

---

## Research Foundation

Based on:
- **General Learning Naturalism**: Intelligence emerges from pattern discovery
- **Perceptual Magnet Effect**: Brain warps perception through experience
- **Hierarchical Aggregation**: "Pixels to eternity"
- **Swarm Intelligence**: Independent entities self-organize

> *"Intelligence is the search for the biggest possible reward by constantly predicting what's next."*

---

## License

MIT License - Free for research and commercial use.
