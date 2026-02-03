"""
Distributed Storage Layer

Implements distributed pattern and state storage using:
- Redis for fast key-value storage and pub/sub
- Consistent hashing for pattern distribution
- Replication for fault tolerance
- Local caching for hot patterns
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import OrderedDict
import logging

# Import base pattern classes
import sys
sys.path.insert(0, '..')
from core.pattern import Pattern, PatternType

logger = logging.getLogger(__name__)


class ConsistentHash:
    """
    Consistent hashing for distributing patterns across nodes.
    Minimizes redistribution when nodes are added/removed.
    """

    def __init__(self, nodes: List[str] = None, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []
        self.nodes: Set[str] = set()

        if nodes:
            for node in nodes:
                self.add_node(node)

    def _hash(self, key: str) -> int:
        """Generate hash for a key"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_node(self, node: str):
        """Add a node to the ring"""
        self.nodes.add(node)
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}:{i}"
            hash_val = self._hash(virtual_key)
            self.ring[hash_val] = node
        self.sorted_keys = sorted(self.ring.keys())

    def remove_node(self, node: str):
        """Remove a node from the ring"""
        self.nodes.discard(node)
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}:{i}"
            hash_val = self._hash(virtual_key)
            self.ring.pop(hash_val, None)
        self.sorted_keys = sorted(self.ring.keys())

    def get_node(self, key: str) -> Optional[str]:
        """Get the node responsible for a key"""
        if not self.ring:
            return None

        hash_val = self._hash(key)

        # Binary search for the first node with hash >= key hash
        for node_hash in self.sorted_keys:
            if node_hash >= hash_val:
                return self.ring[node_hash]

        # Wrap around to first node
        return self.ring[self.sorted_keys[0]]

    def get_nodes(self, key: str, count: int = 1) -> List[str]:
        """Get multiple nodes for replication"""
        if not self.ring:
            return []

        nodes = []
        hash_val = self._hash(key)

        # Find starting position
        start_idx = 0
        for i, node_hash in enumerate(self.sorted_keys):
            if node_hash >= hash_val:
                start_idx = i
                break

        # Collect unique nodes
        seen = set()
        idx = start_idx
        while len(nodes) < count and len(seen) < len(self.nodes):
            node = self.ring[self.sorted_keys[idx % len(self.sorted_keys)]]
            if node not in seen:
                seen.add(node)
                nodes.append(node)
            idx += 1

        return nodes


class LocalCache:
    """LRU cache for frequently accessed patterns"""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        if key in self.cache:
            self.hits += 1
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key][0]
        self.misses += 1
        return None

    def set(self, key: str, value: Any, ttl: float = 300.0):
        """Set item in cache with TTL"""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = (value, time.time() + ttl)

        # Evict if over capacity
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def invalidate(self, key: str):
        """Remove item from cache"""
        self.cache.pop(key, None)

    def clear_expired(self):
        """Remove expired items"""
        now = time.time()
        expired = [k for k, (_, exp) in self.cache.items() if exp < now]
        for k in expired:
            del self.cache[k]

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class DistributedPatternStore:
    """
    Distributed pattern storage with:
    - Consistent hashing for distribution
    - Replication for fault tolerance
    - Local caching for performance
    - Async operations for concurrency
    """

    def __init__(self, node_id: str, redis_client=None,
                 replication_factor: int = 2, cache_size: int = 10000):
        self.node_id = node_id
        self.redis = redis_client
        self.replication_factor = replication_factor

        # Consistent hash ring for pattern distribution
        self.hash_ring = ConsistentHash()

        # Local cache
        self.cache = LocalCache(cache_size)

        # Local storage fallback (when Redis unavailable)
        self.local_patterns: Dict[str, Pattern] = {}

        # Indices for fast lookup
        self.domain_index: Dict[str, Set[str]] = {}
        self.type_index: Dict[PatternType, Set[str]] = {}

        # Statistics
        self.stats = {
            'stores': 0,
            'retrieves': 0,
            'cache_hits': 0,
            'remote_fetches': 0
        }

    def add_node(self, node_id: str):
        """Add a storage node to the cluster"""
        self.hash_ring.add_node(node_id)
        logger.info(f"Added storage node: {node_id}")

    def remove_node(self, node_id: str):
        """Remove a storage node from the cluster"""
        self.hash_ring.remove_node(node_id)
        logger.info(f"Removed storage node: {node_id}")

    async def store(self, pattern: Pattern) -> bool:
        """
        Store a pattern in the distributed store.
        Replicates to multiple nodes for fault tolerance.
        """
        self.stats['stores'] += 1

        # Serialize pattern
        pattern_data = json.dumps(pattern.to_dict())

        # Determine which nodes should store this pattern
        target_nodes = self.hash_ring.get_nodes(pattern.id, self.replication_factor)

        # If this node is a target, store locally
        if self.node_id in target_nodes or not target_nodes:
            self.local_patterns[pattern.id] = pattern
            self._update_indices(pattern)

        # Store in Redis (distributed)
        if self.redis:
            try:
                # Use Redis pipeline for efficiency
                pipe = self.redis.pipeline()

                # Store pattern data
                pipe.set(f"pattern:{pattern.id}", pattern_data)

                # Update domain index
                for domain in pattern.domains:
                    pipe.sadd(f"domain:{domain}", pattern.id)

                # Update type index
                pipe.sadd(f"type:{pattern.pattern_type.value}", pattern.id)

                # Set expiration for cache invalidation tracking
                pipe.expire(f"pattern:{pattern.id}", 86400 * 7)  # 7 days

                await asyncio.to_thread(pipe.execute)

            except Exception as e:
                logger.error(f"Redis store failed: {e}")
                # Fallback to local storage already done above

        # Update local cache
        self.cache.set(pattern.id, pattern)

        return True

    async def retrieve(self, pattern_id: str) -> Optional[Pattern]:
        """Retrieve a pattern by ID"""
        self.stats['retrieves'] += 1

        # Check local cache first
        cached = self.cache.get(pattern_id)
        if cached:
            self.stats['cache_hits'] += 1
            return cached

        # Check local storage
        if pattern_id in self.local_patterns:
            pattern = self.local_patterns[pattern_id]
            self.cache.set(pattern_id, pattern)
            return pattern

        # Fetch from Redis
        if self.redis:
            try:
                self.stats['remote_fetches'] += 1
                data = await asyncio.to_thread(
                    self.redis.get, f"pattern:{pattern_id}"
                )
                if data:
                    pattern = Pattern.from_dict(json.loads(data))
                    self.cache.set(pattern_id, pattern)
                    return pattern
            except Exception as e:
                logger.error(f"Redis retrieve failed: {e}")

        return None

    async def find_similar(self, query_pattern: Pattern,
                          threshold: float = 0.7,
                          limit: int = 10,
                          domain: str = None) -> List[Tuple[Pattern, float]]:
        """
        Find similar patterns across the distributed store.
        This is a scatter-gather operation across all nodes.
        """
        results = []

        # Get candidate pattern IDs
        if domain:
            # Filter by domain
            if self.redis:
                try:
                    pattern_ids = await asyncio.to_thread(
                        self.redis.smembers, f"domain:{domain}"
                    )
                    pattern_ids = [pid.decode() if isinstance(pid, bytes) else pid
                                  for pid in pattern_ids]
                except Exception:
                    pattern_ids = list(self.domain_index.get(domain, set()))
            else:
                pattern_ids = list(self.domain_index.get(domain, set()))
        else:
            # Get all patterns (limited)
            if self.redis:
                try:
                    pattern_ids = []
                    cursor = 0
                    while True:
                        cursor, keys = await asyncio.to_thread(
                            self.redis.scan, cursor, match="pattern:*", count=1000
                        )
                        pattern_ids.extend([
                            k.decode().replace("pattern:", "") if isinstance(k, bytes)
                            else k.replace("pattern:", "")
                            for k in keys
                        ])
                        if cursor == 0 or len(pattern_ids) >= 10000:
                            break
                except Exception:
                    pattern_ids = list(self.local_patterns.keys())
            else:
                pattern_ids = list(self.local_patterns.keys())

        # Compute similarities in parallel batches
        batch_size = 100
        for i in range(0, len(pattern_ids), batch_size):
            batch_ids = pattern_ids[i:i + batch_size]

            # Fetch patterns in batch
            patterns = await asyncio.gather(*[
                self.retrieve(pid) for pid in batch_ids
            ])

            # Compute similarities
            for pattern in patterns:
                if pattern and pattern.id != query_pattern.id:
                    similarity = query_pattern.similarity(pattern)
                    if similarity >= threshold:
                        results.append((pattern, similarity))

        # Sort by similarity and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    async def get_by_domain(self, domain: str) -> List[Pattern]:
        """Get all patterns for a domain"""
        pattern_ids = []

        if self.redis:
            try:
                ids = await asyncio.to_thread(
                    self.redis.smembers, f"domain:{domain}"
                )
                pattern_ids = [pid.decode() if isinstance(pid, bytes) else pid
                              for pid in ids]
            except Exception:
                pattern_ids = list(self.domain_index.get(domain, set()))
        else:
            pattern_ids = list(self.domain_index.get(domain, set()))

        patterns = await asyncio.gather(*[
            self.retrieve(pid) for pid in pattern_ids
        ])

        return [p for p in patterns if p is not None]

    def _update_indices(self, pattern: Pattern):
        """Update local indices"""
        # Domain index
        for domain in pattern.domains:
            if domain not in self.domain_index:
                self.domain_index[domain] = set()
            self.domain_index[domain].add(pattern.id)

        # Type index
        if pattern.pattern_type not in self.type_index:
            self.type_index[pattern.pattern_type] = set()
        self.type_index[pattern.pattern_type].add(pattern.id)

    async def sync_with_cluster(self):
        """Synchronize local patterns with cluster"""
        if not self.redis:
            return

        # Get patterns this node should own
        owned_patterns = set()
        for pattern_id in self.local_patterns:
            target_nodes = self.hash_ring.get_nodes(pattern_id, self.replication_factor)
            if self.node_id in target_nodes:
                owned_patterns.add(pattern_id)

        # Store owned patterns to Redis
        for pattern_id in owned_patterns:
            pattern = self.local_patterns[pattern_id]
            await self.store(pattern)

        logger.info(f"Synced {len(owned_patterns)} patterns to cluster")

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        return {
            **self.stats,
            'local_patterns': len(self.local_patterns),
            'cache_size': len(self.cache.cache),
            'cache_hit_rate': self.cache.hit_rate,
            'nodes_in_ring': len(self.hash_ring.nodes)
        }


class DistributedStateStore:
    """
    Distributed state storage for context and instance data.
    Uses Redis for distributed access with local caching.
    """

    def __init__(self, node_id: str, redis_client=None,
                 max_local_entries: int = 1000, decay_rate: float = 0.95):
        self.node_id = node_id
        self.redis = redis_client
        self.max_local_entries = max_local_entries
        self.decay_rate = decay_rate

        # Local state storage
        self.local_state: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    async def store(self, key: str, data: Any,
                   linked_patterns: List[str] = None, ttl: int = 3600):
        """Store state data"""
        entry = {
            'data': data,
            'timestamp': time.time(),
            'relevance': 1.0,
            'linked_patterns': linked_patterns or [],
            'node_id': self.node_id
        }

        # Store locally
        self.local_state[key] = entry
        if len(self.local_state) > self.max_local_entries:
            self.local_state.popitem(last=False)

        # Store in Redis
        if self.redis:
            try:
                await asyncio.to_thread(
                    self.redis.setex,
                    f"state:{key}",
                    ttl,
                    json.dumps(entry, default=str)
                )

                # Update pattern-to-state index
                for pid in (linked_patterns or []):
                    await asyncio.to_thread(
                        self.redis.sadd, f"state_links:{pid}", key
                    )
            except Exception as e:
                logger.error(f"Redis state store failed: {e}")

    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve state data"""
        # Check local first
        if key in self.local_state:
            entry = self.local_state[key]
            entry['relevance'] = min(1.0, entry['relevance'] + 0.1)
            return entry['data']

        # Check Redis
        if self.redis:
            try:
                data = await asyncio.to_thread(
                    self.redis.get, f"state:{key}"
                )
                if data:
                    entry = json.loads(data)
                    self.local_state[key] = entry
                    return entry['data']
            except Exception as e:
                logger.error(f"Redis state retrieve failed: {e}")

        return None

    async def get_context(self, pattern_ids: List[str],
                         limit: int = 10) -> List[Tuple[str, Any]]:
        """Get state entries linked to patterns"""
        results = []

        for pid in pattern_ids:
            if self.redis:
                try:
                    state_keys = await asyncio.to_thread(
                        self.redis.smembers, f"state_links:{pid}"
                    )
                    for key in state_keys:
                        key = key.decode() if isinstance(key, bytes) else key
                        data = await self.retrieve(key)
                        if data:
                            results.append((key, data))
                except Exception:
                    pass

        return results[:limit]

    async def decay_all(self):
        """Apply temporal decay to all state entries"""
        for entry in self.local_state.values():
            entry['relevance'] *= self.decay_rate
