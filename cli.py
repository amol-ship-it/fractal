#!/usr/bin/env python3
"""
Recursive Learning AI - Command Line Interface

Manage and interact with the distributed cluster.

Usage:
    python cli.py start --workers 8
    python cli.py status
    python cli.py process input.json
    python cli.py scale --workers 16
"""

import argparse
import asyncio
import json
import sys
import math
from typing import List

from distributed.config import ClusterConfig, NodeRole, ScalingStrategy
from distributed.cluster import RecursiveLearningCluster


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='recursive-learning-ai',
        description='Distributed Recursive Learning AI System'
    )
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Start command
    start_parser = subparsers.add_parser('start', help='Start the cluster')
    start_parser.add_argument('--workers', '-w', type=int, default=None,
                             help='Number of workers (default: auto-detect CPUs)')
    start_parser.add_argument('--role', '-r', choices=['coordinator', 'worker', 'storage', 'hybrid'],
                             default='hybrid', help='Node role')
    start_parser.add_argument('--coordinator', '-c', type=str, default=None,
                             help='Coordinator address (for joining existing cluster)')
    start_parser.add_argument('--redis', type=str, nargs='+', default=None,
                             help='Redis host(s) for distributed storage')

    # Status command
    subparsers.add_parser('status', help='Show cluster status')

    # Process command
    process_parser = subparsers.add_parser('process', help='Process input data')
    process_parser.add_argument('input', type=str,
                               help='Input file (JSON) or inline data')
    process_parser.add_argument('--domain', '-d', type=str, default='general',
                               help='Processing domain')
    process_parser.add_argument('--batch', '-b', action='store_true',
                               help='Process as batch')

    # Query command
    query_parser = subparsers.add_parser('query', help='Query for similar patterns')
    query_parser.add_argument('query', type=str,
                             help='Query data (JSON array)')
    query_parser.add_argument('--domain', '-d', type=str, default=None,
                             help='Filter by domain')
    query_parser.add_argument('--threshold', '-t', type=float, default=0.7,
                             help='Similarity threshold')
    query_parser.add_argument('--limit', '-l', type=int, default=10,
                             help='Max results')

    # Scale command
    scale_parser = subparsers.add_parser('scale', help='Scale the cluster')
    scale_parser.add_argument('--workers', '-w', type=int, required=True,
                             help='Target number of workers')

    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run performance benchmark')
    bench_parser.add_argument('--signals', '-s', type=int, default=100,
                             help='Number of signals to process')
    bench_parser.add_argument('--signal-length', '-l', type=int, default=100,
                             help='Length of each signal')
    bench_parser.add_argument('--workers', '-w', type=int, default=None,
                             help='Number of workers')

    # Demo command
    subparsers.add_parser('demo', help='Run interactive demo')

    return parser


async def cmd_start(args):
    """Start the cluster"""
    print("Starting Recursive Learning AI Cluster...")

    if args.coordinator:
        # Join existing cluster
        config = ClusterConfig.for_multi_machine(
            coordinator_host=args.coordinator,
            redis_hosts=args.redis or ['localhost'],
            num_workers=args.workers or 4
        )
        role = NodeRole(args.role)
    else:
        # Start new local cluster
        config = ClusterConfig.for_single_machine(args.workers)
        role = NodeRole.HYBRID

    cluster = RecursiveLearningCluster(config)
    await cluster.start(role)

    print(f"\nCluster started successfully!")
    print(f"  Workers: {args.workers or 'auto'}")
    print(f"  Role: {role.value}")

    status = cluster.get_status()
    print(f"  Node ID: {status.get('node_id', 'unknown')}")

    # Keep running
    print("\nCluster is running. Press Ctrl+C to stop.")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping cluster...")
        await cluster.stop()


async def cmd_status(args):
    """Show cluster status"""
    # In a real implementation, this would query the running cluster
    print("Cluster Status")
    print("=" * 50)
    print("Note: Connect to running cluster for live status")
    print("\nTo view status, start a cluster first with 'start' command")


async def cmd_process(args):
    """Process input data"""
    # Parse input
    try:
        if args.input.endswith('.json'):
            with open(args.input, 'r') as f:
                data = json.load(f)
        else:
            data = json.loads(args.input)
    except Exception as e:
        print(f"Error parsing input: {e}")
        return 1

    # Create cluster
    cluster = RecursiveLearningCluster.create_local(num_workers=4)
    await cluster.start()

    try:
        if args.batch and isinstance(data, list) and isinstance(data[0], list):
            results = await cluster.batch_perceive(data, domain=args.domain)
            print(f"Processed {len(results)} inputs")
            print(json.dumps(results, indent=2))
        else:
            if not isinstance(data, list):
                data = [data]
            result = await cluster.perceive(data, domain=args.domain)
            print(json.dumps(result, indent=2))
    finally:
        await cluster.stop()


async def cmd_query(args):
    """Query for similar patterns"""
    query_data = json.loads(args.query)

    cluster = RecursiveLearningCluster.create_local(num_workers=2)
    await cluster.start()

    try:
        result = await cluster.query(
            query_data,
            domain=args.domain,
            threshold=args.threshold,
            limit=args.limit
        )
        print(json.dumps(result, indent=2))
    finally:
        await cluster.stop()


async def cmd_scale(args):
    """Scale the cluster"""
    print(f"Scaling cluster to {args.workers} workers...")
    # In production, this would send a scale command to the coordinator
    print("Note: Connect to running cluster to scale")


async def cmd_benchmark(args):
    """Run performance benchmark"""
    import time

    print("=" * 60)
    print("RECURSIVE LEARNING AI - Performance Benchmark")
    print("=" * 60)

    num_signals = args.signals
    signal_length = args.signal_length
    num_workers = args.workers

    # Generate test data
    print(f"\nGenerating {num_signals} signals of length {signal_length}...")
    signals = [
        [math.sin(x * (0.1 + i * 0.01)) + math.random() * 0.1
         if hasattr(math, 'random') else math.sin(x * (0.1 + i * 0.01))
         for x in range(signal_length)]
        for i in range(num_signals)
    ]

    # Actually use random for noise
    import random
    signals = [
        [math.sin(x * (0.1 + i * 0.01)) + random.gauss(0, 0.1)
         for x in range(signal_length)]
        for i in range(num_signals)
    ]

    # Create cluster
    cluster = RecursiveLearningCluster.create_local(num_workers=num_workers)
    await cluster.start()

    try:
        # Warmup
        print("Warming up...")
        await cluster.perceive(signals[0], domain="benchmark")

        # Sequential benchmark
        print(f"\nSequential processing ({num_signals} signals)...")
        start = time.time()
        for signal in signals[:min(10, num_signals)]:  # Limit for sequential
            await cluster.perceive(signal, domain="benchmark")
        sequential_time = time.time() - start
        sequential_throughput = min(10, num_signals) / sequential_time

        # Batch benchmark
        print(f"Batch processing ({num_signals} signals)...")
        start = time.time()
        results = await cluster.batch_perceive(signals, domain="benchmark")
        batch_time = time.time() - start
        batch_throughput = num_signals / batch_time

        # Collect results
        total_patterns = sum(len(r.get('patterns', [])) for r in results)
        success_rate = sum(1 for r in results if r.get('success', False)) / len(results)

        # Print results
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Workers: {num_workers or 'auto'}")
        print(f"Signals: {num_signals}")
        print(f"Signal length: {signal_length}")
        print("-" * 60)
        print(f"Sequential throughput: {sequential_throughput:.1f} signals/sec")
        print(f"Batch throughput: {batch_throughput:.1f} signals/sec")
        print(f"Speedup: {batch_throughput / sequential_throughput:.1f}x")
        print("-" * 60)
        print(f"Total patterns discovered: {total_patterns}")
        print(f"Patterns per signal: {total_patterns / num_signals:.1f}")
        print(f"Success rate: {success_rate * 100:.1f}%")
        print(f"Total time (batch): {batch_time:.2f}s")

    finally:
        await cluster.stop()


async def cmd_demo(args):
    """Run interactive demo"""
    from distributed.cluster import demo_distributed
    await demo_distributed()


def main():
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Map commands to functions
    commands = {
        'start': cmd_start,
        'status': cmd_status,
        'process': cmd_process,
        'query': cmd_query,
        'scale': cmd_scale,
        'benchmark': cmd_benchmark,
        'demo': cmd_demo,
    }

    if args.command in commands:
        return asyncio.run(commands[args.command](args))
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main() or 0)
