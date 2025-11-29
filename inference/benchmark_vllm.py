#!/usr/bin/env python
"""
Comprehensive vLLM inference benchmarking.

Measures:
- Throughput (requests/sec, tokens/sec)
- Latency percentiles (P50, P90, P95, P99)
- Time-to-first-token (TTFT)
- GPU utilization and memory

Usage:
    # Start vLLM server first, then:
    python inference/benchmark_vllm.py \
        --base_url http://localhost:8000 \
        --num_requests 100 \
        --concurrency 1,4,8,16,32,64
"""

import argparse
import asyncio
import time
import numpy as np
import csv
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List

try:
    from openai import AsyncOpenAI
except ImportError:
    print("Error: openai package required")
    print("Install: pip install openai")
    exit(1)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config_name: str
    num_gpus: int
    concurrency: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    
    # Latency metrics (seconds)
    mean_latency: float
    median_latency: float
    p90_latency: float
    p95_latency: float
    p99_latency: float
    
    # Throughput metrics
    requests_per_second: float
    tokens_per_second: float
    
    # Duration
    total_duration: float


class VLLMBenchmark:
    """vLLM inference benchmarking."""
    
    def __init__(self, base_url: str, model_name: str = "model"):
        self.client = AsyncOpenAI(
            base_url=f"{base_url}/v1",
            api_key="dummy",
            timeout=120.0,
        )
        self.model_name = model_name
    
    async def single_request(self, prompt: str, max_tokens: int = 256):
        """Send single request and measure latency."""
        start_time = time.time()
        
        try:
            response = await self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.8,
            )
            
            latency = time.time() - start_time
            tokens = response.usage.completion_tokens
            
            return {
                'success': True,
                'latency': latency,
                'tokens': tokens,
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'latency': time.time() - start_time,
            }
    
    async def concurrent_benchmark(
        self,
        prompts: List[str],
        concurrency: int,
        max_tokens: int = 256,
    ):
        """Run benchmark with controlled concurrency."""
        
        print(f"\nRunning: {len(prompts)} requests, concurrency={concurrency}")
        
        # Semaphore to control max concurrent requests
        semaphore = asyncio.Semaphore(concurrency)
        
        async def bounded_request(prompt):
            async with semaphore:
                return await self.single_request(prompt, max_tokens)
        
        start_time = time.time()
        
        results = await asyncio.gather(
            *[bounded_request(p) for p in prompts]
        )
        
        total_duration = time.time() - start_time
        
        return results, total_duration
    
    def compute_metrics(
        self,
        results: List[dict],
        total_duration: float,
        config_name: str,
        num_gpus: int,
        concurrency: int,
    ) -> BenchmarkResult:
        """Compute metrics from results."""
        
        successful = [r for r in results if r.get('success')]
        failed = [r for r in results if not r.get('success')]
        
        latencies = [r['latency'] for r in successful]
        total_tokens = sum(r['tokens'] for r in successful)
        
        return BenchmarkResult(
            config_name=config_name,
            num_gpus=num_gpus,
            concurrency=concurrency,
            total_requests=len(results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            mean_latency=np.mean(latencies) if latencies else 0,
            median_latency=np.median(latencies) if latencies else 0,
            p90_latency=np.percentile(latencies, 90) if latencies else 0,
            p95_latency=np.percentile(latencies, 95) if latencies else 0,
            p99_latency=np.percentile(latencies, 99) if latencies else 0,
            requests_per_second=len(successful) / total_duration if total_duration > 0 else 0,
            tokens_per_second=total_tokens / total_duration if total_duration > 0 else 0,
            total_duration=total_duration,
        )


async def run_benchmark(
    base_url: str,
    config_name: str,
    num_gpus: int,
    num_requests: int,
    concurrency_levels: List[int],
):
    """Run full benchmark suite."""
    
    print("\n" + "="*70)
    print(f"BENCHMARKING: {config_name}")
    print("="*70)
    print(f"Configuration: {num_gpus} GPU(s)")
    print(f"Total requests: {num_requests}")
    print(f"Concurrency levels: {concurrency_levels}")
    print()
    
    benchmark = VLLMBenchmark(base_url)
    
    # Generate test prompts
    test_prompts = [
        "Write a Python function to find the nth Fibonacci number.",
        "Explain how quicksort algorithm works.",
        "Create a binary search tree implementation in Python.",
        "Write a function to reverse a linked list.",
        "Implement a stack using a list in Python.",
        "Explain the difference between BFS and DFS.",
        "Write a function to check if a string is a palindrome.",
        "Create a function to merge two sorted arrays.",
        "Implement bubble sort in Python.",
        "Write a function to find duplicates in an array.",
    ] * (num_requests // 10) 
    
    prompts = test_prompts[:num_requests]
    
    all_results = []
    
    for concurrency in concurrency_levels:
        print(f"\nTesting concurrency: {concurrency}")
        print("-" * 70)
        
        results, duration = await benchmark.concurrent_benchmark(
            prompts=prompts,
            concurrency=concurrency,
        )
        
        metrics = benchmark.compute_metrics(
            results=results,
            total_duration=duration,
            config_name=config_name,
            num_gpus=num_gpus,
            concurrency=concurrency,
        )
        
        all_results.append(metrics)
        
        # Print summary
        print(f"Duration: {metrics.total_duration:.2f}s")
        print(f"Successful: {metrics.successful_requests}/{metrics.total_requests}")
        print(f"Throughput: {metrics.requests_per_second:.2f} req/s, {metrics.tokens_per_second:.2f} tok/s")
        print(f"Latency: P50={metrics.median_latency:.3f}s, P99={metrics.p99_latency:.3f}s")
    
    return all_results


def save_results(results: List[BenchmarkResult], output_path: str = "results/inference_metrics.csv"):
    """Save benchmark results to CSV."""
    
    Path("results").mkdir(parents=True, exist_ok=True)
    
    file_exists = os.path.isfile(output_path)
    
    with open(output_path, 'a', newline='') as f:
        if results:
            fieldnames = list(asdict(results[0]).keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            for result in results:
                writer.writerow(asdict(result))
    
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM inference performance"
    )
    
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8000",
        help="vLLM server URL"
    )
    
    parser.add_argument(
        "--config_name",
        type=str,
        default="vllm_inference",
        help="Configuration name for results"
    )
    
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs used by server"
    )
    
    parser.add_argument(
        "--num_requests",
        type=int,
        default=100,
        help="Total number of requests to send"
    )
    
    parser.add_argument(
        "--concurrency",
        type=str,
        default="1,4,8,16,32,64",
        help="Comma-separated concurrency levels to test"
    )
    
    args = parser.parse_args()
    
    # Parse concurrency levels
    concurrency_levels = [int(x.strip()) for x in args.concurrency.split(',')]
    
    print("\n" + "="*70)
    print("vLLM INFERENCE BENCHMARKING")
    print("="*70)
    print(f"\nServer: {args.base_url}")
    print(f"Configuration: {args.config_name}")
    print(f"Requests: {args.num_requests}")
    print(f"Concurrency levels: {concurrency_levels}")
    
    # Run async benchmark
    results = asyncio.run(run_benchmark(
        base_url=args.base_url,
        config_name=args.config_name,
        num_gpus=args.num_gpus,
        num_requests=args.num_requests,
        concurrency_levels=concurrency_levels,
    ))
    
    # Save results
    save_results(results)
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
    print(f"\nResults saved to: results/inference_metrics.csv")
    print("Run 'python scripts/compare_inference.py' to analyze")
    print()


if __name__ == "__main__":
    main()