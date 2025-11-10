"""
Shared utility functions for training scripts.
"""

import csv
import os
from pathlib import Path


def save_training_metrics(metrics, csv_path="results/training_metrics.csv"):
    """
    Save training metrics to CSV for comparison.
    
    Args:
        metrics: Dictionary with core metrics
        csv_path: Path to CSV file
    """
    Path("results").mkdir(parents=True, exist_ok=True)
    
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)
    
    print(f"\nMetrics saved to {csv_path}")


def print_metrics_summary(metrics):
    """Print formatted metrics summary."""
    
    print("\n" + "="*70)
    print("TRAINING METRICS")
    print("="*70)
    
    print(f"\nExperiment: {metrics['experiment']}")
    print(f"GPUs: {metrics['num_gpus']}")
    print(f"Strategy: {metrics['strategy']}")
    print()
    print(f"Training time: {metrics['training_time_hours']:.2f} hours")
    print(f"Throughput: {metrics['samples_per_second']:.1f} samples/sec")
    print(f"Memory/GPU: {metrics['peak_memory_gb']:.2f} GB")
    print(f"Final loss: {metrics['final_loss']:.4f}")
    print(f"Speedup: {metrics['speedup']:.2f}x")
    print(f"Efficiency: {metrics['efficiency_percent']:.1f}%")
    
    print("="*70)