"""
Shared utility functions for training scripts.
"""

import csv
import json
import os
from pathlib import Path


def create_experiment_name(num_gpus, zero_stage):
    """
    Create consistent experiment name.
    
    Args:
        num_gpus: Number of GPUs used
        zero_stage: DeepSpeed ZeRO stage (0 for baseline, 2, or 3)
    
    Returns:
        str: Experiment name
        
    Examples:
        >>> create_experiment_name(1, 0)
        'baseline'
        >>> create_experiment_name(2, 2)
        'zero2_2gpu'
        >>> create_experiment_name(4, 3)
        'zero3_4gpu'
    """
    if zero_stage == 0:
        return "baseline"
    else:
        return f"zero{zero_stage}_{num_gpus}gpu"


def get_zero_stage_from_config(config_path):
    """
    Extract ZeRO stage from DeepSpeed config file.
    
    Args:
        config_path: Path to DeepSpeed JSON config
        
    Returns:
        int: ZeRO stage (2 or 3)
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config["zero_optimization"]["stage"]


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
    
    print("="*70)