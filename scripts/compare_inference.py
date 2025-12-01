#!/usr/bin/env python
"""
Compare inference results across configurations.

Analyzes results/inference_metrics.csv and generates comparison plots.

Usage:
    python scripts/compare_inference.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np


def load_results(csv_path="results/inference_metrics.csv"):
    """Load inference benchmark results."""
    
    if not os.path.exists(csv_path):
        print(f"Results not found: {csv_path}")
        print("\nRun benchmarks first:")
        print("1. Start vLLM server")
        print("2. python inference/benchmark_vllm.py")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*70)
    print("LOADED INFERENCE RESULTS")
    print("="*70)
    print(f"\nTotal benchmark runs: {len(df)}")
    print(f"Configurations: {df['config_name'].unique().tolist()}")
    print(f"GPU counts: {sorted(df['num_gpus'].unique())}")
    print()
    
    return df


def print_comparison_table(df):
    """Print formatted comparison table."""
    
    print("="*70)
    print("INFERENCE COMPARISON")
    print("="*70)
    print()
    
    # Group by configuration, show best concurrency for each
    summary = df.loc[df.groupby('config_name')['tokens_per_second'].idxmax()]
    
    display_cols = [
        'config_name', 'num_gpus', 'concurrency',
        'requests_per_second', 'tokens_per_second',
        'median_latency', 'p99_latency'
    ]
    
    display_df = summary[display_cols].copy()
    display_df['requests_per_second'] = display_df['requests_per_second'].round(2)
    display_df['tokens_per_second'] = display_df['tokens_per_second'].round(1)
    display_df['median_latency'] = display_df['median_latency'].round(3)
    display_df['p99_latency'] = display_df['p99_latency'].round(3)
    
    print(display_df.to_string(index=False))
    print()


def calculate_scaling_metrics(df):
    """Calculate scaling efficiency."""
    
    # Find baseline (1 GPU) throughput
    baseline = df[df['num_gpus'] == 1]
    if len(baseline) == 0:
        return df
    
    # Get best throughput for 1 GPU
    baseline_throughput = baseline['tokens_per_second'].max()
    
    # Calculate speedup and efficiency
    df['throughput_speedup'] = df['tokens_per_second'] / baseline_throughput
    df['scaling_efficiency'] = (df['throughput_speedup'] / df['num_gpus']) * 100
    
    return df


def print_key_findings(df):
    """Print analysis insights."""
    
    print("="*70)
    print("KEY FINDINGS")
    print("="*70)
    print()
    
    # Get unique configs
    configs = df['config_name'].unique()
    
    for config in configs:
        config_df = df[df['config_name'] == config]
        best_run = config_df.loc[config_df['tokens_per_second'].idxmax()]
        
        print(f"{config.upper()}:")
        print(f"Best throughput: {best_run['tokens_per_second']:.1f} tokens/sec")
        print(f"At concurrency: {best_run['concurrency']}")
        print(f"P99 latency: {best_run['p99_latency']:.3f}s")
        
        if 'throughput_speedup' in best_run:
            print(f"Speedup: {best_run['throughput_speedup']:.2f}x")
            print(f"Efficiency: {best_run['scaling_efficiency']:.1f}%")
        
        print()


def create_plots(df):
    """Generate comparison visualizations."""
    
    print("="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Throughput vs Concurrency
    for config in df['config_name'].unique():
        config_data = df[df['config_name'] == config]
        axes[0, 0].plot(
            config_data['concurrency'],
            config_data['tokens_per_second'],
            marker='o',
            label=config,
            linewidth=2
        )
    
    axes[0, 0].set_xlabel('Concurrency', fontsize=12)
    axes[0, 0].set_ylabel('Tokens/Second', fontsize=12)
    axes[0, 0].set_title('Throughput vs Concurrency', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Latency Percentiles
    best_concurrency_df = df.loc[df.groupby('config_name')['tokens_per_second'].idxmax()]
    
    x = np.arange(len(best_concurrency_df))
    width = 0.2
    
    axes[0, 1].bar(x - width, best_concurrency_df['median_latency'], width, label='P50', color='#3498db')
    axes[0, 1].bar(x, best_concurrency_df['p90_latency'], width, label='P90', color='#2ecc71')
    axes[0, 1].bar(x + width, best_concurrency_df['p99_latency'], width, label='P99', color='#e74c3c')
    
    axes[0, 1].set_xlabel('Configuration', fontsize=12)
    axes[0, 1].set_ylabel('Latency (seconds)', fontsize=12)
    axes[0, 1].set_title('Latency Percentiles', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(best_concurrency_df['config_name'], rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Scaling Efficiency (if multiple GPU counts)
    if 'scaling_efficiency' in df.columns:
        for config in df['config_name'].unique():
            config_data = df[df['config_name'] == config]
            # Get best concurrency for each GPU count
            best_per_gpu = config_data.loc[config_data.groupby('num_gpus')['tokens_per_second'].idxmax()]
            
            axes[1, 0].plot(
                best_per_gpu['num_gpus'],
                best_per_gpu['scaling_efficiency'],
                marker='o',
                label=config,
                linewidth=2
            )
        
        axes[1, 0].axhline(y=100, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7, label='Ideal')
        axes[1, 0].set_xlabel('Number of GPUs', fontsize=12)
        axes[1, 0].set_ylabel('Efficiency (%)', fontsize=12)
        axes[1, 0].set_title('Scaling Efficiency', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Requests/sec comparison
    best_runs = df.loc[df.groupby('config_name')['requests_per_second'].idxmax()]
    
    axes[1, 1].bar(
        best_runs['config_name'],
        best_runs['requests_per_second'],
        color='#9b59b6'
    )
    axes[1, 1].set_xlabel('Configuration', fontsize=12)
    axes[1, 1].set_ylabel('Requests/Second', fontsize=12)
    axes[1, 1].set_title('Peak Requests Throughput', fontsize=14, fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    for i, (config, val) in enumerate(zip(best_runs['config_name'], best_runs['requests_per_second'])):
        axes[1, 1].text(i, val + 0.5, f'{val:.1f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    os.makedirs("results/plots", exist_ok=True)
    output_path = "results/plots/inference_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"\nPlots saved to {output_path}")


def main():
    """Main analysis function."""
    
    print("\n" + "="*70)
    print("INFERENCE RESULTS ANALYSIS")
    print("="*70)
    
    # Load results
    df = load_results()
    
    # Calculate scaling metrics
    df = calculate_scaling_metrics(df)
    
    # Print comparison
    print_comparison_table(df)
    
    # Print insights
    print_key_findings(df)
    
    # Generate plots
    create_plots(df)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nOutputs:")
    print("results/inference_metrics.csv")
    print("results/plots/inference_comparison.png")
    print()


if __name__ == "__main__":
    main()