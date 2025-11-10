#!/usr/bin/env python
"""
Compare training results across experiments.

Usage:
    python scripts/compare_training.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os


def load_and_analyze():
    """Load results and generate comparison."""
    
    csv_path = "results/training_metrics.csv"
    
    if not os.path.exists(csv_path):
        print(f"No results found at {csv_path}")
        print("\nRun training experiments first!")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*70)
    print("TRAINING COMPARISON")
    print("="*70)
    print()
    print(df.to_string(index=False))
    print()
    
    # Key findings
    print("="*70)
    print("KEY FINDINGS")
    print("="*70)
    print()
    
    baseline = df[df['experiment'] == 'baseline']
    
    for _, row in df.iterrows():
        if row['experiment'] == 'baseline':
            continue
        
        baseline_time = baseline['training_time_hours'].values[0]
        time_saved = baseline_time - row['training_time_hours']
        
        print(f"{row['experiment'].upper()}:")
        print(f"  • {row['speedup']:.2f}x speedup ({row['efficiency_percent']:.1f}% efficient)")
        print(f"  • {time_saved:.2f} hours saved vs baseline")
        print(f"  • {row['peak_memory_gb']:.1f} GB memory per GPU")
        print()
    
    # Create plots
    create_plots(df)
    
    return df


def create_plots(df):
    """Generate comparison plots."""
    
    print("="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Training Time
    axes[0, 0].bar(df['experiment'], df['training_time_hours'], color='#3498db')
    axes[0, 0].set_title('Training Time', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Hours', fontsize=12)
    for i, (exp, val) in enumerate(zip(df['experiment'], df['training_time_hours'])):
        axes[0, 0].text(i, val + 0.2, f'{val:.1f}h', ha='center', fontsize=11)
    
    # Plot 2: Speedup
    axes[0, 1].bar(df['experiment'], df['speedup'], color='#2ecc71')
    axes[0, 1].axhline(y=1, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7)
    axes[0, 1].set_title('Speedup vs Baseline', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Speedup (x)', fontsize=12)
    for i, (exp, val) in enumerate(zip(df['experiment'], df['speedup'])):
        axes[0, 1].text(i, val + 0.1, f'{val:.2f}x', ha='center', fontsize=11)
    
    # Plot 3: Memory per GPU
    axes[1, 0].bar(df['experiment'], df['peak_memory_gb'], color='#9b59b6')
    axes[1, 0].set_title('Peak Memory per GPU', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('GB', fontsize=12)
    for i, (exp, val) in enumerate(zip(df['experiment'], df['peak_memory_gb'])):
        axes[1, 0].text(i, val + 0.3, f'{val:.1f} GB', ha='center', fontsize=11)
    
    # Plot 4: Scaling Efficiency
    axes[1, 1].plot(df['num_gpus'], df['efficiency_percent'], 
                    marker='o', linewidth=3, markersize=12, color='#3498db')
    axes[1, 1].plot(df['num_gpus'], [100]*len(df), 
                    linestyle='--', linewidth=2, color='#e74c3c', alpha=0.7, label='Ideal')
    axes[1, 1].set_title('Scaling Efficiency', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Efficiency (%)', fontsize=12)
    axes[1, 1].set_xlabel('Number of GPUs', fontsize=12)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    for gpus, eff in zip(df['num_gpus'], df['efficiency_percent']):
        axes[1, 1].text(gpus, eff + 3, f'{eff:.1f}%', ha='center', fontsize=11)
    
    plt.tight_layout()
    
    # Save
    os.makedirs("results/plots", exist_ok=True)
    output_path = "results/plots/training_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"\nPlots saved to {output_path}")
    print("\n" + "="*70)


if __name__ == "__main__":
    load_and_analyze()