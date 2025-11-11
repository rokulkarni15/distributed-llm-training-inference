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


def load_and_calculate():
    """Load results and calculate speedup/efficiency."""
    
    csv_path = "results/training_metrics.csv"
    
    if not os.path.exists(csv_path):
        print(f"No results found at {csv_path}")
        print("\nRun training experiments first!")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*70)
    print(f"LOADED {len(df)} EXPERIMENTS")
    print("="*70)
    print(f"Experiments: {', '.join(df['experiment'].tolist())}")
    print()
    
    # Find baseline time for speedup calculation
    baseline_row = df[df['experiment'] == 'baseline']
    
    if len(baseline_row) == 0:
        print("No baseline experiment found.")
        print("Using first experiment as reference for speedup calculation.")
        baseline_time = df.iloc[0]['training_time_hours']
    else:
        baseline_time = baseline_row['training_time_hours'].values[0]
        print(f"Baseline time: {baseline_time:.2f} hours")
    
    # Calculate speedup and efficiency
    df['speedup'] = baseline_time / df['training_time_hours']
    df['efficiency_percent'] = (df['speedup'] / df['num_gpus']) * 100
    
    return df


def print_comparison_table(df):
    """Print formatted comparison table."""
    
    print("\n" + "="*70)
    print("TRAINING COMPARISON")
    print("="*70)
    print()
    
    # Format for display
    display_df = df.copy()
    display_df['training_time_hours'] = display_df['training_time_hours'].round(2)
    display_df['samples_per_second'] = display_df['samples_per_second'].round(1)
    display_df['peak_memory_gb'] = display_df['peak_memory_gb'].round(2)
    display_df['speedup'] = display_df['speedup'].round(2)
    display_df['efficiency_percent'] = display_df['efficiency_percent'].round(1)
    
    print(display_df.to_string(index=False))
    print()


def print_key_findings(df):
    """Print analysis insights."""
    
    print("="*70)
    print("KEY FINDINGS")
    print("="*70)
    print()
    
    baseline = df[df['experiment'] == 'baseline']
    
    if len(baseline) == 0:
        print("No baseline found - showing all experiments")
        for _, row in df.iterrows():
            print(f"{row['experiment']}: {row['training_time_hours']:.2f}h")
        return
    
    baseline_time = baseline['training_time_hours'].values[0]
    
    # Analyze each experiment vs baseline
    for _, row in df.iterrows():
        if row['experiment'] == 'baseline':
            continue
        
        time_saved = baseline_time - row['training_time_hours']
        
        print(f"{row['experiment'].upper()}:")
        print(f"  • {row['speedup']:.2f}x speedup ({row['efficiency_percent']:.1f}% efficient)")
        print(f"  • Saved {time_saved:.2f} hours vs baseline")
        print(f"  • {row['peak_memory_gb']:.1f} GB memory/GPU")
        print()


def create_plots(df):
    """Generate comparison visualizations."""
    
    print("="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Training Time
    axes[0, 0].bar(df['experiment'], df['training_time_hours'], color='#3498db')
    axes[0, 0].set_title('Training Time', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Hours', fontsize=12)
    axes[0, 0].set_xlabel('Experiment', fontsize=12)
    for i, (exp, val) in enumerate(zip(df['experiment'], df['training_time_hours'])):
        axes[0, 0].text(i, val + 0.2, f'{val:.1f}h', ha='center', fontsize=10)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Speedup
    axes[0, 1].bar(df['experiment'], df['speedup'], color='#2ecc71')
    axes[0, 1].axhline(y=1, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7, label='Baseline')
    axes[0, 1].set_title('Speedup vs Baseline', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Speedup (x)', fontsize=12)
    axes[0, 1].set_xlabel('Experiment', fontsize=12)
    axes[0, 1].legend()
    for i, (exp, val) in enumerate(zip(df['experiment'], df['speedup'])):
        axes[0, 1].text(i, val + 0.1, f'{val:.2f}x', ha='center', fontsize=10)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Memory per GPU
    axes[1, 0].bar(df['experiment'], df['peak_memory_gb'], color='#9b59b6')
    axes[1, 0].set_title('Peak Memory per GPU', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('GB', fontsize=12)
    axes[1, 0].set_xlabel('Experiment', fontsize=12)
    for i, (exp, val) in enumerate(zip(df['experiment'], df['peak_memory_gb'])):
        axes[1, 0].text(i, val + 0.3, f'{val:.1f}', ha='center', fontsize=10)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Scaling Efficiency
    axes[1, 1].plot(df['num_gpus'], df['efficiency_percent'], 
                    marker='o', linewidth=3, markersize=12, color='#3498db', label='Actual')
    
    # Ideal line
    max_gpus = df['num_gpus'].max()
    axes[1, 1].plot([1, max_gpus], [100, 100], 
                    linestyle='--', linewidth=2, color='#e74c3c', alpha=0.7, label='Ideal')
    
    axes[1, 1].set_title('Scaling Efficiency', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Efficiency (%)', fontsize=12)
    axes[1, 1].set_xlabel('Number of GPUs', fontsize=12)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    for gpus, eff in zip(df['num_gpus'], df['efficiency_percent']):
        axes[1, 1].text(gpus, eff + 3, f'{eff:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    os.makedirs("results/plots", exist_ok=True)
    output_path = "results/plots/training_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"\nPlots saved to {output_path}")


def main():
    """Main analysis function."""
    
    print("\n" + "="*70)
    print("TRAINING RESULTS ANALYSIS")
    print("="*70)
    
    # Load and calculate
    df = load_and_calculate()
    
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
    print("  • results/training_metrics.csv (raw data)")
    print("  • results/plots/training_comparison.png (visualizations)")
    print()


if __name__ == "__main__":
    main()