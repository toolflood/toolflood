#!/usr/bin/env python3
"""
Plot ASR vs B Level from CSV file with legend above the plot.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Plot ASR vs B Level from CSV with legend above"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("experiment_output_b_levels/asr_by_b_level.csv"),
        help="Path to CSV file with ASR data"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for plot (default: same directory as CSV with _legend_above suffix)"
    )
    
    args = parser.parse_args()
    
    # Read CSV
    csv_path = args.csv.resolve()
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # Pivot data for plotting
    pivot_df = df.pivot(index="Domain", columns="B", values="ASR")
    
    # Create mapping from domain to task number (Task 1, Task 2, etc.)
    # Use sorted order to ensure consistent numbering
    sorted_domains = sorted(pivot_df.index)
    domain_to_task = {domain: f"Task {i+1}" for i, domain in enumerate(sorted_domains)}
    
    # Create figure with extra space at top for legend
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot lines for each domain
    for domain in pivot_df.index:
        task_label = domain_to_task[domain]
        ax.plot(pivot_df.columns, pivot_df.loc[domain], marker="o", label=task_label, linewidth=2)
    
    # Set labels and title
    ax.set_xlabel("B Level", fontsize=12)
    ax.set_ylabel("ASR (Attack Success Rate)", fontsize=12)
    ax.set_title("ASR vs B Level per Task (Test)", fontsize=14, fontweight="bold")
    
    # Add grid
    ax.grid(alpha=0.3)
    
    # Set x-axis ticks to match B levels
    ax.set_xticks(pivot_df.columns)
    
    # Place legend above the plot
    # Use ncol to arrange legend items horizontally
    ncol = min(len(pivot_df.index), 5)  # Max 5 columns, adjust if needed
    ax.legend(
        bbox_to_anchor=(0.5, 1.08),
        loc="lower center",
        ncol=ncol,
        frameon=True,
        fontsize=9
    )
    
    # Adjust layout to make room for legend
    plt.tight_layout(rect=[0, 0, 1, 0.92])  # Leave top 8% for legend
    
    # Determine output path
    if args.output:
        output_path = args.output.resolve()
    else:
        # Default: same directory as CSV with _legend_above suffix
        output_path = csv_path.parent / f"{csv_path.stem}_legend_above.png"
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
