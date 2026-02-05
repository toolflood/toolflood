#!/usr/bin/env python3
"""
Calculate average B, Poisoning rate, ASR, and TDR per model from all results_table.csv
under outputs/. Concatenates results into one table with attack and benchmark columns.
"""

from pathlib import Path
import pandas as pd


def get_outputs_dir() -> Path:
    """Outputs directory (project root / outputs)."""
    return Path(__file__).resolve().parent.parent.parent / "outputs"


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load results from CSV file."""
    return pd.read_csv(csv_path)


def calculate_metrics_per_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average B, Poisoning rate, ASR, and TDR per model.

    Uses columns from results_table.csv:
    - B = Num_Injected_Tools
    - Poisoning rate = Num_Injected_Tools / Benign_Tools (per row)
    - ASR, TDR from CSV

    Args:
        df: DataFrame from results_table.csv (Model, Num_Injected_Tools, Benign_Tools, ASR, TDR)
    """
    df = df.copy()
    df["B"] = df["Num_Injected_Tools"]
    df["Poisoning_Rate"] = df["Num_Injected_Tools"] / df["Benign_Tools"]

    metrics_per_model = (
        df.groupby("Model")
        .agg(
            {
                "B": "mean",
                "Poisoning_Rate": "mean",
                "ASR": "mean",
                "TDR": "mean",
            }
        )
        .reset_index()
    )
    metrics_per_model.columns = [
        "Model",
        "Avg_B",
        "Avg_Poisoning_Rate",
        "Avg_ASR",
        "Avg_TDR",
    ]
    return metrics_per_model


def find_results_tables(outputs_dir: Path) -> list[tuple[Path, str, str]]:
    """Find top-level results_table.csv only: outputs/<attack>/<benchmark>/results_table.csv.
    Skips per-task CSVs (e.g. rsi/toole/Scenario/victim_emb.../results_table.csv) so each
    (attack, benchmark) contributes one row per model.
    """
    results = []
    for csv_path in outputs_dir.rglob("results_table.csv"):
        try:
            rel = csv_path.relative_to(outputs_dir)
            parts = rel.parts
            if len(parts) == 3 and parts[2] == "results_table.csv":
                attack, benchmark = parts[0], parts[1]
                results.append((csv_path, attack, benchmark))
        except ValueError:
            continue
    return results


if __name__ == "__main__":
    outputs_dir = get_outputs_dir()
    if not outputs_dir.exists():
        raise SystemExit(f"Outputs dir not found: {outputs_dir}")

    tables = find_results_tables(outputs_dir)
    if not tables:
        raise SystemExit(f"No results_table.csv found under {outputs_dir}")

    all_metrics = []
    for csv_path, attack, benchmark in tables:
        try:
            df = load_results(csv_path)
            metrics_df = calculate_metrics_per_model(df)
            metrics_df.insert(0, "benchmark", benchmark)
            metrics_df.insert(0, "attack", attack)
            all_metrics.append(metrics_df)
        except Exception as e:
            print(f"Skip {csv_path}: {e}")

    if not all_metrics:
        raise SystemExit("No CSVs could be processed.")

    combined = pd.concat(all_metrics, ignore_index=True)
    # Reorder columns: attack, benchmark, Model, then metrics
    cols = ["attack", "benchmark", "Model", "Avg_B", "Avg_Poisoning_Rate", "Avg_ASR", "Avg_TDR"]
    combined = combined[cols]

    # Format poisoning rate as percentage for display
    display_df = combined.copy()
    display_df["Avg_Poisoning_Rate"] = (display_df["Avg_Poisoning_Rate"] * 100).round(4)

    print("\n" + "=" * 80)
    print("METRICS PER MODEL (all outputs)")
    print("=" * 80)
    print(display_df.to_string(index=False))

    output_csv = outputs_dir / "metrics_per_model.csv"
    combined.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"CSV files processed: {len(all_metrics)}")
    print(f"Total rows: {len(combined)}")
    print(f"Attacks: {sorted(combined['attack'].unique().tolist())}")
    print(f"Benchmarks: {sorted(combined['benchmark'].unique().tolist())}")
