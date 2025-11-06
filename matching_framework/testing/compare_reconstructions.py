#!/usr/bin/env python3
"""
Compare multiple reconstruction experiments from CSV files.

This script reads multiple statistics.csv files from different reconstruction
runs and creates a comparison table for easy analysis.

Usage:
    python -m matching_framework.testing.compare_reconstructions \
        results/exp1/statistics.csv \
        results/exp2/statistics.csv \
        results/exp3/statistics.csv \
        --output comparison.csv

Example:
    # Run multiple experiments
    python -m matching_framework.sfm_pipeline \
        --images data/temple --output results/sp_lg \
        --extractor superpoint --matcher lightglue

    python -m matching_framework.sfm_pipeline \
        --images data/temple --output results/sift_nn \
        --extractor sift --matcher nn

    python -m matching_framework.sfm_pipeline \
        --images data/temple --output results/orb_nn \
        --extractor orb --matcher nn

    # Compare results
    python -m matching_framework.testing.compare_reconstructions \
        results/sp_lg/statistics.csv \
        results/sift_nn/statistics.csv \
        results/orb_nn/statistics.csv \
        --output comparison.csv
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List
import sys


def parse_statistics_csv(csv_file: Path) -> Dict[str, str]:
    """Parse a statistics CSV file.

    Args:
        csv_file: Path to statistics.csv

    Returns:
        Dictionary of metric -> value
    """
    stats = {}

    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        for row in reader:
            if len(row) >= 2:
                metric, value = row[0], row[1]
                if metric and value:  # Skip empty rows
                    stats[metric] = value

    return stats


def create_comparison_table(
    csv_files: List[Path],
    output_file: Path = None
) -> None:
    """Create a comparison table from multiple CSV files.

    Args:
        csv_files: List of paths to statistics.csv files
        output_file: Optional output CSV file for comparison table
    """
    # Parse all CSV files
    all_stats = []
    for csv_file in csv_files:
        if not csv_file.exists():
            print(f"Warning: {csv_file} not found, skipping")
            continue

        stats = parse_statistics_csv(csv_file)
        stats["experiment"] = csv_file.parent.name  # Use directory name as experiment name
        all_stats.append(stats)

    if not all_stats:
        print("Error: No valid CSV files found")
        return

    # Get all metric keys (use first experiment as reference)
    metrics = [
        "extractor",
        "matcher",
        "pair_selector",
        "device",
        "num_images",
        "num_pairs",
        "num_matches_total",
        "avg_matches_per_pair",
        "feature_extraction_time",
        "matching_time",
        "num_registered_images",
        "registration_rate",
        "num_3d_points",
        "num_observations",
        "mean_track_length",
        "median_track_length",
        "mean_reprojection_error",
        "median_reprojection_error",
        "reconstruction_time",
        "total_time",
    ]

    # Create comparison table
    print("\n" + "=" * 120)
    print("RECONSTRUCTION COMPARISON")
    print("=" * 120)

    # Print header
    header = ["Metric"] + [stats["experiment"] for stats in all_stats]
    print(f"{header[0]:<40}", end="")
    for exp_name in header[1:]:
        print(f"{exp_name:>15}", end="")
    print()
    print("-" * 120)

    # Print each metric
    for metric in metrics:
        # Skip if metric not in any experiment
        if not any(metric in stats for stats in all_stats):
            continue

        # Print metric name
        print(f"{metric:<40}", end="")

        # Print values for each experiment
        for stats in all_stats:
            value = stats.get(metric, "N/A")
            print(f"{value:>15}", end="")

        print()

    print("=" * 120 + "\n")

    # Save to CSV if requested
    if output_file:
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(header)

            # Write each metric
            for metric in metrics:
                if not any(metric in stats for stats in all_stats):
                    continue

                row = [metric]
                for stats in all_stats:
                    row.append(stats.get(metric, "N/A"))
                writer.writerow(row)

        print(f"Comparison table saved to {output_file}")

    # Print key insights
    print("\n" + "=" * 120)
    print("KEY INSIGHTS")
    print("=" * 120)

    # Find best extractor/matcher combination
    if all("num_registered_images" in stats for stats in all_stats):
        best_registration = max(
            all_stats,
            key=lambda s: float(s.get("registration_rate", 0)) if s.get("registration_rate") != "N/A" else 0
        )
        print(f"Best registration rate: {best_registration['experiment']}")
        print(f"  {best_registration.get('extractor', 'N/A')} + {best_registration.get('matcher', 'N/A')}")
        print(f"  {best_registration.get('registration_rate', 'N/A')} registration rate")

    if all("num_3d_points" in stats for stats in all_stats):
        best_points = max(
            all_stats,
            key=lambda s: int(s.get("num_3d_points", 0)) if s.get("num_3d_points", "0").isdigit() else 0
        )
        print(f"\nMost 3D points: {best_points['experiment']}")
        print(f"  {best_points.get('num_3d_points', 'N/A')} points")

    if all("mean_reprojection_error" in stats for stats in all_stats):
        best_error = min(
            all_stats,
            key=lambda s: float(s.get("mean_reprojection_error", float("inf"))) if s.get("mean_reprojection_error") not in ["N/A", ""] else float("inf")
        )
        if best_error.get("mean_reprojection_error") not in ["N/A", ""]:
            print(f"\nLowest reprojection error: {best_error['experiment']}")
            print(f"  {best_error.get('mean_reprojection_error', 'N/A')} px")

    if all("total_time" in stats for stats in all_stats):
        fastest = min(
            all_stats,
            key=lambda s: float(s.get("total_time", float("inf"))) if s.get("total_time") not in ["N/A", ""] else float("inf")
        )
        print(f"\nFastest: {fastest['experiment']}")
        print(f"  {fastest.get('total_time', 'N/A')}s total")

    print("=" * 120 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple reconstruction experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare three experiments
  python -m matching_framework.testing.compare_reconstructions \
      results/exp1/statistics.csv \
      results/exp2/statistics.csv \
      results/exp3/statistics.csv

  # Save comparison to file
  python -m matching_framework.testing.compare_reconstructions \
      results/*/statistics.csv \
      --output comparison.csv
        """
    )

    parser.add_argument(
        "csv_files",
        type=str,
        nargs="+",
        help="List of statistics.csv files to compare"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file for comparison table"
    )

    args = parser.parse_args()

    # Convert to Path objects
    csv_files = [Path(f) for f in args.csv_files]
    output_file = Path(args.output) if args.output else None

    # Create comparison
    create_comparison_table(csv_files, output_file)


if __name__ == "__main__":
    main()
