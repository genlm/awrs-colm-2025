#!/usr/bin/env python3
"""
Script to aggregate experimental results across all tasks and methods.
Produces a CSV with mean accuracy, mean runtime, and 95% bootstrap confidence intervals.
Also generates LaTeX tables for each task.
"""

import json
import os
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import argparse


def bootstrap_confidence_interval(
    data: List[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for a dataset.

    Args:
        data: List of values to bootstrap
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95 for 95% CI)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(data) == 0:
        return (np.nan, np.nan)

    if len(data) == 1:
        return (data[0], data[0])

    # Set seed for reproducibility
    np.random.seed(seed)

    data = np.array(data)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))

    bootstrap_means = np.array(bootstrap_means)
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)

    return (lower_bound, upper_bound)


def extract_method_info(path_parts: List[str]) -> Tuple[str, Optional[int]]:
    """
    Extract method name and number of particles from path.

    Args:
        path_parts: List of path components

    Returns:
        Tuple of (method_name, n_particles)
    """
    method = None
    n_particles = None

    for part in path_parts:
        if part in ["awrs_smc", "twisted_smc", "sample_rerank", "lcd", "base-lm"]:
            method = part
        elif part.endswith("_particles"):
            n_particles = int(part.split("_")[0])

    return method, n_particles


def load_results_file(filepath: Path) -> Optional[Dict]:
    """
    Load and parse a results JSON file.

    Args:
        filepath: Path to the results file

    Returns:
        Parsed JSON data or None if file cannot be loaded
    """
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None


def crawl_results_directory(
    results_dir: Path,
) -> Dict[str, Dict[str, Dict[str, List[Dict]]]]:
    """
    Crawl the results directory and organize data by task, method, and particles.

    Args:
        results_dir: Path to the results directory

    Returns:
        Nested dictionary: {task: {method: {particles: [results]}}}
    """
    aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Walk through all directories and files
    for root, dirs, files in os.walk(results_dir):
        root_path = Path(root)

        # Look for results JSON files
        for file in files:
            if file.endswith("-results.json"):
                filepath = root_path / file

                # Extract path components relative to results directory
                rel_path = filepath.relative_to(results_dir)
                path_parts = list(rel_path.parts)

                if len(path_parts) < 2:
                    continue

                # First part is task name
                task = path_parts[0]

                # Extract method and particles info
                method, n_particles = extract_method_info(path_parts)

                if method is None:
                    continue

                # Load the results file
                results_data = load_results_file(filepath)
                if results_data is None:
                    continue

                # Store the data - treat no_particles as 1 particle
                if n_particles is None:
                    particles_key = "1_particles"
                else:
                    particles_key = f"{n_particles}_particles"
                aggregated[task][method][particles_key].append(results_data)

    return aggregated


def compute_statistics(results_list: List[Dict]) -> Dict[str, float]:
    """
    Compute mean and confidence intervals for a list of results.

    Args:
        results_list: List of result dictionaries

    Returns:
        Dictionary with computed statistics
    """
    accuracies = []
    runtimes = []

    for result in results_list:
        if "weighted_accuracy" in result:
            accuracies.append(result["weighted_accuracy"])
        if "runtime_seconds" in result:
            runtimes.append(result["runtime_seconds"])

    stats = {}

    # Accuracy statistics
    if accuracies:
        stats["mean_accuracy"] = np.mean(accuracies)
        acc_ci_lower, acc_ci_upper = bootstrap_confidence_interval(accuracies, seed=42)
        stats["accuracy_ci_lower"] = acc_ci_lower
        stats["accuracy_ci_upper"] = acc_ci_upper
        stats["n_samples_accuracy"] = len(accuracies)
    else:
        stats["mean_accuracy"] = np.nan
        stats["accuracy_ci_lower"] = np.nan
        stats["accuracy_ci_upper"] = np.nan
        stats["n_samples_accuracy"] = 0

    # Runtime statistics
    if runtimes:
        stats["mean_runtime"] = np.mean(runtimes)
        runtime_ci_lower, runtime_ci_upper = bootstrap_confidence_interval(
            runtimes, seed=43
        )
        stats["runtime_ci_lower"] = runtime_ci_lower
        stats["runtime_ci_upper"] = runtime_ci_upper
        stats["n_samples_runtime"] = len(runtimes)
    else:
        stats["mean_runtime"] = np.nan
        stats["runtime_ci_lower"] = np.nan
        stats["runtime_ci_upper"] = np.nan
        stats["n_samples_runtime"] = 0

    return stats


def format_value_with_ci(
    mean_val: float,
    ci_lower: float,
    ci_upper: float,
    mean_precision: int = 2,
    ci_precision: int = 2,
) -> str:
    """
    Format a value with confidence interval for display.

    Args:
        mean_val: Mean value
        ci_lower: Lower confidence interval bound
        ci_upper: Upper confidence interval bound
        mean_precision: Number of decimal places for mean value
        ci_precision: Number of decimal places for confidence interval

    Returns:
        Formatted string like "0.85 (0.82, 0.88)"
    """
    if np.isnan(mean_val):
        return "N/A"

    mean_format_str = f"{{:.{mean_precision}f}}"
    ci_format_str = f"{{:.{ci_precision}f}}"
    mean_str = mean_format_str.format(mean_val)

    if np.isnan(ci_lower) or np.isnan(ci_upper):
        return mean_str

    ci_lower_str = ci_format_str.format(ci_lower)
    ci_upper_str = ci_format_str.format(ci_upper)

    return f"{mean_str} ({ci_lower_str}, {ci_upper_str})"


def method_display_name(method: str, n_particles: int) -> str:
    """
    Generate display name for method including particle count.

    Args:
        method: Method name
        n_particles: Number of particles

    Returns:
        Display name for the method
    """
    method_names = {
        "awrs_smc": "AWRS-SMC",
        "twisted_smc": "Twisted SMC",
        "sample_rerank": "Sample Rerank",
        "lcd": "LCD",
        "base-lm": "Base LM",
    }

    base_name = method_names.get(method, method)

    if method in ["awrs_smc", "twisted_smc"] and n_particles > 1:
        return f"{base_name} ({n_particles})"
    else:
        return base_name


def generate_latex_tables(aggregated_data: Dict, output_dir: Path):
    """
    Generate LaTeX tables for each task.

    Args:
        aggregated_data: Nested dictionary of results
        output_dir: Directory to save LaTeX files
    """
    output_dir.mkdir(exist_ok=True)

    for task in sorted(aggregated_data.keys()):
        task_data = aggregated_data[task]

        # Collect all method-particle combinations for this task
        method_stats = []

        for method in sorted(task_data.keys()):
            for particles_key in sorted(task_data[method].keys()):
                results_list = task_data[method][particles_key]

                if not results_list:
                    continue

                stats = compute_statistics(results_list)
                n_particles = int(particles_key.split("_")[0])

                method_stats.append(
                    {"method": method, "n_particles": n_particles, "stats": stats}
                )

        if not method_stats:
            continue

        # Generate LaTeX table
        latex_filename = output_dir / f"{task}_results.tex"

        with open(latex_filename, "w") as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{lcc}\n")
            f.write("\\toprule\n")
            f.write("Method & Accuracy & Runtime (sec/ex) \\\\\n")
            f.write("\\midrule\n")

            for entry in method_stats:
                method = entry["method"]
                n_particles = entry["n_particles"]
                stats = entry["stats"]

                display_name = method_display_name(method, n_particles)

                # Format accuracy with CI
                acc_str = format_value_with_ci(
                    stats["mean_accuracy"],
                    stats["accuracy_ci_lower"],
                    stats["accuracy_ci_upper"],
                    mean_precision=3,
                    ci_precision=2,
                )

                # Format runtime with CI
                runtime_str = format_value_with_ci(
                    stats["mean_runtime"],
                    stats["runtime_ci_lower"],
                    stats["runtime_ci_upper"],
                    mean_precision=2,
                    ci_precision=2,
                )

                f.write(f"{display_name} & {acc_str} & {runtime_str} \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write(f"\\caption{{Results for {task.replace('_', ' ').title()} task}}\n")
            f.write(f"\\label{{tab:{task}_results}}\n")
            f.write("\\end{table}\n")

        print(f"LaTeX table written to {latex_filename}")


def generate_csv_report(aggregated_data: Dict, output_file: Path):
    """
    Generate CSV report from aggregated data.

    Args:
        aggregated_data: Nested dictionary of results
        output_file: Path to output CSV file
    """
    fieldnames = [
        "task",
        "method",
        "n_particles",
        "mean_accuracy",
        "accuracy_ci_lower",
        "accuracy_ci_upper",
        "mean_runtime",
        "runtime_ci_lower",
        "runtime_ci_upper",
        "n_samples_accuracy",
        "n_samples_runtime",
    ]

    rows = []

    for task in sorted(aggregated_data.keys()):
        for method in sorted(aggregated_data[task].keys()):
            for particles_key in sorted(aggregated_data[task][method].keys()):
                results_list = aggregated_data[task][method][particles_key]

                if not results_list:
                    continue

                stats = compute_statistics(results_list)

                # Extract n_particles from key
                n_particles = int(particles_key.split("_")[0])

                row = {
                    "task": task,
                    "method": method,
                    "n_particles": n_particles,
                    "mean_accuracy": stats["mean_accuracy"],
                    "accuracy_ci_lower": stats["accuracy_ci_lower"],
                    "accuracy_ci_upper": stats["accuracy_ci_upper"],
                    "mean_runtime": stats["mean_runtime"],
                    "runtime_ci_lower": stats["runtime_ci_lower"],
                    "runtime_ci_upper": stats["runtime_ci_upper"],
                    "n_samples_accuracy": stats["n_samples_accuracy"],
                    "n_samples_runtime": stats["n_samples_runtime"],
                }

                rows.append(row)

    # Write CSV file
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results written to {output_file}")
    print(f"Total rows: {len(rows)}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate experimental results and generate CSV report and LaTeX tables"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default="results",
        help="Path to results directory (default: results)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="aggregated_results.csv",
        help="Output CSV file path (default: aggregated_results.csv)",
    )
    parser.add_argument(
        "--latex-dir",
        type=Path,
        default="latex_tables",
        help="Directory for LaTeX table files (default: latex_tables)",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
        help="Number of bootstrap samples for confidence intervals (default: 10000)",
    )

    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: Results directory {args.results_dir} does not exist")
        return 1

    print(f"Crawling results directory: {args.results_dir}")
    aggregated_data = crawl_results_directory(args.results_dir)

    if not aggregated_data:
        print("No results found in the specified directory")
        return 1

    print(f"Found results for {len(aggregated_data)} tasks")
    for task, methods in aggregated_data.items():
        print(f"  {task}: {len(methods)} methods")
        for method, particles in methods.items():
            print(f"    {method}: {list(particles.keys())}")

    print(f"Generating CSV report with {args.bootstrap_samples} bootstrap samples...")
    generate_csv_report(aggregated_data, args.output)

    print("Generating LaTeX tables...")
    generate_latex_tables(aggregated_data, args.latex_dir)

    return 0


if __name__ == "__main__":
    exit(main())
