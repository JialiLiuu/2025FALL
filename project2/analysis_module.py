"""
Filename: analysis_module.py
Author: jiali liu
Date: 2025-10-15
Description: 
This module benchmarks and visualizes the performance of two algorithms for computing Pareto-optimal points:
1. Iterative algorithm with time complexity O(nh)
2. Divide & Conquer algorithm with time complexity O(n log n)

It generates random 2D points, computes Pareto staircases using both algorithms, measures execution time,
normalizes theoretical complexity curves, and plots visual comparisons and performance metrics
"""

import random
import time
import math
import matplotlib.pyplot as plt
from pareto_optimal import staircase_oh_nh, staircase_dc_optimized

def generate_random_points(n, seed_offset=45):
    """Generates n random 2D points with floating-point coordinates, seed offset for reproducibility."""
    random.seed(seed_offset + n)
    return [(random.uniform(0, 1000), random.uniform(0, 1000)) for _ in range(n)]

def plot_staircase_comparison(n_values):
    """Plots visual comparison of both Pareto staircase algorithms."""
    plt.figure(figsize=(14, 8))

    for i, n in enumerate(n_values, start=1):
        points = generate_random_points(n)

        # Compute Pareto staircases
        stair_oh_nh = staircase_oh_nh(points.copy())
        stair_dc = staircase_dc_optimized(points.copy())

        # Plot each subplot
        plt.subplot(2, (len(n_values) + 1) // 2, i)
        plt.scatter(*zip(*points), color='gray', alpha=0.5, label='All Points', s=10)
        plt.plot(*zip(*stair_oh_nh), 'bo-', linewidth=1.2, markersize=4, label='Iterative: O(nh)')
        plt.plot(*zip(*stair_dc), 'ro--', linewidth=1.2, markersize=4, label='Divide & Conquer: O(n log n)')

        plt.title(f'n = {n} points')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.legend(fontsize=8, loc='lower left')

    plt.suptitle('Visual Comparison of Staircase Algorithms', fontsize=14, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def benchmark_algorithms(n_values):
    """Benchmarks both algorithms and compares experimental vs theoretical performance."""
    dc_times, nh_times = [], []
    h_values, nh_theory = [], []

    for n in n_values:
        points = generate_random_points(n)

        # Time Divide & Conquer
        start = time.perf_counter_ns()
        staircase_dc_optimized(points)
        dc_times.append(time.perf_counter_ns() - start)

        # Time Iterative and measure h
        start = time.perf_counter_ns()
        h = len(staircase_oh_nh(points))
        nh_times.append(time.perf_counter_ns() - start)
        h_values.append(h)
        nh_theory.append(n * h)

    return dc_times, nh_times, h_values, nh_theory

def normalize_theoretical_curves(n_values, dc_times, nh_times, nh_theory):
    """Normalizes theoretical curves using median scaling factors."""
    nlogn_theory = [n * math.log(n) for n in n_values]

    dc_ratios = [exp / theo for exp, theo in zip(dc_times, nlogn_theory)]
    nh_ratios = [exp / theo for exp, theo in zip(nh_times, nh_theory)]

    dc_scale = sorted(dc_ratios)[len(dc_ratios) // 2]
    nh_scale = sorted(nh_ratios)[len(nh_ratios) // 2]

    dc_theoretical = [dc_scale * v for v in nlogn_theory]
    nh_theoretical = [nh_scale * v for v in nh_theory]

    return dc_theoretical, nh_theoretical, dc_scale, nh_scale

def print_comparison_table(n_values, h_values, dc_times, nh_times, dc_theoretical, nh_theoretical):
    """Prints formatted comparison table of results."""
    print("\nComparison Table:")
    print("n\t\th\t\tDC_Experimental(ns)\tNH_Experimental(ns)\tNormalized (DC)\tNormalized (NH)")
    print("-" * 100)
    for i, n in enumerate(n_values):
        print(f"{n}\t\t{h_values[i]}\t\t{dc_times[i]:.6f}\t{nh_times[i]:.6f}\t{dc_theoretical[i]:.6f}\t{nh_theoretical[i]:.6f}")

def plot_performance_curves(n_values, dc_times, nh_times, dc_theoretical, nh_theoretical):
    """Plots experimental and theoretical performance curves."""
    plt.figure(figsize=(12, 8))

    # Iterative O(nh)
    plt.subplot(2, 2, 1)
    plt.plot(n_values, nh_times, 'go-', label='Experimental')
    plt.plot(n_values, nh_theoretical, 'r--', label='Theoretical (normalized nh)')
    plt.title('Iterative Algorithm: O(nh)')
    plt.xlabel('n')
    plt.ylabel('Time (ns)')
    plt.legend()
    plt.grid(True)

    # Divide & Conquer O(n log n)
    plt.subplot(2, 2, 3)
    plt.plot(n_values, dc_times, 'bo-', label='Experimental')
    plt.plot(n_values, dc_theoretical, 'r--', label='Theoretical (normalized n log n)')
    plt.title('Divide & Conquer: O(n log n)')
    plt.xlabel('n')
    plt.ylabel('Time (ns)')
    plt.legend()
    plt.grid(True)

    # Log-log plots O(nh)
    plt.subplot(2, 2, 2)
    plt.loglog(n_values, nh_times, 'bo-', label='Experimental')
    plt.loglog(n_values, nh_theoretical, 'r--', label='Theoretical (normalized nh)')
    plt.title('Iterative Algorithm (Log-Log)')
    plt.xlabel('n (log scale)')
    plt.ylabel('Time (log scale)')
    plt.legend()
    plt.grid(True)

    # Log-log plots O(n log n)
    plt.subplot(2, 2, 4)
    plt.loglog(n_values, dc_times, 'bo-', label='Experimental')
    plt.loglog(n_values, dc_theoretical, 'r--', label='Theoretical (normalized n log n)')
    plt.title('Divide & Conquer (Log-Log)')
    plt.xlabel('n (log scale)')
    plt.ylabel('Time (log scale)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# ---------- Run Experiment ----------
n_values = [100, 200, 400, 800, 1600, 3200, 6400, 12800]

plot_staircase_comparison(n_values)

dc_times, nh_times, h_values, nh_theory = benchmark_algorithms(n_values)
dc_theoretical, nh_theoretical, dc_scale, nh_scale = normalize_theoretical_curves(n_values, dc_times, nh_times, nh_theory)

print(f"\nDivide & Conquer scaling factor: {dc_scale:.6f}")
print(f"Iterative scaling factor: {nh_scale:.6f}")

print_comparison_table(n_values, h_values, dc_times, nh_times, dc_theoretical, nh_theoretical)
plot_performance_curves(n_values, dc_times, nh_times, dc_theoretical, nh_theoretical)