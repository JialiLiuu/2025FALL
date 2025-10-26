"""
Filename: analysis_module.py
Author: Jiali Liu
Date: 2025-10-15
Description:
This module benchmarks and visualizes the performance of four algorithms for computing Pareto-optimal points:
1. Iterative algorithm with time complexity O(nh)
2. Divide & Conquer algorithm with time complexity O(n log n)
3. Logarithmic insertion algorithm with time complexity O(n log h)
4. Presorted linear algorithm with time complexity O(n) (requires input sorted by x)

It generates random 2D points, computes Pareto staircases using all algorithms, measures execution time,
normalizes theoretical complexity curves, and plots visual comparisons and performance metrics.
"""

import random
import time
import math
import matplotlib.pyplot as plt
from pareto_optimal import (
    staircase_oh_nh,
    staircase_dc_optimized,
    staircase_nlogh,
    staircase_presorted
)

def generate_random_points(n, seed_offset=45):
    """Generates n random 2D points with floating-point coordinates, seed offset for reproducibility."""
    random.seed(seed_offset + n)
    return [(random.uniform(0, 1000), random.uniform(0, 1000)) for _ in range(n)]

def plot_staircase_comparison(n_values):
    """Plots visual comparison of all Pareto staircase algorithms."""
    plt.figure(figsize=(12, 8))

    for i, n in enumerate(n_values, start=1):
        points = generate_random_points(n)
        sorted_points = sorted(points, key=lambda p: p[0])  # For staircase_presorted
        # Compute Pareto staircases
        stair_oh_nh = staircase_oh_nh(points.copy())
        stair_dc = staircase_dc_optimized(points.copy())
        stair_nlogh = staircase_nlogh(points.copy())
        stair_presorted = staircase_presorted(sorted_points)

        # Plot each subplot
        plt.subplot(2, (len(n_values) + 1) // 2, i)
        plt.scatter(*zip(*points), color='gray', alpha=0.5, label='All Points', s=10)
        plt.plot(*zip(*stair_oh_nh), 'bo-', linewidth=1.2, markersize=8, label='O(nh)')
        plt.plot(*zip(*stair_dc), 'ro--', linewidth=1.2, markersize=6, label='O(n log n)')
        plt.plot(*zip(*stair_nlogh), 'gs-', linewidth=1.2, markersize=4, label='O(n log h)')
        plt.plot(*zip(*stair_presorted), 'k^-', linewidth=1.2, markersize=2, label='O(n) presorted')

        plt.title(f'n = {n} points')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.legend(fontsize=7, loc='lower left')

    plt.suptitle('Visual Comparison of Staircase Algorithms', fontsize=14, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def benchmark_algorithms(n_values):
    """Benchmarks all algorithms and compares experimental vs theoretical performance."""
    dc_times, nh_times, nlogh_times, presorted_times = [], [], [], []
    h_values, nh_theory, nlogh_theory = [], [], []

    for n in n_values:
        points = generate_random_points(n)
        sorted_points = sorted(points, key=lambda p: p[0])

        # Divide & Conquer
        start = time.perf_counter_ns()
        staircase_dc_optimized(points)
        dc_times.append(time.perf_counter_ns() - start)

        # Iterative O(nh)
        start = time.perf_counter_ns()
        stair_nh = staircase_oh_nh(points)
        nh_times.append(time.perf_counter_ns() - start)
        h = len(stair_nh)
        h_values.append(h)
        nh_theory.append(n * h)

        # Logarithmic insertion O(n log h)
        start = time.perf_counter_ns()
        staircase_nlogh(points)
        nlogh_times.append(time.perf_counter_ns() - start)
        nlogh_theory.append(n * math.log(h + 1))  # Avoid log(0)

        # Presorted linear O(n) 
        start = time.perf_counter_ns()
        staircase_presorted(sorted_points)
        presorted_times.append(time.perf_counter_ns() - start)

    return dc_times, nh_times, nlogh_times, presorted_times, h_values, nh_theory, nlogh_theory

def normalize_theoretical_curves(n_values, dc_times, nh_times, nlogh_times, presorted_times, nh_theory, nlogh_theory):
    """Normalizes theoretical curves using median scaling factors."""
    nlogn_theory = [n * math.log(n) for n in n_values]
    n_theory = n_values

    dc_ratios = [exp / theo for exp, theo in zip(dc_times, nlogn_theory)]
    nh_ratios = [exp / theo for exp, theo in zip(nh_times, nh_theory)]
    nlogh_ratios = [exp / theo for exp, theo in zip(nlogh_times, nlogh_theory)]
    presorted_ratios = [exp / n for exp, n in zip(presorted_times, n_theory)]

    dc_scale = sorted(dc_ratios)[len(dc_ratios) // 2]
    nh_scale = sorted(nh_ratios)[len(nh_ratios) // 2]
    nlogh_scale = sorted(nlogh_ratios)[len(nlogh_ratios) // 2]
    presorted_scale = sorted(presorted_ratios)[len(presorted_ratios) // 2]
    print(f"\nDivide & Conquer scaling factor: {dc_scale:.6f}")
    print(f"Iterative scaling factor: {nh_scale:.6f}")
    print(f"Logarithmic insertion scaling factor: {nlogh_scale:.6f}")
    print(f"Presorted linear scaling factor: {presorted_scale:.6f}")

    dc_theoretical = [dc_scale * v for v in nlogn_theory]
    nh_theoretical = [nh_scale * v for v in nh_theory]
    nlogh_theoretical = [nlogh_scale * v for v in nlogh_theory]
    presorted_theoretical = [presorted_scale * v for v in n_theory]

    return dc_theoretical, nh_theoretical, nlogh_theoretical, presorted_theoretical

def print_comparison_table(n_values, h_values,
                            dc_times, nh_times, nlogh_times, presorted_times,
                            dc_theoretical, nh_theoretical, nlogh_theoretical, presorted_theoretical):
    """
    Prints formatted comparison table of results including experimental and theoretical values.
    """
    print("\nComparison Table:")
    print("n\t\th\t\t"
          "DC_Exp(ns)\tNH_Exp(ns)\tNlogH_Exp(ns)\tPresorted_Exp(ns)\t"
          "DC_Theo(ns)\tNH_Theo(ns)\tNlogH_Theo(ns)\tPresorted_Theo(ns)")
    print("-" * 150)

    for i, n in enumerate(n_values):
        print(f"{n}\t\t{h_values[i]}\t\t"
              f"{dc_times[i]:.6f}\t{nh_times[i]:.6f}\t{nlogh_times[i]:.6f}\t{presorted_times[i]:.6f}\t"
              f"{dc_theoretical[i]:.6f}\t{nh_theoretical[i]:.6f}\t{nlogh_theoretical[i]:.6f}\t{presorted_theoretical[i]:.6f}")
        
def plot_performance_curves(n_values, dc_times, nh_times, nlogh_times, presorted_times,
                            dc_theoretical, nh_theoretical, nlogh_theoretical, presorted_theoretical):
    """Plots experimental and theoretical performance curves."""
    plt.figure(figsize=(12, 8))

    # Iterative O(nh)
    plt.subplot(4, 2, 1)
    plt.plot(n_values, nh_times, 'go-', label='Experimental')
    plt.plot(n_values, nh_theoretical, 'r--', label='Theoretical (normalized nh)')
    plt.title('Iterative Algorithm: O(nh)')
    plt.xlabel('n')
    plt.ylabel('Time (ns)')
    plt.legend()
    plt.grid(True)

    # Log-log plots O(nh)
    plt.subplot(4, 2, 2)
    plt.loglog(n_values, nh_times, 'bo-', label='Experimental')
    plt.loglog(n_values, nh_theoretical, 'r--', label='Theoretical (normalized nh)')
    plt.title('Iterative Algorithm O(nh)')
    plt.xlabel('n (log scale)')
    plt.ylabel('Time (log scale)')
    plt.legend()
    plt.grid(True)

    # Divide & Conquer O(n log n)
    plt.subplot(4, 2, 3)
    plt.plot(n_values, dc_times, 'bo-', label='Experimental')
    plt.plot(n_values, dc_theoretical, 'r--', label='Theoretical (normalized n log n)')
    plt.title('Divide & Conquer: O(n log n)')
    plt.xlabel('n')
    plt.ylabel('Time (ns)')
    plt.legend()
    plt.grid(True)

    # Log-log plots O(n log n)
    plt.subplot(4, 2, 4)
    plt.loglog(n_values, dc_times, 'bo-', label='Experimental')
    plt.loglog(n_values, dc_theoretical, 'r--', label='Theoretical (normalized n log n)')
    plt.title('Divide & Conquer O(n log n)')
    plt.xlabel('n (log scale)')
    plt.ylabel('Time (log scale)')
    plt.legend()
    plt.grid(True)

    # O(n log h)
    plt.subplot(4, 2, 5)
    plt.plot(n_values, nlogh_times, 'go-', label='Experimental')
    plt.plot(n_values, nlogh_theoretical, 'r--', label='Theoretical (normalized n log h)')
    plt.title('Logarithmic insertion: O(n log h)')
    plt.xlabel('n')
    plt.ylabel('Time (ns)')
    plt.legend()
    plt.grid(True)

    # Log-log plots O(n log h)
    plt.subplot(4, 2, 6)
    plt.loglog(n_values, nlogh_times, 'bo-', label='Experimental')
    plt.loglog(n_values, nlogh_theoretical, 'r--', label='Theoretical (normalized n log h)')
    plt.title('Logarithmic insertion O(n log h)')
    plt.xlabel('n (log scale)')
    plt.ylabel('Time (log scale)')
    plt.legend()
    plt.grid(True)

    # O(n)
    plt.subplot(4, 2, 7)
    plt.plot(n_values, presorted_times, 'go-', label='Experimental')
    plt.plot(n_values, presorted_theoretical, 'r--', label='Theoretical (normalized n)')
    plt.title('Presorted linear: O(n)')
    plt.xlabel('n')
    plt.ylabel('Time (ns)')
    plt.legend()
    plt.grid(True)

    # Log-log plots O(n)
    plt.subplot(4, 2, 8)
    plt.loglog(n_values, presorted_times, 'bo-', label='Experimental')
    plt.loglog(n_values, presorted_theoretical, 'r--', label='Theoretical (normalized n)')
    plt.title('Presorted linear O(n)')
    plt.xlabel('n (log scale)')
    plt.ylabel('Time (log scale)')
    plt.legend()
    plt.grid(True)   

    plt.tight_layout()
    plt.show()

# ---------- Run Experiment ----------
n_values = [100, 200, 400, 800, 1600, 3200, 6400, 12800]

plot_staircase_comparison(n_values)

dc_times, nh_times, nlogh_times, presorted_times, h_values, nh_theory, nlogh_theory = benchmark_algorithms(n_values)
dc_theoretical, nh_theoretical, nlogh_theoretical, presorted_theoretical = normalize_theoretical_curves(
    n_values, dc_times, nh_times, nlogh_times, presorted_times, nh_theory, nlogh_theory)


print_comparison_table(n_values, h_values, dc_times, nh_times, nlogh_times, presorted_times,
                        dc_theoretical, nh_theoretical, nlogh_theoretical, presorted_theoretical)

plot_performance_curves(n_values, dc_times, nh_times, nlogh_times, presorted_times,
                        dc_theoretical, nh_theoretical, nlogh_theoretical, presorted_theoretical)