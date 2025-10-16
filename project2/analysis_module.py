import random
import time
import math
import matplotlib.pyplot as plt
from pareto_optimal import (
    staircase_oh_nh,
    staircase_dc_optimized
)

# ---------- Experiment ----------
n_values = [100, 200, 400, 800, 1600, 3200, 6400, 12800]
dc_times, nh_times = [], []
h_values, nh_theory = [], []

for n in n_values:
    # This ensures that for each value of n, the random points generated are reproducible
    random.seed(45 + n)
    pts = [(random.uniform(0, 1000), random.uniform(0, 1000)) for _ in range(n)]

    # O(n log n)
    start = time.perf_counter_ns()
    staircase_dc_optimized(pts)
    dc_times.append(time.perf_counter_ns() - start)

    # O(nh) 
    start = time.perf_counter_ns()
    # measure h precisely
    h = len(staircase_oh_nh(pts))
    h_values.append(h)
    nh_theory.append(n * h)    # theoretical O(nh) raw value
    nh_times.append(time.perf_counter_ns() - start)

# ---------- Normalize theoretical curves ----------
nlogn_theory = [n * math.log(n) for n in n_values]

# Normalizes theoretical complexity values using the median ratio of experimental to theoretical values
dc_ratios = [exp / theo for exp, 
          theo in zip(dc_times, nlogn_theory) if theo > 0]
for ratio in dc_ratios:
    print(f"dc_ratio = {ratio:.6f}")
dc_scale = sorted(dc_ratios)[len(dc_ratios) // 2]   # Median ratio

nh_ratios = [exp / theo for exp, 
          theo in zip(nh_times, nh_theory) if theo > 0]
for ratio in nh_ratios:
    print(f"nh_ratio = {ratio:.6f}")
nh_scale = sorted(nh_ratios)[len(nh_ratios) // 2]   # Median ratio

print("Divide & Conquer scaling factor:", dc_scale)
print("Iterative scaling factor:", nh_scale)

dc_theoretical = [dc_scale * v for v in nlogn_theory]
nh_theoretical = [nh_scale * v for v in nh_theory]

# Prints a formatted comparison table of experimental and normalized theoretical values.
print("\nComparison Table (Divide & Conquer):")
print("n\t\th\t\tDC_Experimental(ns)\tNH_Experimental(ns)\tNormalized (DC)\tNormalized (NH)")
print("-" * 100)
for i, n in enumerate(n_values):
    print(
        f"{n}\t\t{h_values[i]}\t\t{dc_times[i]:.6f}\t{nh_times[i]:.6f}\t{dc_theoretical[i]:.6f}\t{nh_theoretical[i]:.6f}")


# ---------- Plot ----------
plt.figure(figsize=(12, 8))

# O(nh)
plt.subplot(2, 2, 1)
plt.plot(n_values, nh_times, 'go-', label='Experimental')
plt.plot(n_values, nh_theoretical, 'r--', label='Theoretical (normalized nh)')
plt.xlabel('n (number of points)')
plt.ylabel('Time (ns)')
plt.title('Iterative Algorithm: O(nh)')
plt.legend()
plt.grid(True)

# O(n log n)
plt.subplot(2, 2, 3)
plt.plot(n_values, dc_times, 'bo-', label='Experimental')
plt.plot(n_values, dc_theoretical, 'r--', label='Theoretical (normalized n log n)')
plt.xlabel('n (number of points)')
plt.ylabel('Time (ns)')
plt.title('Divide & Conquer: O(n log n)')
plt.legend()
plt.grid(True)

# Log-log scale plot O(nh)
plt.subplot(2, 2, 2)
plt.loglog(n_values, nh_times, 'bo-', label='Experimental')
plt.loglog(n_values, nh_theoretical, 'r--', label='Theoretical (normalized nh)')
plt.xlabel('n (log scale)')
plt.ylabel('Time (log scale)')
plt.title('Iterative Algorithm: O(nh) (Log-Log Scale)')
plt.legend()
plt.grid(True)

# Log-log scale plot O(n log n)
plt.subplot(2, 2, 4)
plt.loglog(n_values, dc_times, 'bo-', label='Experimental')
plt.loglog(n_values, dc_theoretical, 'r--', label='Theoretical (normalized n log n)')
plt.xlabel('n (log scale)')
plt.ylabel('Time (log scale)')
plt.title('Divide & Conquer: O(n log n) (Log-Log Scale)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
