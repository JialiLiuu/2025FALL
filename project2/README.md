# Pareto Staircase Analysis

## Overview
This project benchmarks and visualizes the performance of multiple algorithms for computing **Pareto-optimal points** (also called the **Pareto frontier** or **staircase**) in a 2D space.  
A point is Pareto-optimal if no other point dominates it in both dimensions (x and y).

The analysis includes:
- Generating random 2D floating-point points in the range [1, 1000]
- Computing Pareto staircases using different algorithms
- Measuring execution time in nanoseconds
- Comparing experimental performance with theoretical time complexities
- Visualizing results using **linear** and **log-log** scale plots
---

## Modules

### `pareto_optimal.py`
Implements four algorithms for computing Pareto-optimal points:

| Function Name             | Description                                                                 | Time Complexity |
|---------------------------|-----------------------------------------------------------------------------|-----------------|
| `staircase_oh_nh`        | Iterative method that checks each point against the current staircase       | **O(nh)**       |
| `staircase_dc_optimized` | Divide-and-conquer approach that recursively merges Pareto-optimal subsets  | **O(n log n)**  |
| `staircase_nlogh`        | Binary search-based method using bisect for efficient insertion             | **O(n log h)**  |
| `staircase_presorted`    | Linear-time method assuming input is sorted by x-coordinate                 | **O(n)**        |

---

### `analysis_module.py`
Provides benchmarking and visualization tools:
- Generates random 2D points for increasing input sizes:  
  `n_values = [100, 200, 400, 800, 1600, 3200, 6400, 12800]`
- Computes Pareto staircases using selected algorithms
- Measures execution time using `time.perf_counter_ns()`
- Normalizes theoretical complexity curves using median scaling
- Produces visual plots:
  - **Linear scale performance comparison**
  - **Log-log scale performance comparison**
- Validates correctness by comparing outputs of different algorithms

---

## Usage Instructions
1. Ensure both `pareto_optimal.py` and `analysis_module.py` are in the same directory.
2. Install required dependencies:
   ```bash
   pip install matplotlib
  