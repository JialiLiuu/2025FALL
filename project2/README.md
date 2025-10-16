# Pareto Staircase Analysis

## Overview

This project benchmarks and visualizes the performance of two algorithms for computing Pareto-optimal points in a 2D space:

1. **Iterative Algorithm** (`staircase_oh_nh`) — Time complexity: **O(nh)**
2. **Divide & Conquer Algorithm** (`staircase_dc_optimized`) — Time complexity: **O(n log n)**

The project generates random 2D points, computes Pareto staircases using both algorithms, measures execution time, normalizes theoretical complexity curves, and plots visual comparisons and performance metrics.

---

## Modules

### `pareto_optimal.py`
Contains implementations of two Pareto-optimal algorithms:
- `staircase_oh_nh`: A simple iterative method that checks dominance for each point.
- `staircase_dc_optimized`: A divide-and-conquer approach that recursively computes and merges Pareto-optimal subsets.

### `analysis_module.py`
Provides benchmarking and visualization tools:
- Generates random 2D points.
- Computes Pareto staircases using both algorithms.
- Measures execution time and compares against theoretical complexity.
- Produces visual plots and performance tables.

---

## Usage Instructions

1. Ensure both `pareto_optimal.py` and `analysis_module.py` are in the same directory.
2. Install required dependencies:
   ```bash
   pip install matplotlib
3. Run the analysis:
   ```bash
   python analysis_module.py
   ```
This will generate visual comparisons of the algorithms and print performance metrics to the console.

---

## Dependencies

- Python 3.x
- matplotlib


---

## Example Output

- Visual comparison of Pareto staircases for increasing input sizes.
- Performance graphs (linear and log-log scale).
- Comparison table of experimental vs. theoretical time complexities.