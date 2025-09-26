
# Time Complexity Analysis of Nested Loop Algorithm

## Overview

This Python project analyzes the time complexity of a nested loop algorithm using both **experimental timing** and **theoretical modeling**. It evaluates the performance of the algorithm across various input sizes and compares the results with theoretical expectations using:

- Median ratio normalization
- Linear regression fitting
- Multiple visualization scales

The project also includes **unit testing** using PyUnit (`unittest`) to ensure correctness and reliability.

---

## Features

- Experimental timing of a nested loop algorithm
- Theoretical complexity modeling using $\log(n) \cdot \log(\log(n))$
- Median ratio normalization for scaling theoretical values
- Linear regression to fit experimental data
- Multiple plots:
  - Linear scale
  - Semi-log scale
  - Log-log scale
- Unit testing with PyUnit

---

## Installation

1. **Clone the repository**:



2. **Install dependencies** :

```Shell
   pip install matplotlib numpy scikit-learn
```

## Usage

Run the analysis with predefined input sizes:

```Shell
python analysis_module.py
```

Or customize the input values in your script:

```Python
inputs = [1000, 5000, 10000, 50000, 100000]
run_analysis(inputs)
```

This will:

* Measure execution time for each input size
* Compute theoretical complexity
* Normalize and fit data
* Print a comparison table
* Display plots comparing experimental and theoretical results

---

## Testing

Run the unit tests using PyUnit:

```Shell
python test_analysis.py
```

Tests include:

* Validating output types and values from `analyze_code`
* Checking data collection and normalization
* Verifying regression model predictions

---

## Acknowledgments

* Python Standard Library
* NumPy
* Matplotlib
* scikit-learn
