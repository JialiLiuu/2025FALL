# Schöning’s Algorithm Experiment for 3-SAT

### **Runtime Comparison: Empirical Performance vs. (2^n) and ((4/3)^n)**

---

## 📌 Overview

This project implements:

1. **A random satisfiable 3-SAT generator** , biased toward a hidden satisfying assignment.
2. **Schöning’s randomized local search algorithm** for solving 3-SAT.
3. **An empirical experiment** comparing:
   * Actual runtime of Schöning’s algorithm
   * Theoretical brute-force runtime (2^n)
   * Schöning’s theoretical upper bound ((4/3)^n)

The results are normalized and plotted on a logarithmic scale to visualize growth rates.

---

## 📁 File Structure

This file contains four major components:

1. **3-SAT instance generator**
2. **Schöning’s algorithm implementation**
3. **Experiment and runtime measurement**
4. **Visualization and comparison plot**

---

# 1. Random 3-SAT Generator

### Function: `generate_random_3_sat(n_vars, m_clauses)`

This function generates a random 3-SAT formula that is  **likely satisfiable** .

It works by:

* Creating a  **hidden truth assignment** .
* Building each clause by selecting three distinct variables.
* Choosing positive or negative literals with a **70% bias** toward being satisfied by the hidden assignment.

This ensures most clauses are satisfied, making the instance easier (and more realistic) for Schöning’s algorithm.

---

# 2. Utility Functions for Satisfiability Checking

### ✔ `random_assignment(n_vars)`

Returns a random Boolean assignment.

### ✔ `is_clause_satisfied(clause, assignment)`

Evaluates whether a clause is satisfied under the given assignment.

### ✔ `find_unsatisfied_clause(formula, assignment)`

Returns a random clause that is currently unsatisfied.

If all clauses are satisfied → returns `None`.

This function is essential for the **random-walk step** of Schöning’s algorithm.

---

# 3. Schöning’s Randomized Local Search Algorithm

### Function: `schoning_3sat(formula, n_vars, max_restarts=2000, steps_per_restart=None)`

This implements the classic **random walk** algorithm:

1. Repeat for up to `max_restarts`:
   * Start with a random assignment.
   * For up to `3n` steps:
     * If the formula is satisfied → return success.
     * Otherwise choose a random unsatisfied clause.
     * Pick a random literal from that clause and flip its variable.

### Output:

* `True/False` depending on whether a solution is found
* Total runtime in milliseconds

The expected running time for 3-SAT is known to be  **O*((4/3)^n)** .

---

# 4. Running the Experiment

### Function:

`run_experiment_three_curves(n_values, clause_ratio, trials_per_n)`

For each `n` in `n_values`:

1. Generate `trials_per_n` random formulas.
2. Run Schöning’s algorithm and record runtime.
3. Compute the average runtime.
4. Compute theoretical values:
   * (2^n)
   * ((4/3)^n)

### Normalization

The theoretical curves are scaled so that at the smallest `n`:

```
scaled_curve[n0] == actual_runtime[n0]
```

This allows direct comparison of *growth rates* rather than absolute values.

---

# 5. Plotting

The script produces a log-scale plot:

* **Blue line** : empirical runtime of Schöning’s algorithm
* **Orange dashed line** : scaled (2^n) (brute force)
* **Green dashed-dot line** : scaled ((4/3)^n)

This makes it easy to verify that empirical behavior aligns better with ((4/3)^n) than with (2^n).

---

# 6. Main Execution

The block:

```python
if __name__ == "__main__":
    random.seed(0)
    run_experiment_three_curves(...)
```

ensures:

* The experiment runs only when the script is executed directly.
* Using `random.seed(0)` makes the results  **reproducible** .

---

# 📊 Expected Results

The plot will typically show:

* **Actual runtime grows much slower than (2^n)** .
* Growth roughly follows the shape of  **((4/3)^n)** , consistent with theoretical expectations.
* Because of normalization, all three curves start at the same point for the smallest tested `n`.

This confirms that Schöning’s algorithm performs closer to its theoretical exponential bound rather than brute force.

---

# 🧠 Summary

This file provides:

* A working implementation of Schöning’s random walk algorithm.
* A random satisfiable 3-SAT generator.
* A reproducible experiment for runtime comparison.
* A clear visualization that demonstrates theoretical behaviors.

It is suitable for academic presentations on algorithms, SAT solvers, and analysis of randomized algorithms.
