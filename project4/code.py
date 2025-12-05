import random
import time
import math
import statistics
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

# ----------------------------------------------------------
# 1. Random 3-SAT generator
# ----------------------------------------------------------

def generate_random_3_sat(n_vars: int, m_clauses: int) -> Tuple[List[List[int]], List[bool]]:
    """
    Generate a random satisfiable 3-SAT instance.
    We create a hidden assignment and generate clauses that are (very likely) satisfied by it.
    """
    hidden = [False] + [bool(random.getrandbits(1)) for _ in range(n_vars)]
    formula: List[List[int]] = []

    for _ in range(m_clauses):
        clause = []
        vars_in_clause = random.sample(range(1, n_vars + 1), 3)
        for v in vars_in_clause:
            # Bias literals so that the hidden assignment tends to satisfy the clause
            if hidden[v]:
                lit = v if random.random() < 0.7 else -v
            else:
                lit = -v if random.random() < 0.7 else v
            clause.append(lit)
        formula.append(clause)

    return formula, hidden


# ----------------------------------------------------------
# 2. Schöning's algorithm for 3-SAT
# ----------------------------------------------------------

def random_assignment(n_vars: int) -> List[bool]:
    """Generate a random Boolean assignment."""
    return [False] + [bool(random.getrandbits(1)) for _ in range(n_vars)]


def is_clause_satisfied(clause: List[int], assignment: List[bool]) -> bool:
    """Check whether a clause is satisfied under the given assignment."""
    for lit in clause:
        v = abs(lit)
        if (lit > 0 and assignment[v]) or (lit < 0 and not assignment[v]):
            return True
    return False


def find_unsatisfied_clause(formula: List[List[int]], assignment: List[bool]) -> Optional[List[int]]:
    """Return a random unsatisfied clause, or None if all clauses are satisfied."""
    unsat = [c for c in formula if not is_clause_satisfied(c, assignment)]
    return random.choice(unsat) if unsat else None


def schoning_3sat(
    formula: List[List[int]],
    n_vars: int,
    max_restarts: int = 2000,
    steps_per_restart: Optional[int] = None
) -> Tuple[bool, float]:
    """
    Implementation of Schöning's random walk algorithm for 3-SAT.
    Returns:
      (found, runtime_ms)
    """
    if steps_per_restart is None:
        # The paper suggests 3n steps per restart for 3-SAT
        steps_per_restart = 3 * n_vars

    start = time.perf_counter()

    for _ in range(max_restarts):
        assignment = random_assignment(n_vars)
        for _ in range(steps_per_restart):
            clause = find_unsatisfied_clause(formula, assignment)
            if clause is None:
                end = time.perf_counter()
                return True, (end - start) * 1000.0  # milliseconds
            lit = random.choice(clause)
            v = abs(lit)
            assignment[v] = not assignment[v]

    end = time.perf_counter()
    return False, (end - start) * 1000.0


# ----------------------------------------------------------
# 3. Experiment: actual runtime vs 2^n vs (4/3)^n
# ----------------------------------------------------------

def run_experiment_three_curves(
    n_values = (20, 24, 28, 32, 36, 40),
    clause_ratio: float = 4.2,
    trials_per_n: int = 8
):
    """
    For each n in n_values:
      - Generate random satisfiable 3-SAT instances
      - Measure the average runtime of Schöning's algorithm
      - Compute theoretical curves 2^n and (4/3)^n
    Then:
      - Normalize the 2^n and (4/3)^n curves so they start at the same value as the actual runtime.
      - Plot all three curves on the same figure
    """
    actual_times: List[float] = []
    theo_2n: List[float] = []
    theo_schoening: List[float] = []  # (4/3)^n

    print("\n=== Running experiments for 3-SAT ===\n")

    for n in n_values:
        m = int(clause_ratio * n)
        print(f"n = {n}, clauses = {m}")

        # Measure empirical runtime
        times = []
        for _ in range(trials_per_n):
            formula, _ = generate_random_3_sat(n, m)
            found, runtime_ms = schoning_3sat(formula, n)
            times.append(runtime_ms)

        avg_time = sum(times) / len(times)
        actual_times.append(avg_time)

        # Theoretical curves
        theo_2n.append(2 ** n)
        theo_schoening.append((4.0 / 3.0) ** n)  # base from Schöning's bound for 3-SAT

        print(f"  avg runtime = {avg_time:.3f} ms\n")

    # ------------------------------------------------------
    # Normalization
    # ------------------------------------------------------
    # Goal: align the theoretical curves with the empirical runtime at the first data point,
    # so that all curves start from the same value and we can compare their growth rates.


    scale_2n = actual_times[0] / theo_2n[0]
    scale_schoening = actual_times[0] / theo_schoening[0]

    scaled_2n = [val * scale_2n for val in theo_2n]
    scaled_schoening = [val * scale_schoening for val in theo_schoening]

    # ------------------------------------------------------
    # Plot the three curves
    # ------------------------------------------------------
    plt.figure(figsize=(10, 6))

    plt.plot(n_values, actual_times, marker='o', label="Actual runtime (ms)")
    plt.plot(n_values, scaled_2n, marker='s', linestyle='--', label="Scaled 2^n (brute-force growth)")
    plt.plot(n_values, scaled_schoening, marker='^', linestyle='-.', label="Scaled (4/3)^n (Schöning bound)")

    plt.yscale("log") 
    plt.title("3-SAT: Actual Runtime vs 2^n vs (4/3)^n (normalized)")
    plt.xlabel("Number of variables n")
    plt.ylabel("Runtime (ms) / Scaled theoretical value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

if __name__ == "__main__":
    random.seed(0)
    run_experiment_three_curves(
        n_values=(20, 24, 28, 32, 36, 40),
        clause_ratio=4.2,
        trials_per_n=8
    )
