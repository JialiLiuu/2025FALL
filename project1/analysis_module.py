import time
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Simulate a computational workload and measure
def analyze_code(n):
    # Initialize two arrays of size n+1 filled with 1s
    a = [1] * (n + 1)
    b = [1] * (n + 1)
    Sum = 0

    start_time = time.time()

    # Outer loop with non-linear increment
    j = 2
    while j < n:
        k = 2
        # Inner loop with non-linear increment using square
        while k < n:
            Sum += a[int(k)] * b[int(k)]    # Convert k to integer for indexing
            k = k * math.sqrt(k)
        j += j // 2     # This is j = j * 1.5

    end_time = time.time()
    return end_time - start_time         # Return elapsed time

# Collect experimental timing and theoretical 
def collect_data(n_values):
    experimental_times = []
    theoretical_complexity = []

    for n in n_values:
        # Measure actual execution time
        time_taken = analyze_code(n)
        experimental_times.append(time_taken)

        # Calculate theoretical complexity: log(n) * log(log(n)
        if n > 2:       # Avoid log(0) or log(negative)
            log_n = math.log(n)
            log_log_n = math.log(log_n) if log_n > 1 else 1
            theoretical_val = log_n * log_log_n
        else:
            theoretical_val = 0
        theoretical_complexity.append(theoretical_val)

        print(f"n = {n}: Time = {time_taken:.6f}s")

    return experimental_times, theoretical_complexity


# Normalize theoretical values using the median ratio of experimental to theoretical
def normalize_with_median_ratio(experimental_times, theoretical_complexity):
    # The denominator(theoretical value) cannot be 0
    ratios = [exp / theo for exp,
              theo in zip(experimental_times, theoretical_complexity) if theo > 0]
    scaling_factor = sorted(ratios)[len(ratios) // 2]   # Median ratio
    normalized = [val * scaling_factor for val in theoretical_complexity]
    print(f"Median scaling factor: {scaling_factor:.2e}")
    return normalized, scaling_factor


# Fit a linear regression model to predict time from theoretical complexity
def fit_linear_regression(theoretical_complexity, experimental_times):
    X = np.array(theoretical_complexity).reshape(-1, 1)      # Reshape for sklearn
    y = np.array(experimental_times)
    # Regression: Fit T(n) ≈ c * log(n)*log(log(n))
    model = LinearRegression()
    model.fit(X, y)
    # Predicted times using regression
    predictions = model.predict(X)
    print(
        f"Best fit equation: Time ≈ {model.coef_[0]:.6f} * log(n)*log(log(n)) + {model.intercept_:.6f}")
    return predictions, model.coef_[0], model.intercept_

# Print a table comparing experimental and theoretical values
def print_comparison_table(n_values, experimental_times, normalized_median, normalized_regression):
    print("\nComparison Table:")
    print("n\t\tExperimental\tNormalized (Median)\tNormalized (Regression)")
    print("-" * 60)
    for i, n in enumerate(n_values):
        print(
            f"{n}\t\t{experimental_times[i]:.6f}\t{normalized_median[i]:.6f}\t{normalized_regression[i]:.6f}")

# Plot experimental and theoretical values using different scales
def plot_results(n_values, experimental_times, normalized_median, normalized_regression):
    # Plot results
    plt.figure(figsize=(9, 8))

    # Linear scale plot (Median Ratio)
    plt.subplot(3, 2, 1)
    plt.plot(n_values, experimental_times, 'bo-', label='Experimental')
    plt.plot(n_values, normalized_median,
         'ro--', label='Theoretical (Normalized)')
    plt.xlabel('n')
    plt.ylabel('Time (seconds)')
    plt.title('Time Complexity: Linear Scale (Median Ratio)')
    plt.legend()
    plt.grid(True)

    # Semi-log scale plot (Median Ratio)
    plt.subplot(3, 2, 3)
    plt.semilogx(n_values, experimental_times, 'bo-', label='Experimental')
    plt.semilogx(n_values, normalized_median,
             'ro--', label='Theoretical (Normalized)')
    plt.xlabel('n (log scale)')
    plt.ylabel('Time (seconds)')
    plt.title('Time Complexity: Semi-Log Scale (Median Ratio)')
    plt.legend()
    plt.grid(True)

    # Log-log scale plot (Median Ratio)
    plt.subplot(3, 2, 5)
    plt.loglog(n_values, experimental_times, 'bo-', label='Experimental')
    plt.loglog(n_values, normalized_median,
           'ro--', label='Theoretical (Normalized)')
    plt.xlabel('n (log scale)')
    plt.ylabel('Time (log scale)')
    plt.title('Time Complexity: Log-Log Scale (Median Ratio)')
    plt.legend()
    plt.grid(True)

    # Linear scale plot (Linear Regression)
    plt.subplot(3, 2, 2)
    plt.plot(n_values, experimental_times, 'bo-', label='Experimental')
    plt.plot(n_values, normalized_regression,
         'ro--', label='Theoretical (Normalized)')
    plt.xlabel('n')
    plt.ylabel('Time (seconds)')
    plt.title('Time Complexity: Linear Scale (Linear Regression)')
    plt.legend()
    plt.grid(True)

    # Semi-log scale plot (Linear Regression)
    plt.subplot(3, 2, 4)
    plt.semilogx(n_values, experimental_times, 'bo-', label='Experimental')
    plt.semilogx(n_values, normalized_regression,
             'ro--', label='Theoretical (Normalized)')
    plt.xlabel('n (log scale)')
    plt.ylabel('Time (seconds)')
    plt.title('Time Complexity: Semi-Log Scale (Linear Regression)')
    plt.legend()
    plt.grid(True)

    # Log-log scale plot (Linear Regression)
    plt.subplot(3, 2, 6)
    plt.loglog(n_values, experimental_times, 'bo-', label='Experimental')
    plt.loglog(n_values, normalized_regression,
           'ro--', label='Theoretical (Normalized)')
    plt.xlabel('n (log scale)')
    plt.ylabel('Time (log scale)')
    plt.title('Time Complexity: Log-Log Scale (Linear Regression)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Main function to run the full analysis pipeline
def run_analysis(values):
    n_values = values
    experimental_times, theoretical_complexity = collect_data(n_values)
    normalized_median, _ = normalize_with_median_ratio(
        experimental_times, theoretical_complexity)
    normalized_regression, _, _ = fit_linear_regression(
        theoretical_complexity, experimental_times)
    print_comparison_table(n_values, experimental_times,
                           normalized_median, normalized_regression)
    plot_results(n_values, experimental_times,
                 normalized_median, normalized_regression)

# Example usage
if __name__ == "__main__":
    # Define input values to test
    inputs = [1000, 5000, 10000, 50000, 100000,
            500000, 1000000, 5000000, 10000000]
    run_analysis(inputs)
