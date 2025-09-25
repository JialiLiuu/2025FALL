import time
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def analyze_code(n):
    # Dummy arrays for the computation
    a = [1] * (n + 1)
    b = [1] * (n + 1)
    Sum = 0
    
    start_time = time.time()
    
    j = 2
    while j < n:
        k = 2
        while k < n:
            Sum += a[int(k)] * b[int(k)]  # Convert k to integer for indexing
            k = k * math.sqrt(k)
        j += j // 2  # This is j = j * 1.5
    
    end_time = time.time()
    return end_time - start_time

# Test different values of n (need larger values to see the pattern)
n_values = [1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]
experimental_times = []
theoretical_complexity = []

for n in n_values:
    time_taken = analyze_code(n)
    experimental_times.append(time_taken)
    
    # Theoretical complexity: log(n) × log(log(n))
    if n > 2:  # Avoid log(0) or log(negative)
        log_n = math.log(n)
        log_log_n = math.log(log_n) if log_n > 1 else 1
        theoretical_val = log_n * log_log_n
        theoretical_complexity.append(theoretical_val)
    else:
        theoretical_complexity.append(0)
    
    print(f"n = {n}: Time = {time_taken:.6f}s")

# Use Median Ratio!!
# Calculate ratio for each n-value pair
ratios = []
for i in range(len(n_values)):
    if theoretical_complexity[i] > 0:
        ratio = experimental_times[i] / theoretical_complexity[i]
        ratios.append(ratio)
        print(f"n = {n_values[i]}: ratio = {ratio:.2e}")

# Use median ratio as scaling factor
scaling_factor = sorted(ratios)[len(ratios) // 2]
print(f"Median scaling factor: {scaling_factor:.2e}")
normalized_theoretical_median = [val * scaling_factor for val in theoretical_complexity]

# Use Linear Regression!!
# Compute theoretical complexity values, the .reshape(-1, 1) command format the data as "many samples, one feature," 
experimental_times_regression = np.array(experimental_times)
theoretical_complexity_regression = np.array(theoretical_complexity).reshape(-1, 1)

# Regression: Fit T(n) ≈ c * log(n)*log(log(n))
model = LinearRegression()
model.fit(theoretical_complexity_regression, experimental_times_regression)

c = model.coef_[0]     # Best-fit scaling factor
intercept = model.intercept_

print(f"Best fit equation: Time ≈ {c:.6f} * log(n)*log(log(n)) + {intercept:.6f}")

# Predicted times using regression
normalized_theoretical_regression = model.predict(theoretical_complexity_regression)

# Compare experimental vs predicted
for n, exp, pred in zip(n_values, experimental_times_regression, normalized_theoretical_regression):
    print(f"n={n}, Experimental={exp:.6f}, Predicted={pred:.6f}")



# Create comparison table
print("\nComparison Table:")
print("n\t\tExperimental\tNormalized Theoretical(Median Ratio)\tNormalized Theoretical(Linear Regression)")
print("-" * 50)
for i, n in enumerate(n_values):
    print(f"{n}\t\t{experimental_times[i]:.6f}\t{normalized_theoretical_median[i]:.6f}\t{normalized_theoretical_regression[i]:.6f}")

# Plot results
plt.figure(figsize=(12, 6))

# Linear scale plot (Median Ratio)
plt.subplot(3, 2, 1)
plt.plot(n_values, experimental_times, 'bo-', label='Experimental')
plt.plot(n_values, normalized_theoretical_median, 'ro--', label='Theoretical (Normalized)')
plt.xlabel('n')
plt.ylabel('Time (seconds)')
plt.title('Time Complexity: Linear Scale (Median Ratio)')
plt.legend()
plt.grid(True)

# Semi-log scale plot (Median Ratio)
plt.subplot(3, 2, 3)
plt.semilogx(n_values, experimental_times, 'bo-', label='Experimental')
plt.semilogx(n_values, normalized_theoretical_median, 'ro--', label='Theoretical (Normalized)')
plt.xlabel('n (log scale)')
plt.ylabel('Time (seconds)')
plt.title('Time Complexity: Semi-Log Scale (Median Ratio)')
plt.legend()
plt.grid(True)

# Log-log scale plot (Median Ratio)
plt.subplot(3, 2, 5)
plt.loglog(n_values, experimental_times, 'bo-', label='Experimental')
plt.loglog(n_values, normalized_theoretical_median, 'ro--', label='Theoretical (Normalized)')
plt.xlabel('n (log scale)')
plt.ylabel('Time (log scale)')
plt.title('Time Complexity: Log-Log Scale (Median Ratio)')
plt.legend()
plt.grid(True)

# Linear scale plot (Linear Regression)
plt.subplot(3, 2, 2)
plt.plot(n_values, experimental_times, 'bo-', label='Experimental')
plt.plot(n_values, normalized_theoretical_regression, 'ro--', label='Theoretical (Normalized)')
plt.xlabel('n')
plt.ylabel('Time (seconds)')
plt.title('Time Complexity: Linear Scale (Linear Regression)')
plt.legend()
plt.grid(True)

# Semi-log scale plot (Linear Regression)
plt.subplot(3, 2, 4)
plt.semilogx(n_values, experimental_times, 'bo-', label='Experimental')
plt.semilogx(n_values, normalized_theoretical_regression, 'ro--', label='Theoretical (Normalized)')
plt.xlabel('n (log scale)')
plt.ylabel('Time (seconds)')
plt.title('Time Complexity: Semi-Log Scale (Linear Regression)')
plt.legend()
plt.grid(True)

# Log-log scale plot (Linear Regression)
plt.subplot(3, 2, 6)
plt.loglog(n_values, experimental_times, 'bo-', label='Experimental')
plt.loglog(n_values, normalized_theoretical_regression, 'ro--', label='Theoretical (Normalized)')
plt.xlabel('n (log scale)')
plt.ylabel('Time (log scale)')
plt.title('Time Complexity: Log-Log Scale (Linear Regression)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()