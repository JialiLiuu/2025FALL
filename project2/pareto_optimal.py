import matplotlib.pyplot as plt
import random

def staircase_oh_nh(points):
    """
    Computes Pareto-optimal points in O(nh) time
    where h = number of Pareto-optimal points
    """
    if not points:
        return []
    
    # Sort points by x-coordinate in descending order
    # (so we process from rightmost to leftmost)
    points_sorted = sorted(points, key=lambda p: (-p[0], -p[1]))
    
    staircase = []
    
    for point in points_sorted:
        # Check if current point is dominated by any point in staircase
        dominated = False
        for stair_point in staircase:
            if (stair_point[0] >= point[0] and 
                stair_point[1] >= point[1]):
                dominated = True
                break
        
        # If not dominated, add to staircase
        if not dominated:
            staircase.append(point)
    
    # Sort staircase by x-coordinate for proper staircase order
    staircase.sort(key=lambda p: p[0])
    return staircase

def staircase_dc_optimized(points):
    """
    Computes Pareto-optimal points in O(n log n) time
    using divide and conquer approach
    """
    if not points:
        return []
    
    # Sort points by x-coordinate in ascending order
    points_sorted = sorted(points, key=lambda p: p[0])
    
    def recursive_helper(point_subset):
        """
        Recursively compute Pareto-optimal points for a subset
        """
        # Base case: single point is always Pareto-optimal
        if len(point_subset) == 1:
            return point_subset
        
        # Divide: split into left and right halves
        mid_index = len(point_subset) // 2
        left_half = point_subset[:mid_index]
        right_half = point_subset[mid_index:]
        
        # Conquer: recursively solve subproblems
        left_staircase = recursive_helper(left_half)
        right_staircase = recursive_helper(right_half)
        
        # Combine: merge the two staircases
        return merge_staircases(left_staircase, right_staircase)
    
    def merge_staircases(left_points, right_points):
        """
        Merge left and right Pareto-optimal sets
        """
        # Find maximum y-coordinate in right staircase
        max_y_right = max(point[1] for point in right_points)
        
        # Filter left points: keep only those not dominated by right points
        filtered_left = []
        for left_point in left_points:
            # A left point is dominated if its y <= max_y_right
            # (since all right points have x >= any left point due to sorting)
            if left_point[1] > max_y_right:
                filtered_left.append(left_point)
        
        # Combine filtered left with all right points
        return filtered_left + right_points
    
    # Compute the staircase recursively
    staircase = recursive_helper(points_sorted)
    
    return staircase
    
def plot_staircase_comparison(n_values):
    plt.figure(figsize=(14, 8))
    
    for i, n in enumerate(n_values, start=1):
        
        # This ensures that for each value of n, the random points generated are reproducible
        random.seed(45 + n)
        points = [(random.uniform(0, 1000), random.uniform(0, 1000)) for _ in range(n)]
        
        # Compute staircases
        stair_oh_nh = staircase_oh_nh(points.copy())
        stair_dc = staircase_dc_optimized(points.copy())
        
        # Subplot
        plt.subplot(2, (len(n_values) + 1) // 2, i)
        plt.scatter([p[0] for p in points], [p[1] for p in points],
                    color='gray', alpha=0.5, label='All Points', s=10)
        
        plt.plot([p[0] for p in stair_oh_nh], [p[1] for p in stair_oh_nh],
                 'bo-', linewidth=1.2, markersize=4, label='Iterative: O(nh)')
        
        plt.plot([p[0] for p in stair_dc], [p[1] for p in stair_dc],
                 'ro--', linewidth=1.2, markersize=4, label='Divide & Conquer: O(n log n)')
        
        plt.title(f'n = {n} points')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.legend(fontsize=8)
    
    plt.suptitle('Visual Comparison of Staircase Algorithms for Different Input Sizes', fontsize=14, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# ---------------------------
# Run visualization
# ---------------------------
n_values = [100, 200, 400, 800, 1600, 3200]
plot_staircase_comparison(n_values)
