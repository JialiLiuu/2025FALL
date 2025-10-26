"""
Filename: pareto_optimal.py
Author: jiali liu
Date: 2025-10-25
Description: 
This module implements multiple algorithms for computing the Pareto-optimal set (also known as the staircase) from a list of 2D points.
A point is Pareto-optimal if no other point dominates it in both dimensions (x and y).
"""
def staircase_oh_nh(points):
    """
    Computes Pareto-optimal points in O(nh) time
    where h = number of Pareto-optimal points
    """
    if not points:
        return []
    
    # Sort points by x-coordinate in descending order
    # (so process from rightmost to leftmost)
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

import bisect

def staircase_nlogh(points):
    """
    Computes Pareto-optimal points in O(n log h) time,
    where h = number of Pareto-optimal points.

    Approach:
    - Sort points by x descending (and y descending for tie-breaker).
    - Maintain a staircase list sorted by x.
    - Use binary search for insertion and dominance checks.
    """
    if not points:
        return []

    # Sort points by x descending, y descending
    points_sorted = sorted(points, key=lambda p: (-p[0], -p[1]))

    staircase = []

    for point in points_sorted:
        # Binary search for insertion position
        idx = bisect.bisect_left(staircase, point)

        # Check if point is dominated by any existing point
        dominated = False
        for stair_point in staircase:
            if stair_point[0] >= point[0] and stair_point[1] >= point[1]:
                dominated = True
                break

        if not dominated:
            staircase.insert(idx, point)

    # Sort staircase by x ascending for final output
    staircase.sort(key=lambda p: p[0])
    return staircase

def staircase_presorted(points):
    """
    Computes Pareto-optimal points in O(n) time,
    assuming points are sorted by x ascending.
    Removes previously dominated points.
    """
    if not points:
        return []

    staircase = []
    for point in points:
        # Remove points dominated by the new point
        while staircase and staircase[-1][1] <= point[1]:
            staircase.pop()
        staircase.append(point)

    return staircase
