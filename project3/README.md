# **Maze Solver and Visualization**

## **Overview**

This project implements a **Depth-First Search (DFS)** algorithm to solve ASCII-based mazes and provides a visualization of the computed path. It consists of two main modules:

1.  **`dfs_maze_solver.py`**
    *   Reads maze layouts from text files.
    *   Identifies valid neighboring cells based on ASCII characters.
    *   Computes a path from the start (top-left) to the target (bottom-right) using DFS.
    *   Prints the solution path in a condensed APA-style format.
    *   Benchmarks execution time and path length for multiple mazes.

2.  **`plot_maze_path.py`**
    *   Visualizes the DFS solution path using **Matplotlib**.
    *   Displays maze walls, open spaces, and step numbers for clarity.
    *   Highlights the path in green and annotates each step in yellow.

***

## **Features**

*   **DFS-based Maze Solver**:
    *   Handles ASCII mazes with complex wall structures.
    *   Efficient path reconstruction using parent pointers.
*   **Performance Benchmarking**:
    *   Measures execution time and path length for multiple maze files.
*   **Visualization**:
    *   Generates a graphical representation of the maze and solution path.
    *   Annotates steps for detailed tracing.

***

## **Project Structure**

    project3/
    │
    ├── dfs_maze_solver.py      # Maze solver and benchmarking
    ├── plot_maze_path.py       # Visualization of DFS path
    ├── maze1.txt               # Sample maze input
    ├── maze2.txt               # Sample maze input
    └── README.md               # Project documentation

***

## **Installation**

### **Requirements**

*   Python 3.8+
*   Required libraries:
    ```bash
    pip install matplotlib numpy
    ```

***

## **Usage**

### **Solve and Benchmark Mazes**

Run the solver:

```bash
python dfs_maze_solver.py
```

Output:

*   Unique characters in the maze.
*   Condensed APA-style path.
*   Path length and execution time.

### **Visualize Maze Path**

Run the visualization:

```bash
python plot_maze_path.py
```

Output:

*   A graphical representation of the maze with:
    *   Green path.
    *   Black walls.
    *   Yellow step numbers.

***

## **Example Output**

*   **Console**:
        Unique characters (maze1): {'┌', '─', '┐', '└', '┘', ' '}
        Path length (maze1): 150
        DFS execution time (maze1): 0.002345 seconds
*   **Visualization**:
    A Matplotlib window showing the maze and DFS path.

***

## **Complexity Analysis**

*   **DFS Algorithm**:
    *   Time Complexity: `O(H * W)` where `H` and `W` are maze dimensions.
    *   Space Complexity: `O(H * W)` for visited and parent tracking.
*   **Visualization**:
    *   Time Complexity: `O(H * W)` for rendering walls and path.

