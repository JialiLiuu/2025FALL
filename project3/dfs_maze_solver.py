"""
Filename: dfs_maze_solver.py
Author: Jiali Liu
Date: 2025-11-20
Description:
This module implements a Depth-First Search (DFS) algorithm to solve ASCII-based mazes.
It provides functionality to:
1. Read maze layouts from text files.
2. Identify valid neighboring cells based on maze characters.
3. Compute a path from the start (top-left) to the target (bottom-right) using DFS.
4. Print the solution path in a condensed APA-style format for readability.

The module also benchmarks the DFS algorithm by measuring execution time and path length
for multiple maze inputs, and displays unique characters found in each maze.
"""

import time

def read_maze(file_path):
    """
    Reads a maze from a text file and returns it as a 2D list of characters.
    """
    # Open the file and read each line, stripping trailing spaces.
    # Ignore empty lines and convert each line into a list of characters
    with open(file_path, "r", encoding="utf-8") as f:
        return [list(line.rstrip()) for line in f if line.strip() != ""]

def get_neighbors(r, c, maze, H, W):
    """
    Generates valid neighboring cells for a given position in the maze.
    
    Args:
        r (int): Current row index.
        c (int): Current column index.
        maze (list[list[str]]): The maze grid.
        H (int): Height of the maze in terms of cells.
        W (int): Width of the maze in terms of cells.
    
    Yields:
        tuple[int, int]: Coordinates of a valid neighboring cell.
    """
    # Check right neighbor
    nr,nc=r,c + 1
    if 0<=nr<H and 0<=nc<W and maze[r][c * 2 + 2] in ['─', ' ', '╴', '╶', '┘', '└', '╵', '┴'] and \
        maze[r + 1][c * 2 + 2] in ['┌', '─', '┬', '┐', ' ', '╴', '╷', '╶']:
        yield nr, nc
    # Check down neighbor
    nr,nc=r + 1,c
    if 0<=nr<H and 0<=nc<W and maze[r + 1][c * 2 + 1] == ' ':
        yield nr, nc
    # Check left neighbor
    nr,nc=r,c - 1
    if 0<=nr<H and 0<=nc<W and maze[r][c * 2] in ['─', ' ', '╴', '╶', '┘', '└', '╵', '┴'] and \
        maze[r + 1][c * 2] in ['┌', '─', '┬', '┐', ' ', '╴', '╷', '╶']:
        yield nr, nc
    # Check up neighbor
    nr,nc=r - 1,c
    if 0<=nr<H and 0<=nc<W and maze[r][c * 2 + 1] == ' ':
        yield nr, nc

def dfs_maze_solver(maze):
    """
    Solves the maze using Depth-First Search (DFS) and returns the path.
    
    Args:
        maze (list[list[str]]): The maze grid.
    
    Returns:
        list[tuple[int, int]]: The path from start to target as a list of coordinates.
    """
    # Calculate maze dimensions in terms of cells
    H, W = len(maze) - 1, int((len(maze[0]) - 1) / 2)
    
    # Define start and target positions
    start, target = (0, 0), (H - 1, W - 1)

    # Initialize DFS structures
    stack = [start]
    visited = {start}
    parent = {start: None}

    # Perform DFS until target is found or stack is empty
    while stack:
        r, c = stack.pop()
        if (r, c) == target:
            break
        for nr, nc in get_neighbors(r, c, maze, H, W):
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                parent[(nr, nc)] = (r, c)
                stack.append((nr, nc))

    # # Reconstruct path from target to start using parent dictionary
    path = []
    cur = target
    while cur:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path

def print_condensed_path_apa(path):
    """
    Prints the path in APA-style format, condensing if too long.
    """
    total = len(path)

    # If the path is short, just print normally
    if total <= 40:
        print("Path:", " → ".join(str(p) for p in path))
        return

    # Otherwise, print first 20 and last 20 steps with ellipsis
    first_part = path[:20]
    last_part  = path[-20:]

    # APA-style single line with ellipsis
    output = (
        "Path = " +
        " → ".join(str(p) for p in first_part) +
        " → ... → " +
        " → ".join(str(p) for p in last_part)
    )

    print(output)


if __name__ == "__main__":
    # Solve maze1
    maze = read_maze("project3/maze1.txt")
    unique_chars = set(char for row in maze for char in row)
    print("Unique characters (maze1):", unique_chars)
    start_time = time.time()
    path = dfs_maze_solver(maze)
    print_condensed_path_apa(path)
    end_time = time.time()
    print(f"Path length (maze1): {len(path)}")
    print(f"DFS execution time (maze1): {end_time - start_time:.6f} seconds")

    # Solve maze2
    maze = read_maze("project3/maze2.txt")
    unique_chars = set(char for row in maze for char in row)
    print("Unique characters (maze2):", unique_chars)
    start_time = time.time()
    path = dfs_maze_solver(maze)
    print_condensed_path_apa(path)
    end_time = time.time()
    print(f"Path length (maze2): {len(path)}")
    print(f"DFS execution time (maze2): {end_time - start_time:.6f} seconds")
