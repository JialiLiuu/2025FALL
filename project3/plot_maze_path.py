import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from dfs_maze_solver import read_maze, dfs_maze_solver

maze = read_maze("project3/maze2.txt")
path = dfs_maze_solver(maze)

H, W = len(maze) - 1, int((len(maze[0]) - 1) / 2)

# Create numeric grid: 0 = open space, 1 = path
grid = np.zeros((H, W), dtype=int)
for r, c in path:
    grid[r, c] = 1

# Define colors: open=white, path=green
cmap = ListedColormap(["white", "green"])

plt.figure(figsize=(10, 10))
plt.imshow(grid, cmap=cmap)
plt.title("DFS Path Visualization", fontsize=16)
plt.axis("off")

# Draw walls
ax = plt.gca()
for r in range(H):
    for c in range(W):
        x0, x1 = c - 0.5, c + 0.5
        y0, y1 = r - 0.5, r + 0.5
        if maze[r][c * 2 + 1] != ' ':  # top wall
            ax.plot([x0, x1], [y0, y0], color="black", linewidth=2)
        if maze[r + 1][c * 2 + 1] != ' ':  # bottom wall
            ax.plot([x0, x1], [y1, y1], color="black", linewidth=2)
        if maze[r][c * 2] not in ['─', ' ', '╴', '╶', '┘', '└', '╵', '┴'] or \
           maze[r + 1][c * 2] not in ['┌', '─', '┬', '┐', ' ', '╴', '╷', '╶']:  # left wall
            ax.plot([x0, x0], [y0, y1], color="black", linewidth=2)
        if maze[r][c * 2 + 2] not in ['─', ' ', '╴', '╶', '┘', '└', '╵', '┴'] or \
           maze[r + 1][c * 2 + 2] not in ['┌', '─', '┬', '┐', ' ', '╴', '╷', '╶']:  # right wall
            ax.plot([x1, x1], [y0, y1], color="black", linewidth=2)

# Add step numbers
for step, (r, c) in enumerate(path):
    plt.text(c, r, str(step), color="yellow", ha="center", va="center", fontsize=4, fontweight="bold")

plt.show()