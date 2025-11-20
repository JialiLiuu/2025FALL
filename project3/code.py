import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

with open("project3/maze2.txt","r",encoding="utf-8") as f:
    maze=[list(line.rstrip()) for line in f if line.strip() != ""]

# Number of squares
H,W=len(maze) - 1, int((len(maze[0]) - 1) / 2)

def get_neighbors(r,c):
    nr,nc=r,c + 1
    if 0<=nr<H and 0<=nc<W and maze[r][c * 2 + 2] in ['─', ' ', '╴', '╶', '┘', '└', '╵', '┴'] and \
        maze[r + 1][c * 2 + 2] in ['┌', '─', '┬', '┐', ' ', '╴', '╷', '╶']:
        yield nr, nc
    nr,nc=r + 1,c
    if 0<=nr<H and 0<=nc<W and maze[r + 1][c * 2 + 1] == ' ':
        yield nr, nc
    nr,nc=r,c - 1
    if 0<=nr<H and 0<=nc<W and maze[r][c * 2] in ['─', ' ', '╴', '╶', '┘', '└', '╵', '┴'] and \
        maze[r + 1][c * 2] in ['┌', '─', '┬', '┐', ' ', '╴', '╷', '╶']:
        yield nr, nc
    nr,nc=r - 1,c
    if 0<=nr<H and 0<=nc<W and maze[r][c * 2 + 1] == ' ':
        yield nr, nc

start = (0, 0)
target= (H - 1, W - 1)

stack=[start]
visited={start}
parent={start:None}

while stack:
    r,c=stack.pop()
    if (r,c)==target:
        break
    for nr,nc in get_neighbors(r,c):
        if (nr,nc) not in visited:
            visited.add((nr,nc))
            parent[(nr,nc)]=(r,c)
            stack.append((nr,nc))

# reconstruct path
path=[]
cur=target
while cur:
    path.append(cur)
    cur=parent[cur]
path.reverse()

# Create numeric grid: 0 = open space, 1 = path
grid = np.zeros((H, W), dtype=int)

# Overlay DFS path
for r, c in path:
    grid[r, c] = 1

# Define colors: open=white, path=green
cmap = ListedColormap(["white", "green"])

# Plot the grid
plt.figure(figsize=(10, 10))
plt.imshow(grid, cmap=cmap)
plt.title("DFS Path Visualization", fontsize=16)
plt.axis("off")

# Draw borders only for walls
ax = plt.gca()
for r in range(grid.shape[0]):
    for c in range(grid.shape[1]):
        # Coordinates for the cell edges
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

# Add step numbers on path cells
for step, (r, c) in enumerate(path):
    plt.text(c, r, str(step), color="yellow", ha="center", va="center", fontsize=4, fontweight="bold")

plt.show()


