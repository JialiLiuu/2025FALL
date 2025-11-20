import time

def read_maze(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [list(line.rstrip()) for line in f if line.strip() != ""]

def get_neighbors(r, c, maze, H, W):
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

def dfs_maze_solver(maze):
    H, W = len(maze) - 1, int((len(maze[0]) - 1) / 2)
    start, target = (0, 0), (H - 1, W - 1)

    stack = [start]
    visited = {start}
    parent = {start: None}

    while stack:
        r, c = stack.pop()
        if (r, c) == target:
            break
        for nr, nc in get_neighbors(r, c, maze, H, W):
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                parent[(nr, nc)] = (r, c)
                stack.append((nr, nc))

    # Reconstruct path
    path = []
    cur = target
    while cur:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path

if __name__ == "__main__":
    maze = read_maze("project3/maze1.txt")
    unique_chars = set(char for row in maze for char in row)
    print("Unique characters:", unique_chars)
    start_time = time.time()
    path = dfs_maze_solver(maze)
    end_time = time.time()
    print(f"Path length: {len(path)}")
    print(f"DFS execution time: {end_time - start_time:.6f} seconds")
