# app.py
"""
Maze Visualizer backend (Flask).
Endpoints:
  POST /api/generate_maze  -> { size: int }  returns { maze: [[0/1/2/3]], size }
  POST /api/solve_maze     -> { maze: [...], algorithm: 'bfs'|'dfs' } returns
                             { explored: int, path: [[x,y]..], time: ms, visited_steps: [[x,y]..] }
Run locally:
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  python app.py
"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import random
import time
import os

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Cell codes
WALL = 1
PATH = 0
START = 2
END = 3

def generate_perfect_maze(size):
    """
    Generate a perfect maze using recursive backtracker on an odd-size grid.
    Grid returned where:
      1 = wall
      0 = passage
      2 = start (1,1)
      3 = end (size-2,size-2)
    """
    # Ensure odd
    if size % 2 == 0:
        size += 1
    # Initialize full wall grid
    maze = [[WALL for _ in range(size)] for _ in range(size)]

    # carve passages at odd coordinates
    def neighbors(cx, cy):
        dirs = [(0, -2), (2, 0), (0, 2), (-2, 0)]
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = cx + dx, cy + dy
            if 0 < nx < size - 1 and 0 < ny < size - 1:
                yield nx, ny, dx // 2, dy // 2

    stack = []
    start_x, start_y = 1, 1
    maze[start_y][start_x] = PATH
    stack.append((start_x, start_y))

    while stack:
        x, y = stack[-1]
        found = False
        for nx, ny, wx_off, wy_off in neighbors(x, y):
            if maze[ny][nx] == WALL:
                # carve wall between
                maze[y + wy_off][x + wx_off] = PATH
                maze[ny][nx] = PATH
                stack.append((nx, ny))
                found = True
                break
        if not found:
            stack.pop()

    # set start & end
    maze[1][1] = START
    maze[size - 2][size - 2] = END
    return maze

# --- Solvers (return visited steps in order and final path) --- #
def solve_bfs(maze):
    size = len(maze)
    from collections import deque
    start = (1, 1)
    target = (size - 2, size - 2)
    q = deque([start])
    came_from = {start: None}
    visited_steps = []
    explored = 0
    while q:
        cur = q.popleft()
        explored += 1
        visited_steps.append([cur[0], cur[1]])
        if cur == target:
            break
        x, y = cur
        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and maze[ny][nx] != WALL:
                if (nx, ny) not in came_from:
                    came_from[(nx, ny)] = cur
                    q.append((nx, ny))
    # reconstruct path
    path = []
    cur = target
    if cur in came_from:
        while cur:
            path.append([cur[0], cur[1]])
            cur = came_from[cur]
        path.reverse()
    return explored, path, visited_steps

def solve_dfs(maze):
    size = len(maze)
    start = (1, 1)
    target = (size - 2, size - 2)
    visited = set()
    visited_steps = []
    path = []
    explored = 0
    found = False

    def in_bounds(x,y):
        return 0 <= x < size and 0 <= y < size

    def dfs(x,y):
        nonlocal explored, found
        if found:
            return
        if not in_bounds(x,y) or maze[y][x] == WALL or (x,y) in visited:
            return
        visited.add((x,y))
        explored += 1
        visited_steps.append([x,y])
        if (x,y) == target:
            path.append([x,y])
            found = True
            return
        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
            nx, ny = x + dx, y + dy
            dfs(nx, ny)
            if found:
                path.append([x,y])
                return

    dfs(start[0], start[1])
    path.reverse()
    return explored, path, visited_steps

# --- Flask API routes --- #
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generate_maze', methods=['POST'])
def api_generate_maze():
    payload = request.get_json() or {}
    size = int(payload.get('size', 21))
    size = max(9, min(size, 101))  # constrain
    maze = generate_perfect_maze(size)
    return jsonify({'maze': maze, 'size': size})

@app.route('/api/solve_maze', methods=['POST'])
def api_solve_maze():
    payload = request.get_json() or {}
    maze = payload.get('maze')
    algorithm = (payload.get('algorithm') or 'bfs').lower()
    if not maze:
        return jsonify({'error': 'maze data required'}), 400

    # Convert maze to internal format (list of lists) - assume same coords order as frontend
    start_time = time.time()
    if algorithm == 'dfs':
        explored, path, visited_steps = solve_dfs(maze)
    else:
        explored, path, visited_steps = solve_bfs(maze)
    time_ms = int((time.time() - start_time) * 1000)
    return jsonify({
        'explored': explored,
        'path': path,
        'time': time_ms,
        'visited_steps': visited_steps
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
