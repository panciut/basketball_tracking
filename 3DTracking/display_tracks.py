import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TRACKS_CSV = os.path.join(OUTPUT_DIR, "3d_tracks.csv")  # uses the provided file structure

# Load player tracks: frame -> list of (id, x, y)
frames = {}
with open(TRACKS_CSV, "r", newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        if r["class"] != "player":
            continue
        try:
            fi = int(r["frame"])
            tid = int(r["track_id"])
            x = float(r["X"]); y = float(r["Y"])
        except:
            continue
        frames.setdefault(fi, []).append((tid, x, y))

frame_idxs = sorted(frames.keys())
if not frame_idxs:
    raise RuntimeError("No player data found in 3d_tracks.csv")

# Determine field bounds from data (pad 5%)
xs = []
ys = []
for flist in frames.values():
    for _, x, y in flist:
        xs.append(x); ys.append(y)
xmin, xmax = np.min(xs), np.max(xs)
ymin, ymax = np.min(ys), np.max(ys)
dx = (xmax - xmin) * 0.05 if xmax > xmin else 1.0
dy = (ymax - ymin) * 0.05 if ymax > ymin else 1.0
xmin -= dx; xmax += dx; ymin -= dy; ymax += dy

# Colors per track id for consistency
def color_for_id(tid):
    rng = np.random.RandomState(tid % (2**32-1))
    return rng.rand(3,)

# Keep short trails per track
max_trail = 30
trails = {}  # tid -> list of (x, y)

fig, ax = plt.subplots(figsize=(10, 6))
scat = ax.scatter([], [], s=60, c=[])
texts = []
title = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", ha="left")
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Player positions (X,Y)")

def init():
    scat.set_offsets(np.empty((0,2)))
    global texts
    for t in texts:
        t.remove()
    texts = []
    title.set_text("")
    return [scat, title]

def update(frame_idx):
    global texts
    # Clear previous labels
    for t in texts:
        t.remove()
    texts = []

    pts = frames.get(frame_idx, [])
    if not pts:
        scat.set_offsets(np.empty((0,2)))
        scat.set_color([])
        title.set_text(f"Frame: {frame_idx}")
        return [scat, title]

    coords = []
    colors = []
    for tid, x, y in pts:
        coords.append([x, y])
        colors.append(color_for_id(tid))
        # Update trails
        trails.setdefault(tid, [])
        trails[tid].append((x, y))
        if len(trails[tid]) > max_trail:
            trails[tid] = trails[tid][-max_trail:]

    scat.set_offsets(np.array(coords))
    scat.set_color(colors)

    # Draw labels next to points
    for (tid, x, y), col in zip(pts, colors):
        txt = ax.text(x+0.01*(xmax-xmin), y+0.01*(ymax-ymin), f"{tid}", color=col, fontsize=9, weight="bold")
        texts.append(txt)

    # Draw trails
    # Remove existing trail artists by clearing and redrawing the axes background elements
    # Simpler approach: draw trails as Line2D each frame
    # First, remove previous trail lines by filtering artists added after init
    # To keep it simple and performant, we'll draw trails as a single PathCollection-like scatter with alpha
    # But for clarity, draw lines per trail:
    # Clean previous lines: remove lines added in prior frame (except patch, spines)
    for line in [obj for obj in ax.get_lines()]:
        obj_label = getattr(line, "_is_trail", False)
        if obj_label:
            line.remove()

    for tid, hist in trails.items():
        if len(hist) < 2:
            continue
        xs_t, ys_t = zip(*hist)
        line, = ax.plot(xs_t, ys_t, '-', color=color_for_id(tid), alpha=0.6, linewidth=2)
        line._is_trail = True  # mark for removal next frame

    title.set_text(f"Frame: {frame_idx}")
    return [scat, title] + texts + [obj for obj in ax.get_lines() if getattr(obj, "_is_trail", False)]

anim = FuncAnimation(fig, update, frames=frame_idxs, init_func=init, blit=False, interval=50, repeat=False)
plt.show()
