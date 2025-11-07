#!/usr/bin/env python3
"""
Player Plane Viewer (Actual-Only)
---------------------------------

Visualize *only the actual* players' detections on a 2D ground plane.
Tracks that are lost/occluded or interpolated are NOT drawn.
Each frame shows only that frame's valid points.

What counts as "actual":
- history entry has no 'interp' flag or interp == False
- and has no 'lost' flag or lost == False
- and has no 'visible' flag or visible == True
(Unknown flags default to "actual" to be robust to different schemas.)

Usage
-----
python player_plane_viewer.py /path/to/tracking3d_output.json

Controls
--------
- Space: toggle Play/Pause
- Left/Right arrows: -25 / +25 frames
- Slider: jump to a specific frame
- Buttons: ⟵ 25, Play/Pause, ⟶ 25

Notes
-----
- Only items with label == "player" are plotted.
- X/Y are taken from the first two components of "x". Z is ignored for the 2D plane view.
"""

TRACKING_JSON_PATH = "output/tracking3d_output.json"

import sys
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation

def _is_actual(entry: dict) -> bool:
    """Return True if the history entry represents an 'actual' (non-lost, non-interpolated) point."""
    # Interpolated points are not 'actual'
    if bool(entry.get("interp", False)):
        return False
    # Explicitly lost/occluded points are not 'actual'
    if bool(entry.get("lost", False)):
        return False
    # If a 'visible' flag exists, require it to be True
    if "visible" in entry and not bool(entry.get("visible", False)):
        return False
    # Otherwise treat as actual
    return True

def load_players(filepath):
    with open(filepath, "r") as f:
        tracks = json.load(f)

    # Only players
    players = [tr for tr in tracks if str(tr.get("label", "")).lower() == "player"]

    frame_points = defaultdict(list)  # frame -> list of (track_id, x, y)
    xs, ys, frames = [], [], []

    for tr in players:
        tid = tr.get("track_id")
        for h in tr.get("history", []):
            if not _is_actual(h):
                continue  # skip non-actual points
            fr = int(h.get("frame"))
            pos = h.get("x", [None, None])
            if pos is None or len(pos) < 2:
                continue
            x, y = float(pos[0]), float(pos[1])
            frame_points[fr].append((tid, x, y))
            xs.append(x); ys.append(y); frames.append(fr)

    if not frames:
        raise ValueError("No 'actual' player points found in the provided file.")
    min_frame, max_frame = min(frames), max(frames)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    # pad bounds a bit
    dx = (xmax - xmin) * 0.05 or 1.0
    dy = (ymax - ymin) * 0.05 or 1.0
    return frame_points, (min_frame, max_frame), (xmin - dx, xmax + dx, ymin - dy, ymax + dy)

class PlayerPlaneViewer:
    def __init__(self, frame_points, frame_range, bounds, fps=25):
        self.frame_points = frame_points
        self.min_frame, self.max_frame = frame_range
        self.bounds = bounds
        self.current = self.min_frame
        self.playing = False
        self.fps = fps

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(left=0.08, right=0.98, bottom=0.22)

        self.ax.set_title("Players on Ground Plane")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_aspect("equal", adjustable="box")
        xmin, xmax, ymin, ymax = self.bounds
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.grid(True, alpha=0.3, linestyle="--")

        # Scatter for players (red), labels overlayed as small text
        self.scat = self.ax.scatter([], [], s=40, alpha=0.9, c="red", edgecolors="white", linewidths=0.5)
        self.labels = []
        self.frame_text = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes,
                                       ha="left", va="top", fontsize=10,
                                       bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.8))

        # Slider
        slider_ax = self.fig.add_axes([0.10, 0.10, 0.80, 0.04])
        self.slider = Slider(slider_ax, "Frame", self.min_frame, self.max_frame,
                             valinit=self.current, valstep=1)
        self.slider.on_changed(self._on_slider)

        # Buttons
        btn_back_ax = self.fig.add_axes([0.10, 0.02, 0.15, 0.06])
        btn_play_ax = self.fig.add_axes([0.28, 0.02, 0.20, 0.06])
        btn_fwd_ax  = self.fig.add_axes([0.51, 0.02, 0.15, 0.06])

        self.btn_back = Button(btn_back_ax, "⟵ 25")
        self.btn_play = Button(btn_play_ax, "Play ▶")
        self.btn_fwd  = Button(btn_fwd_ax,  "25 ⟶")

        self.btn_back.on_clicked(lambda evt: self.step(-25))
        self.btn_fwd.on_clicked(lambda evt: self.step(+25))
        self.btn_play.on_clicked(self.toggle_play)

        # Keyboard shortcuts
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Animation driver (kept stopped until Play)
        self.ani = FuncAnimation(self.fig, self._tick, interval=int(1000/self.fps))
        self.ani.event_source.stop()

        # Initial draw
        self.render(self.current)

    def _on_key(self, event):
        if event.key == " ":
            self.toggle_play(None)
        elif event.key == "left":
            self.step(-25)
        elif event.key == "right":
            self.step(+25)

    def _on_slider(self, val):
        frame = int(round(val))
        self.current = max(self.min_frame, min(self.max_frame, frame))
        self.render(self.current)

    def toggle_play(self, _event):
        self.playing = not self.playing
        if self.playing:
            self.btn_play.label.set_text("Pause ❚❚")
            self.ani.event_source.start()
        else:
            self.btn_play.label.set_text("Play ▶")
            self.ani.event_source.stop()
        self.fig.canvas.draw_idle()

    def step(self, delta):
        self.current = int(self.current + delta)
        if self.current < self.min_frame:
            self.current = self.min_frame
        if self.current > self.max_frame:
            self.current = self.max_frame
        self.slider.set_val(self.current)  # triggers render

    def _tick(self, _i):
        if not self.playing:
            return
        nxt = self.current + 1
        if nxt > self.max_frame:
            nxt = self.min_frame
        self.current = nxt
        self.slider.set_val(self.current)  # triggers render

    def render(self, frame):
        # Only draw actual points for the *current* frame
        pts = self.frame_points.get(frame, [])
        if pts:
            arr = np.array([[x, y] for (_tid, x, y) in pts], dtype=float)
        else:
            arr = np.empty((0, 2), dtype=float)

        # Update scatter to current frame only
        self.scat.set_offsets(arr)

        # Remove old labels
        for txt in self.labels:
            txt.remove()
        self.labels.clear()

        # Add labels for current frame only
        for (tid, x, y) in pts:
            t = self.ax.text(x, y, str(tid), fontsize=8, color="white",
                             ha="center", va="center",
                             bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.6))
            self.labels.append(t)

        self.frame_text.set_text(f"Frame: {frame}   Players: {len(pts)}")
        self.fig.canvas.draw_idle()

def main():

    frame_points, frame_range, bounds = load_players(TRACKING_JSON_PATH)
    viewer = PlayerPlaneViewer(frame_points, frame_range, bounds, fps=25)
    plt.show()

if __name__ == "__main__":
    main()
