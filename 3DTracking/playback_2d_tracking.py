#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive 2D playback for multi-object 3D tracking (basketball scenario).
- Shows "player" tokens in red (all same color)
- Shows "referee" tokens in black
- Optionally shows any other labels in light gray
- Play/Pause, Stop, and a draggable timeline slider
- Keyboard controls: Space (play/pause), Left/Right (step), Home/End (jump), Up/Down (speed)
- Mouse: drag the slider handle to seek

USAGE
-----
python playback_2d_tracking.py --input output/tracking3d_output_multi.json

The script attempts to be robust to a few common JSON schemas:
1) { "0": [ { "label": "...", "x": ..., "y": ... }, ... ], "1": [ ... ], ... }
2) { "0": [ { "label": "...", "position": [x, y, z] }, ... ], ... }
3) [ { "frame": 0, "label": "...", "x": ..., "y": ... }, ... ]
4) [ { "frame": 0, "label": "...", "position": [x, y, z] }, ... ]

Only x,y are plotted (z is ignored).
"""
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
import numpy as np

# Limits for displayed detections
MAX_DISPLAY_PLAYERS = 12
MAX_DISPLAY_REFEREES = 2

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def _coerce_label(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip().lower()

def _extract_xy(obj: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Try to extract (x,y) from a detection object with flexible schema."""
    # Direct x,y
    if all(k in obj for k in ("x", "y")):
        try:
            x = float(obj["x"]); y = float(obj["y"])
            return x, y
        except Exception:
            pass
    # x provided as an array (e.g. [x,y,z])
    if "x" in obj and isinstance(obj["x"], (list, tuple)) and len(obj["x"]) >= 2:
        try:
            x = float(obj["x"][0]); y = float(obj["x"][1])
            return x, y
        except Exception:
            pass
    # position: [x, y] or [x, y, z]
    if "position" in obj and isinstance(obj["position"], (list, tuple)) and len(obj["position"]) >= 2:
        try:
            x = float(obj["position"][0]); y = float(obj["position"][1])
            return x, y
        except Exception:
            pass
    # world: {x:..., y:..., z:...}
    if "world" in obj and isinstance(obj["world"], dict):
        try:
            x = float(obj["world"].get("x")); y = float(obj["world"].get("y"))
            if x is not None and y is not None:
                return x, y
        except Exception:
            pass
    # coords / pos keys
    for k in ("coords", "pos", "xy"):
        if k in obj and isinstance(obj[k], (list, tuple)) and len(obj[k]) >= 2:
            try:
                x = float(obj[k][0]); y = float(obj[k][1])
                return x, y
            except Exception:
                pass
    return None

def load_frames(path: str) -> Dict[int, List[Dict[str, Any]]]:
    """
    Return a dict: frame_index -> list of {"x": float, "y": float, "label": str, "id": optional}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    if isinstance(data, dict):
        # Keys might be "0", "1", ...
        for k, detections in data.items():
            try:
                frame_idx = int(k)
            except Exception:
                # maybe a nested dict like {"frames": {...}}
                if k == "frames" and isinstance(detections, dict):
                    for kk, dd in detections.items():
                        try:
                            frame_idx = int(kk)
                        except Exception:
                            continue
                        if isinstance(dd, list):
                            for obj in dd:
                                xy = _extract_xy(obj)
                                if xy is None:
                                    continue
                                lbl = _coerce_label(obj.get("label", ""))
                                frames[frame_idx].append({
                                    "x": xy[0], "y": xy[1], "label": lbl, "id": obj.get("id")
                                })
                    continue
                else:
                    # Skip unknown keys
                    continue

            if isinstance(detections, list):
                for obj in detections:
                    xy = _extract_xy(obj)
                    if xy is None:
                        # Sometimes 3D may be "X/Y/Z"
                        try:
                            x = float(obj.get("X", obj.get("x")))
                            y = float(obj.get("Y", obj.get("y")))
                            if x is not None and y is not None:
                                xy = (x, y)
                        except Exception:
                            pass
                    if xy is None:
                        continue
                    lbl = _coerce_label(obj.get("label", ""))
                    frames[frame_idx].append({
                        "x": xy[0], "y": xy[1], "label": lbl, "id": obj.get("id")
                    })
    elif isinstance(data, list):
        # Track-centric schema: [{"track_id": ..., "label": "...", "history": [...]}, ...]
        if data and all(isinstance(row, dict) and "history" in row for row in data):
            for track in data:
                if not isinstance(track, dict):
                    continue
                lbl = _coerce_label(track.get("label", ""))
                tid = track.get("track_id", track.get("id"))
                history = track.get("history", [])
                if not isinstance(history, list):
                    continue
                for step in history:
                    if not isinstance(step, dict):
                        continue
                    frame_idx = step.get("frame", step.get("frame_id"))
                    if frame_idx is None:
                        continue
                    try:
                        frame_idx = int(frame_idx)
                    except Exception:
                        continue
                    xy = _extract_xy(step)
                    if xy is None:
                        continue
                    entry = {"x": xy[0], "y": xy[1], "label": lbl, "id": tid}
                    if "interp" in step:
                        entry["interp"] = bool(step["interp"])
                    frames[frame_idx].append(entry)
        else:
            # List of per-detection or per-frame rows
            for row in data:
                if not isinstance(row, dict):
                    continue
                frame_idx = row.get("frame")
                try:
                    frame_idx = int(frame_idx)
                except Exception:
                    continue
                xy = _extract_xy(row)
                if xy is None:
                    # Try common alternatives
                    try:
                        x = float(row.get("X", row.get("x")))
                        y = float(row.get("Y", row.get("y")))
                        if x is not None and y is not None:
                            xy = (x, y)
                    except Exception:
                        pass
                if xy is None:
                    continue
                lbl = _coerce_label(row.get("label", ""))
                frames[frame_idx].append({
                    "x": xy[0], "y": xy[1], "label": lbl, "id": row.get("id")
                })
    else:
        raise ValueError("Unsupported JSON structure for tracking data.")

    # Normalize: ensure frames from 0..max exist (even if empty) for the slider
    if frames:
        max_frame = max(frames.keys())
        for i in range(max_frame + 1):
            frames.setdefault(i, [])
    return dict(sorted(frames.items(), key=lambda kv: kv[0]))

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

class Playback2D:
    def __init__(self, frames: Dict[int, List[Dict[str, Any]]], title: str = "2D Tracking Playback"):
        self.frames = frames
        self.frame_indices = list(frames.keys())
        self.n_frames = len(self.frame_indices)
        self.idx = 0
        self.playing = False
        self.speed = 1  # frames per tick

        # Compute data extents for autoscaling
        xs, ys = [], []
        for dets in frames.values():
            for d in dets:
                xs.append(d["x"]); ys.append(d["y"])
        if not xs or not ys:
            raise ValueError("No x/y points found in the dataset.")

        pad_x = max(1e-6, (max(xs) - min(xs)) * 0.05)
        pad_y = max(1e-6, (max(ys) - min(ys)) * 0.05)

        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.canvas.manager.set_window_title(title)
        plt.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.18)

        # Collections for scatter points (players/referees/others)
        self.player_scatter = self.ax.scatter([], [], s=80, marker='o', color='red', label='player')
        self.ref_scatter = self.ax.scatter([], [], s=100, marker='o', color='black', label='referee')
        self.other_scatter = self.ax.scatter([], [], s=60, marker='o', color='0.6', label='other')

        # Text for frame counter
        self.txt = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes, va='top', ha='left')

        self.ax.set_title(title)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        self.ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right')

        # Slider for timeline
        ax_slider = self.fig.add_axes([0.07, 0.06, 0.86, 0.04])
        self.slider = Slider(ax_slider, 'Frame', 0, self.n_frames - 1, valinit=0, valfmt='%0.0f')
        self.slider.on_changed(self._on_slider_change)

        # Buttons: Play/Pause, Stop
        ax_play = self.fig.add_axes([0.07, 0.01, 0.1, 0.04])
        ax_stop = self.fig.add_axes([0.19, 0.01, 0.1, 0.04])
        self.btn_play = Button(ax_play, 'Play/Pause')
        self.btn_stop = Button(ax_stop, 'Stop')
        self.btn_play.on_clicked(lambda evt: self.toggle_play())
        self.btn_stop.on_clicked(lambda evt: self.stop())

        # Keyboard bindings
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # Timer for playback
        self.timer = self.fig.canvas.new_timer(interval=30)  # ~33 fps timer; speed scales frames per tick
        self.timer.add_callback(self._on_timer)

        # Initial draw
        self._draw_frame(0)

        # Animation object (not strictly necessary but keeps reference alive)
        self.ani = FuncAnimation(self.fig, lambda i: None)

        print("[INFO] Controls: Space=Play/Pause | Left/Right=Step | Home/End=Jump | Up/Down=Speed")

    # ----------------------- Controls -----------------------

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()

    def stop(self):
        self.playing = False
        self.timer.stop()
        self.seek(0)

    def seek(self, idx: int):
        idx = max(0, min(self.n_frames - 1, int(idx)))
        self.idx = idx
        self.slider.set_val(self.idx)  # Triggers on_changed -> _draw_frame

    def step(self, delta: int):
        self.seek(self.idx + delta)

    def _on_slider_change(self, val):
        self.idx = int(val)
        self._draw_frame(self.idx)

    def _on_timer(self):
        if not self.playing:
            return
        next_idx = self.idx + self.speed
        if next_idx >= self.n_frames:
            next_idx = 0  # loop
        self.seek(next_idx)

    def _on_key(self, event):
        if event.key == ' ':
            self.toggle_play()
        elif event.key == 'left':
            self.step(-1)
        elif event.key == 'right':
            self.step(+1)
        elif event.key == 'home':
            self.seek(0)
        elif event.key == 'end':
            self.seek(self.n_frames - 1)
        elif event.key == 'up':
            self.speed = min(25, self.speed + 1)
            print(f"[INFO] Speed: {self.speed} frames/tick")
        elif event.key == 'down':
            self.speed = max(1, self.speed - 1)
            print(f"[INFO] Speed: {self.speed} frames/tick")

    # ----------------------- Drawing ------------------------

    def _draw_frame(self, idx: int):
        frame_id = self.frame_indices[idx]
        detections = self.frames.get(frame_id, [])

        xs_p, ys_p = [], []
        xs_r, ys_r = [], []
        xs_o, ys_o = [], []

        for d in detections:
            lbl = _coerce_label(d.get("label", ""))
            x, y = d["x"], d["y"]
            if lbl == 'player':
                if len(xs_p) >= MAX_DISPLAY_PLAYERS:
                    continue
                xs_p.append(x); ys_p.append(y)
            elif lbl == 'referee':
                if len(xs_r) >= MAX_DISPLAY_REFEREES:
                    continue
                xs_r.append(x); ys_r.append(y)
            else:
                # Comment the next two lines if you prefer to hide "others"
                xs_o.append(x); ys_o.append(y)

        # Update scatter collections
        players = np.column_stack((xs_p, ys_p)) if xs_p else np.empty((0, 2))
        referees = np.column_stack((xs_r, ys_r)) if xs_r else np.empty((0, 2))
        others = np.column_stack((xs_o, ys_o)) if xs_o else np.empty((0, 2))
        self.player_scatter.set_offsets(players)
        self.ref_scatter.set_offsets(referees)
        self.other_scatter.set_offsets(others)

        self.txt.set_text(f"Frame: {frame_id} / {self.frame_indices[-1]} | "
                          f"Players: {len(xs_p)} | Referees: {len(xs_r)}")

        self.fig.canvas.draw_idle()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

JSON_INPUT = "output/tracking3d_output_multi.json"
def main():

    frames = load_frames(JSON_INPUT)
    app = Playback2D(frames, title="Basket 2D Multi-Object Playback")
    plt.show()

if __name__ == "__main__":
    main()
