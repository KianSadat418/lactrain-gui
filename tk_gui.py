# Placeholder for Tkinter-based application converting existing PyQt functionality.
# Due to environment and complexity constraints, this is a simplified rewrite
# using tkinter and matplotlib for 3D rendering. Not all original functionality
# is fully reproduced.

import sys
import json
import socket
import threading
from typing import List

import numpy as np

# Tkinter + matplotlib imports
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused in code but required

MATRIX_BUTTON_LABELS = [
    "Similarity Xform",
    "Affine Xform",
    "Similarity Xform RANSAC",
    "Affine Xform RANSAC",
]


class DataReceiver(threading.Thread):
    """Background thread that receives UDP packets and forwards parsed data."""

    def __init__(self, host="0.0.0.0", port=9991, callback=None):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.callback = callback
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((self.host, self.port))
        except Exception as e:
            print(f"[Receiver] Failed to start server: {e}")
            return

        while self._running:
            try:
                data, _ = sock.recvfrom(650000)
                if not data:
                    continue
                line = data.decode("utf-8").strip()
                if self.callback:
                    self.callback(line)
            except Exception as e:
                print(f"[Receiver] Error: {e}")
                break


class PointViewer(tk.Tk):
    """Tkinter based GUI roughly mirroring the PyQt application."""

    def __init__(self):
        super().__init__()
        self.title("Real-Time Point Viewer (Tk)")

        # Data storage
        self.camera_points: List[np.ndarray] = []
        self.holo_points: List[np.ndarray] = []
        self.matrix_transform_points: List[np.ndarray] = []
        self.manual_points: List[np.ndarray] = []
        self.manual_transformed_points: List[np.ndarray] = []
        self.transform_matrices: List[np.ndarray] = []

        # Build UI
        self._build_ui()

        # Start receiver
        self.receiver = DataReceiver(callback=self.handle_line)
        self.receiver.start()

        # redraw timer
        self.after(100, self.redraw)

    # ------------------------------------------------------------------ UI setup
    def _build_ui(self):
        self.fig = plt.Figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        # options
        self.cam_var = tk.BooleanVar(value=True)
        self.holo_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(right, text="Show Camera Points", variable=self.cam_var, command=self.redraw).pack(anchor="w")
        ttk.Checkbutton(right, text="Show HoloLens Points", variable=self.holo_var, command=self.redraw).pack(anchor="w")

        # manual transform point entry
        self.tx_entry = ttk.Entry(right, width=5)
        self.ty_entry = ttk.Entry(right, width=5)
        self.tz_entry = ttk.Entry(right, width=5)
        for e, lbl in zip((self.tx_entry, self.ty_entry, self.tz_entry), ("X", "Y", "Z")):
            ttk.Label(right, text=lbl).pack(anchor="w")
            e.pack(anchor="w")
        ttk.Button(right, text="Create Point", command=self.apply_transform).pack(fill=tk.X, pady=2)
        ttk.Button(right, text="Clear Points", command=self.clear_transform_points).pack(fill=tk.X, pady=2)
        ttk.Button(right, text="Load Matrices", command=self.load_matrices).pack(fill=tk.X, pady=2)

        # matrix radio buttons
        self.matrix_var = tk.IntVar(value=0)
        for i, label in enumerate(MATRIX_BUTTON_LABELS):
            ttk.Radiobutton(right, text=label, variable=self.matrix_var, value=i, command=self.redraw).pack(anchor="w")

        ttk.Button(right, text="Apply Matrix", command=self.apply_matrix_transform).pack(fill=tk.X, pady=2)
        ttk.Button(right, text="Reset View", command=self.reset_view).pack(fill=tk.X, pady=2)

        self.rmse_label = ttk.Label(right, text="RMSE: N/A")
        self.rmse_label.pack(pady=5)

    # ------------------------------------------------------------- data handling
    def handle_line(self, line: str):
        try:
            if line.startswith("M"):
                self.load_matrices()
            else:
                data = json.loads(line)
                camera_points = []
                holo_points = []
                for value in data.values():
                    cam = np.array(value[0])
                    holo = np.array(value[1])
                    camera_points.append(cam)
                    holo_points.append(holo)
                self.camera_points = camera_points
                self.holo_points = holo_points
                self.update_rmse()
        except Exception as e:
            print(f"[handle_line] error: {e}")

    # ------------------------------------------------------------------ actions
    def load_matrices(self):
        path = "Assets/Scripts/GUI/transform_data.txt"
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.transform_matrices = []
            for key in [
                "similarity_transform",
                "affine_transform",
                "similarity_transform_ransac",
                "affine_transform_ransac",
            ]:
                mat = np.array(data.get(key)).reshape(4, 4)
                self.transform_matrices.append(mat)
            print("[load_matrices] loaded")
        except Exception as e:
            print(f"[load_matrices] failed: {e}")

    def apply_matrix_transform(self):
        idx = self.matrix_var.get()
        if idx < 0 or idx >= len(self.transform_matrices):
            return
        matrix = self.transform_matrices[idx]
        transformed = []
        for pt in self.camera_points:
            pt_h = np.append(pt, 1.0)
            new_pt = matrix @ pt_h
            transformed.append(new_pt[:3])
        self.matrix_transform_points = transformed
        self.redraw()

    def apply_transform(self):
        try:
            x = float(self.tx_entry.get())
            y = float(self.ty_entry.get())
            z = float(self.tz_entry.get())
        except ValueError:
            return
        pt = np.array([x, y, z])
        self.manual_points.append(pt)
        idx = self.matrix_var.get()
        if 0 <= idx < len(self.transform_matrices):
            mat = self.transform_matrices[idx]
            pt_h = np.append(pt, 1.0)
            tpt = mat @ pt_h
            self.manual_transformed_points.append(tpt[:3])
        self.redraw()

    def clear_transform_points(self):
        self.matrix_transform_points.clear()
        self.manual_points.clear()
        self.manual_transformed_points.clear()
        self.redraw()

    def update_rmse(self):
        if not self.camera_points:
            self.rmse_label.config(text="RMSE: N/A")
            return
        cams = np.vstack(self.camera_points)
        holos = np.vstack(self.holo_points)
        diff = cams - holos
        rmse = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
        self.rmse_label.config(text=f"RMSE: {rmse:.4f}")

    # -------------------------------------------------------------- visualization
    def reset_view(self):
        self.ax.view_init(elev=20, azim=-60)
        self.canvas.draw_idle()

    def redraw(self):
        self.ax.clear()
        if self.camera_points and self.cam_var.get():
            pts = np.vstack(self.camera_points)
            self.ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color="red", s=20)
        if self.holo_points and self.holo_var.get():
            pts = np.vstack(self.holo_points)
            self.ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color="blue", s=20)
        if self.matrix_transform_points:
            pts = np.vstack(self.matrix_transform_points)
            self.ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color="green", s=20)
        if self.manual_points:
            pts = np.vstack(self.manual_points)
            self.ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color="yellow", s=40)
        if self.manual_transformed_points:
            pts = np.vstack(self.manual_transformed_points)
            self.ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color="orange", s=40)

        if (
            self.camera_points
            and self.holo_points
            and self.cam_var.get()
            and self.holo_var.get()
        ):
            for cam, holo in zip(self.camera_points, self.holo_points):
                self.draw_dashed_line(cam, holo)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.canvas.draw_idle()

    def draw_dashed_line(self, p1, p2, segments=30):
        points = np.linspace(p1, p2, segments * 2).reshape(-1, 2, 3)
        for i, (start, end) in enumerate(points):
            if i % 2 == 0:
                self.ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color="gray")

    # -------------------------------------------------------------------- cleanup
    def on_close(self):
        self.receiver.stop()
        self.destroy()


if __name__ == "__main__":
    app = PointViewer()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
