import sys
import socket
import time
from datetime import datetime
import json
from typing import List, Optional, Tuple
from collections import deque
import numpy as np
import os
import logging

import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui
import PyQt5.QtWidgets as QtWidgets

import pyvista as pv
from pyvista import Color
pv.global_theme.allow_empty_mesh = True
from pyvistaqt import QtInteractor


# =========================
# ====== CONSTANTS ========
# =========================

MATRIX_BUTTON_LABELS = [
    "Similarity Xform",
    "Affine Xform",
    "Similarity Xform RANSAC",
    "Affine Xform RANSAC",
]

GAZE_LINE_LENGTH = 500.0
DISC_RADIUS = 0.04

# Receiver (unchanged base, optional new key moving_peg supported)
RECEIVER_HOST = "0.0.0.0"
RECEIVER_PORT = 9991

# Default transform paths (kept; picker fallback if missing)
HARD_PATH_GAZE = r"C:\Users\kiansadat\Desktop\Azimi Project\3D Eye Gaze\Assets\Scripts\GUI\transform_data.txt"
HARD_PATH_CALIB = r"Assets/Scripts/GUI/transform_data.txt"


# =========================
# ====== LOGGING ==========
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("SurgicalGUI")


# =========================
# ====== STYLING ==========
# =========================

def apply_app_style(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(37, 37, 37))
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(30, 30, 30))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(64, 128, 255))
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
    app.setPalette(palette)
    app.setFont(QtGui.QFont("Segoe UI", 9))
    app.setStyleSheet("""
        QGroupBox {
            font-weight: 600;
            border: 1px solid #4A4A4A;
            border-radius: 6px;
            margin-top: 10px;
            padding: 8px;
        }
        QGroupBox::title { left: 10px; padding: 0 3px; }
        QPushButton { padding: 6px 10px; border-radius: 6px; border: 1px solid #5A5A5A; }
        QPushButton:disabled { color: #888; border: 1px solid #444; }
        QLineEdit {
            padding: 4px 6px; border-radius: 4px; border: 1px solid #5A5A5A;
            background: #2A2A2A; color: #EEE;
        }
        QSlider::groove:horizontal { height: 6px; background: #555; border-radius: 3px; }
        QSlider::handle:horizontal { background: #8AA8FF; width: 14px; margin: -4px 0; border-radius: 7px; }
        QTableWidget { gridline-color: #666; background-color: #232323; color: #EEE; selection-background-color: #3A6EA5; }
        QLabel[kind="title"] { font-size: 16px; font-weight: 700; }
        QLabel[kind="meter"] { font-size: 16px; font-weight: 600; }
    """)


# =========================
# ===== UTILITIES =========
# =========================

def _format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes:02d}:{seconds:06.3f}"


def _safe_array(a, shape: Tuple[int, ...], name: str) -> Optional[np.ndarray]:
    arr = np.array(a)
    if arr.shape != shape:
        log.warning("[%s] Expected shape %s, got %s", name, shape, arr.shape)
        return None
    return arr


# ======================================
# ===== TRANSFORM LOAD (KEEP PROTO) ====
# ======================================

def load_transform_matrices_from_file(path: str):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError("No valid file selected or file does not exist.")

    try:
        with open(path, "r") as f:
            matrix_json = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")
    except Exception as e:
        raise ValueError(f"Failed to read file: {e}")

    required_keys = [
        "similarity_transform", "affine_transform",
        "similarity_transform_ransac", "affine_transform_ransac"
    ]
    transform_matrices = []
    for key in required_keys:
        if key not in matrix_json:
            raise ValueError(f"Missing required key: {key}")
        matrix_data = matrix_json[key]
        matrix = np.array(matrix_data).reshape(4, 4)
        transform_matrices.append(matrix)

    rmse_values = matrix_json.get("rmse", [0, 0, 0, 0])
    repro_values = matrix_json.get("repro", [0, 0, 0, 0])
    row_dict = matrix_json.get("rows", {})
    camera_points = [np.array(p) for p in row_dict.get("Camera", [])]
    hololens_points = [np.array(p) for p in row_dict.get("Hololens", [])]

    point_rows = []
    num_rows = len(row_dict.get("Camera", []))
    for i in range(num_rows):
        row = [
            row_dict["Camera"][i],
            row_dict["Hololens"][i],
            row_dict["similarity_transformed_B"][i],
            row_dict["similarity_transform_errors"][i],
            row_dict["affine_transformed_B"][i],
            row_dict["affine_transform_errors"][i],
            row_dict["similarity_transformed_ransac_B"][i],
            row_dict["similarity_transform_ransac_errors"][i],
            row_dict["similarity_transform_ransac_mask"][i],
            row_dict["affine_transformed_ransac_B"][i],
            row_dict["affine_transform_ransac_errors"][i],
            row_dict["affine_transform_ransac_mask"][i]
        ]
        point_rows.append(row)

    return {
        "matrices": transform_matrices,
        "camera_points": camera_points,
        "hololens_points": hololens_points,
        "rmse": rmse_values,
        "repro": repro_values,
        "rows": point_rows
    }


# =========================
# ====== RECORDER =========
# =========================

class Recorder:
    def __init__(self):
        self.frames = []
        self.recording = False

    def start(self):
        self.frames = []
        self.recording = True
        log.info("[Recorder] Started recording.")

    def stop(self):
        self.recording = False
        log.info("[Recorder] Stopped recording.")

    def log_frame(self, gaze_data, peg_data, transform_id):
        if self.recording:
            timestamp = QtCore.QDateTime.currentDateTime().toString(QtCore.Qt.ISODate)
            self.frames.append({
                "timestamp": timestamp,
                "gaze_data": {
                    "gaze_line": gaze_data.get("gaze_line", []),
                    "roi": gaze_data.get("roi", 0.0),
                    "intercept": gaze_data.get("intercept", 0),
                    "gaze_distance": gaze_data.get("gaze_distance", 0.0)
                },
                "peg_data": peg_data,
                "transform_id": transform_id
            })

    def save(self, filepath):
        with open(filepath, "w") as f:
            json.dump(self.frames, f, indent=2)
        log.info("[Recorder] Saved %d frames to %s", len(self.frames), filepath)


# =========================
# ===== DATA RECEIVER =====
# =========================

class DataReceiver(QtCore.QThread):
    updated_points = QtCore.pyqtSignal(list, list)
    peg_point_received = QtCore.pyqtSignal(np.ndarray)
    validation_gaze_received = QtCore.pyqtSignal(object, float, int, float)
    full_gaze_received = QtCore.pyqtSignal(object, float, int, float, object)
    moving_peg_received = QtCore.pyqtSignal(int)  # NEW: moving peg index (0-based)

    def __init__(self, host=RECEIVER_HOST, port=RECEIVER_PORT, parent=None):
        super().__init__(parent)
        self.host = host
        self.port = port
        self._running = True
        self.server_socket = None

    def stop(self):
        self._running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception as e:
                log.warning("[Receiver] Error closing socket: %s", e)
            self.server_socket = None

    @staticmethod
    def _extract_moving_peg(d: dict) -> Optional[int]:
        """
        Extract a moving peg index from various possible keys.
        Accepts 0-based or 1-based; returns 0-based int if present and valid.
        """
        candidates = ["moving_peg", "movingPeg", "peg_of_interest", "pegIndex", "peg_index"]
        val = None
        for k in candidates:
            if k in d:
                val = d[k]
                break
        if val is None:
            return None
        try:
            idx = int(val)
        except Exception:
            return None
        # Normalize to 0-based in [0..5]
        if 1 <= idx <= 6:
            return idx - 1
        if 0 <= idx <= 5:
            return idx
        return None

    def run(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.setblocking(False)
            log.info("[Receiver] UDP bound on %s:%d", self.host, self.port)
        except Exception as e:
            log.error("[Receiver] Failed to start server: %s", e)
            return

        while self._running:
            try:
                data, addr = self.server_socket.recvfrom(650000)
            except BlockingIOError:
                QtCore.QThread.msleep(1)
                continue
            except Exception as e:
                log.error("[Receiver] Socket error: %s", e)
                break

            if not data:
                continue

            try:
                line = data.decode("utf-8").strip()

                if line.startswith("M"):
                    QtCore.QMetaObject.invokeMethod(self.parent(), "trigger_matrix_mode", QtCore.Qt.QueuedConnection)

                elif line.startswith("V"):
                    d = json.loads(line[1:])
                    point = np.array([float(d["x"]), float(d["y"]), float(d["z"])])
                    self.peg_point_received.emit(point)
                    self.validation_gaze_received.emit(
                        np.array(d.get("gaze_line")),
                        float(d.get("roi", 0.0)),
                        int(d.get("intercept", 0)),
                        float(d.get("gaze_distance", 0.0)),
                    )
                    mp = self._extract_moving_peg(d)
                    if mp is not None:
                        self.moving_peg_received.emit(mp)

                elif line.startswith("G"):
                    d = json.loads(line[1:])
                    self.full_gaze_received.emit(
                        np.array(d.get("gaze_line")),
                        float(d.get("roi", 0.0)),
                        int(d.get("intercept", 0)),
                        float(d.get("gaze_distance", 0.0)),
                        np.array(d.get("pegs", []))
                    )
                    mp = self._extract_moving_peg(d)
                    if mp is not None:
                        self.moving_peg_received.emit(mp)

                elif line.startswith("D"):
                    d = json.loads(line[1:])
                    log.debug("[Receiver] D data: %s", d)
                    self.full_gaze_received.emit(
                        np.array(d.get("gaze_position")),
                        float(d.get("roi", 0.0)),
                        int(d.get("intercept", 0)),
                        float(d.get("gaze_distance", 0.0)),
                        np.array(d.get("pegs", []))
                    )
                    mp = self._extract_moving_peg(d)
                    if mp is not None:
                        self.moving_peg_received.emit(mp)

                else:
                    raw = json.loads(line)
                    camera_points = [np.array(v[0]) for v in raw.values()]
                    holo_points = [np.array(v[1]) for v in raw.values()]
                    self.updated_points.emit(camera_points, holo_points)

            except Exception as e:
                log.warning("[Receiver] Failed to process message: %s", e)

        log.info("[Receiver] Shutting down cleanly.")


# ===============================
# ===== MATRIX INFO WINDOW ======
# ===============================

class MatrixInfoWindow(QtWidgets.QWidget):
    def __init__(self, matrix_data, point_rows, rmse_values, repro_values, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Transformation Matrix Info")
        self.setMinimumWidth(1000)
        layout = QtWidgets.QVBoxLayout()

        header = QtWidgets.QLabel("Matrix & Fit Details")
        header.setProperty("kind", "title")
        layout.addWidget(header)

        table = QtWidgets.QTableWidget()
        table.setColumnCount(12)
        table.setHorizontalHeaderLabels([
            "Camera", "HoloLens",
            "Sim Xform", "Sim Error",
            "Affine Xform", "Affine Error",
            "Sim RANSAC", "Sim RANSAC Error", "Sim RANSAC Mask",
            "Affine RANSAC", "Affine RANSAC Error", "Affine RANSAC Mask"
        ])
        table.setRowCount(len(point_rows))
        for i, row in enumerate(point_rows):
            for j, val in enumerate(row):
                if isinstance(val, (list, tuple)) and len(val) == 3:
                    formatted = f"({val[0]:.3f}, {val[1]:.3f}, {val[2]:.3f})"
                else:
                    formatted = f"{val:.4f}" if isinstance(val, (float, int)) else str(val)
                table.setItem(i, j, QtWidgets.QTableWidgetItem(formatted))
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        layout.addWidget(table)

        matrix_layout = QtWidgets.QFormLayout()
        for name, mat, rmse, repro in zip(
            ["Similarity", "Affine", "Similarity RANSAC", "Affine RANSAC"],
            matrix_data, rmse_values, repro_values
        ):
            mat_str = "\n".join("  ".join(f"{v:.4f}" for v in row) for row in mat)
            matrix_label = QtWidgets.QLabel(f"<pre>{mat_str}</pre>")
            matrix_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            matrix_layout.addRow(f"{name} Matrix (RMSE: {rmse:.4f}, Repro: {repro:.4f})", matrix_label)

        layout.addLayout(matrix_layout)
        self.setLayout(layout)


# =====================================
# ===== GAZE TRACKING (REAL-TIME) =====
# =====================================

class GazeTrackingWindow(QtWidgets.QWidget):
    def __init__(self, transform_matrices=None, enable_receiver=True, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gaze Tracking")
        self.setMinimumSize(980, 680)

        # === Core Attributes ===
        self.recorder = Recorder()
        self.transform_matrices = transform_matrices or []
        self.latest_gaze_line = None
        self.latest_roi = 0.0
        self.latest_intercept = 0
        self.latest_pegs = None
        self.latest_gaze_distance = 0.0
        self.fix_view_actors = []

        self.peg_smoothing_enabled = True
        self.peg_history = [deque(maxlen=5) for _ in range(6)]

        # moving peg (0-based), default 0 until data arrives
        self.peg_of_interest_index = 0

        self.receiver = None
        if enable_receiver:
            self.receiver = DataReceiver(parent=self)
            self.receiver.full_gaze_received.connect(self.update_gaze_data)
            self.receiver.moving_peg_received.connect(self._set_moving_peg)
            self.receiver.start()

        # === Plotter ===
        self.plotter = QtInteractor(self)

        # Transformed Peg Mesh (all 6)
        self.transformed_peg_mesh = pv.PolyData(np.zeros((6, 3)))
        self.transformed_peg_actor = self.plotter.add_mesh(
            self.transformed_peg_mesh, color="#300053", point_size=12, render_points_as_spheres=True
        )

        # === Dashed Lines (peg -> gaze line) ===
        self.dashed_segments = 10
        self.dashed_meshes = []
        self.dashed_actors = []
        for _ in range(self.dashed_segments):
            mesh = pv.PolyData()
            mesh.points = pv.pyvista_ndarray([[0, 0, 0], [0, 0, 0]])
            mesh.lines = np.array([2, 0, 1])
            actor = self.plotter.add_mesh(mesh, color="red", line_width=2)
            self.dashed_meshes.append(mesh)
            self.dashed_actors.append(actor)

        # === View Controls ===
        self.reset_button = QtWidgets.QPushButton("Reset View")
        self.zoom_in_button = QtWidgets.QPushButton("Zoom In")
        self.zoom_out_button = QtWidgets.QPushButton("Zoom Out")
        self.fix_view_checkbox = QtWidgets.QCheckBox("Fix View")
        self.fix_view_checkbox.setChecked(False)

        self.reset_button.clicked.connect(self.reset_view)
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.fix_view_checkbox.stateChanged.connect(self._update_fix_view_bounds)

        view_layout = QtWidgets.QVBoxLayout()
        view_layout.addWidget(self.reset_button)
        view_layout.addWidget(self.zoom_in_button)
        view_layout.addWidget(self.zoom_out_button)
        view_layout.addWidget(self.fix_view_checkbox)
        view_layout.addStretch()
        view_group = QtWidgets.QGroupBox("View")
        view_group.setLayout(view_layout)

        # === Matrix Transform Radio Buttons ===
        transform_layout = QtWidgets.QVBoxLayout()
        self.matrix_group = QtWidgets.QButtonGroup()
        self.matrix_buttons = []

        self.load_matrices_button = QtWidgets.QPushButton("Load Matrices")
        self.load_matrices_button.setToolTip("Load transformation matrices JSON")
        self.load_matrices_button.clicked.connect(self.load_matrices_from_file)
        transform_layout.addWidget(self.load_matrices_button)

        for i, label in enumerate(MATRIX_BUTTON_LABELS):
            btn = QtWidgets.QRadioButton(label)
            self.matrix_group.addButton(btn, i)
            self.matrix_buttons.append(btn)
            transform_layout.addWidget(btn)

        self.matrix_buttons[0].setChecked(True)
        self.matrix_group.buttonClicked.connect(self._update_gaze_line)

        transform_group = QtWidgets.QGroupBox("Transform")
        transform_group.setLayout(transform_layout)

        # === Moving Peg Display (no radios) ===
        mp_group = QtWidgets.QGroupBox("Moving Peg")
        mp_layout = QtWidgets.QVBoxLayout()
        self.moving_peg_label = QtWidgets.QLabel("Moving Peg: —")
        self.moving_peg_label.setProperty("kind", "meter")
        mp_layout.addWidget(self.moving_peg_label)
        mp_group.setLayout(mp_layout)

        # === Gaze Distance Label ===
        self.gaze_distance_label = QtWidgets.QLabel("Gaze Distance: N/A")
        self.gaze_distance_label.setProperty("kind", "meter")

        # === Timer Display Row ===
        timer_layout = QtWidgets.QHBoxLayout()
        self.timer_label = QtWidgets.QLabel("00:00.000")
        self.timer_label.setProperty("kind", "meter")

        last_time_layout = QtWidgets.QHBoxLayout()
        last_time_label = QtWidgets.QLabel("Last Time:")
        self.last_time_display = QtWidgets.QLabel("00:00.000")
        last_time_layout.addWidget(last_time_label)
        last_time_layout.addWidget(self.last_time_display)
        last_time_layout.addStretch()

        timer_layout.addWidget(self.timer_label)
        timer_layout.addStretch()
        timer_layout.addLayout(last_time_layout)

        # === Recording Controls ===
        self.start_recording_button = QtWidgets.QPushButton("Start Recording")
        self.stop_recording_button = QtWidgets.QPushButton("Stop & Save Recording")
        self.stop_recording_button.setEnabled(False)

        self.record_indicator = QtWidgets.QLabel("● REC")
        self.record_indicator.setStyleSheet("color: red; font-weight: bold;")
        self.record_indicator.hide()

        # Connect buttons
        self.start_recording_button.clicked.connect(self._start_recording)
        self.stop_recording_button.clicked.connect(self._save_recording)

        # === Right Panel ===
        right_panel = QtWidgets.QVBoxLayout()
        right_panel.addWidget(transform_group)
        right_panel.addWidget(mp_group)
        right_panel.addWidget(view_group)
        right_panel.addWidget(self.gaze_distance_label)
        right_panel.addSpacing(8)
        right_panel.addLayout(timer_layout)
        right_panel.addWidget(self.start_recording_button)
        right_panel.addWidget(self.stop_recording_button)
        right_panel.addWidget(self.record_indicator)
        right_panel.addStretch()

        # === Layouts ===
        viewer_layout = QtWidgets.QVBoxLayout()
        viewer_layout.addWidget(self.plotter.interactor)

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(viewer_layout, stretch=4)
        main_layout.addLayout(right_panel, stretch=1)
        self.setLayout(main_layout)

        # Plotter setup
        QtCore.QTimer.singleShot(0, self.plotter.show_axes)
        QtCore.QTimer.singleShot(0, self.plotter.show_grid)
        self.plotter.enable_point_picking(
            callback=self._on_point_picked, use_picker=True,
            show_message=False, left_clicking=True, show_point=True
        )

        # Render timer
        self.render_timer = QtCore.QTimer(self)
        self.render_timer.setInterval(60)
        self.render_timer.timeout.connect(self.plotter.render)
        self.render_timer.start()

        # Recorder timers
        self.record_timer = QtCore.QTimer(self)
        self.record_timer.setInterval(60)
        self.record_timer.timeout.connect(self._log_current_frame)

        self.recording_timer = QtCore.QTimer(self)
        self.recording_timer.timeout.connect(self._update_recording_timer)
        self.recording_duration = 0.0
        self.last_recorded_time = 0.0
        self.record_start_time = None

    # ===== Helpers =====

    def _set_moving_peg(self, index: int):
        """Update moving peg from receiver (0-based)."""
        if not (0 <= index <= 5):
            return
        self.peg_of_interest_index = index
        self.moving_peg_label.setText(f"Moving Peg: {index + 1}")
        self._update_gaze_line()

    @QtCore.pyqtSlot(object, float, int, float, object)
    def update_gaze_data(self, gaze_line, roi, intercept, gaze_distance, pegs):
        try:
            gaze_arr = _safe_array(gaze_line, (2, 3), "GazeTracking.gaze_line")
            if gaze_arr is None:
                return

            origin = gaze_arr[0]
            direction = gaze_arr[1]
            direction_norm = np.linalg.norm(direction)
            if direction_norm == 0:
                log.warning("[GazeTracking] Zero-length direction vector.")
                return
            direction = direction / direction_norm

            peg_arr = np.array(pegs)
            if peg_arr.shape != (6, 3):
                log.warning("[GazeTracking] Invalid peg shape: %s", peg_arr.shape)
                return

            idx = self.matrix_group.checkedId()
            matrix = self.transform_matrices[idx] if 0 <= idx < len(self.transform_matrices) else np.eye(4)
            transformed_pegs = np.array([(matrix @ np.append(p, 1.0))[:3] for p in peg_arr])

            if self.peg_smoothing_enabled:
                for i in range(6):
                    self.peg_history[i].append(transformed_pegs[i])
                smoothed_pegs = np.array([np.mean(hist, axis=0) for hist in self.peg_history])
            else:
                smoothed_pegs = transformed_pegs

            peg_idx = self.peg_of_interest_index
            if 0 <= peg_idx < 6:
                gaze_length = np.linalg.norm(smoothed_pegs[peg_idx] - origin)
            else:
                gaze_length = GAZE_LINE_LENGTH
            target = origin + direction * gaze_length

            self.latest_gaze_line = np.array([origin, target])
            self.latest_roi = roi
            self.latest_intercept = intercept
            self.latest_gaze_distance = gaze_distance
            self.latest_pegs = smoothed_pegs

            self.gaze_distance_label.setText(f"Gaze Distance: {gaze_distance:.2f} mm")
            self._update_gaze_line()

        except Exception as e:
            log.warning("[GazeTracking] Failed to update visuals: %s", e)

    def _update_recording_timer(self):
        if self.record_start_time is not None:
            self.recording_duration = time.time() - self.record_start_time
            self.timer_label.setText(_format_time(self.recording_duration))

    def _update_gaze_line(self):
        if self.latest_gaze_line is None or self.latest_pegs is None:
            return

        A, B = self.latest_gaze_line
        direction = B - A
        length = np.linalg.norm(direction)
        if length == 0:
            return
        norm_direction = direction / length
        cone_center = A + 0.5 * (B - A)

        # 1) Gaze line
        line = pv.Line(A, B)
        if hasattr(self, "gaze_line_mesh"):
            self.gaze_line_mesh.deep_copy(line)
            self.gaze_line_mesh.Modified()
        else:
            self.gaze_line_mesh = line
            self.gaze_line_actor = self.plotter.add_mesh(self.gaze_line_mesh, color="green", line_width=3)

        # 2) Disc at end
        disc = pv.Disc(center=B, inner=0.0, outer=DISC_RADIUS, normal=norm_direction, r_res=1, c_res=10)
        if hasattr(self, "disc_mesh"):
            self.disc_mesh.deep_copy(disc)
            self.disc_mesh.Modified()
        else:
            self.disc_mesh = disc
            self.disc_actor = self.plotter.add_mesh(self.disc_mesh, color="cyan", opacity=0.5)

        # 3) Cone
        cone_color = "green" if bool(self.latest_intercept) else "red"
        cone = pv.Cone(center=cone_center, direction=-norm_direction, height=length, radius=DISC_RADIUS)
        if hasattr(self, "cone_mesh"):
            self.cone_mesh.deep_copy(cone)
            self.cone_mesh.Modified()
            if hasattr(self, "cone_actor"):
                self.cone_actor.GetProperty().SetColor(Color(cone_color).float_rgb)
        else:
            self.cone_mesh = cone
            self.cone_actor = self.plotter.add_mesh(self.cone_mesh, color=cone_color, opacity=0.3)

        # 4) Update pegs
        pegs = np.array(self.latest_pegs)
        if hasattr(self, "transformed_peg_mesh") and self.transformed_peg_mesh.n_points == 6:
            self.transformed_peg_mesh.points = pegs
            self.transformed_peg_mesh.Modified()
        else:
            self.transformed_peg_mesh = pv.PolyData(pegs)
            self.transformed_peg_actor = self.plotter.add_mesh(
                self.transformed_peg_mesh, color="#300053", point_size=12, render_points_as_spheres=True
            )

        # 5) ROI sphere at moving peg + dashed to gaze
        peg = pegs[self.peg_of_interest_index]
        roi_sphere = pv.Sphere(radius=0.02, center=peg, theta_resolution=20, phi_resolution=20)

        if hasattr(self, "roi_sphere_mesh"):
            self.roi_sphere_mesh.deep_copy(roi_sphere)
            self.roi_sphere_mesh.Modified()
        else:
            self.roi_sphere_mesh = roi_sphere
            self.roi_sphere_actor = self.plotter.add_mesh(
                self.roi_sphere_mesh, color="yellow", opacity=0.15,
                specular=0.5, specular_power=15, smooth_shading=True
            )

        def closest_point_on_line(p, a, b):
            ab = b - a
            denom = np.dot(ab, ab)
            if denom == 0:
                return a
            t = np.dot(p - a, ab) / denom
            return a + np.clip(t, 0, 1) * ab

        closest_point = closest_point_on_line(peg, A, B)
        pairs = np.linspace(closest_point, peg, self.dashed_segments * 2).reshape(-1, 2, 3)

        dash_idx = 0
        for i in range(0, len(pairs), 2):
            start, end = pairs[i]
            mesh = self.dashed_meshes[dash_idx]
            mesh.points = np.array([start, end])
            mesh.lines = np.array([2, 0, 1])
            mesh.Modified()
            dash_idx += 1
            if dash_idx >= len(self.dashed_meshes):
                break
        for j in range(dash_idx, len(self.dashed_meshes)):
            self.dashed_meshes[j].points = np.array([[0, 0, 0], [0, 0, 0]])
            self.dashed_meshes[j].Modified()

    def load_matrices_from_file(self):
        path = HARD_PATH_GAZE
        if not os.path.isfile(path):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Transform JSON", "", "JSON (*.txt *.json)")
        if not path:
            QtWidgets.QMessageBox.warning(self, "Load Failed", "No file selected.")
            return
        try:
            result = load_transform_matrices_from_file(path)
            self.transform_matrices = result["matrices"]
            QtWidgets.QMessageBox.information(self, "Matrices Loaded", "Successfully loaded matrices.")
            self._update_gaze_line()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load Failed", str(e))
            log.warning("[GazeTracking] Failed to load matrices: %s", e)

    def reset_view(self):
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        self.plotter.render()

    def _on_point_picked(self, picked_point, picker):
        if picked_point is not None:
            coord_str = f"Picked Point: ({picked_point[0]:.2f}, {picked_point[1]:.2f}, {picked_point[2]:.2f})"
            QtWidgets.QMessageBox.information(self, "Point Coordinates", coord_str)
            log.info(coord_str)

    def zoom_in(self):
        self._zoom(1.2)

    def zoom_out(self):
        self._zoom(0.8)

    def _zoom(self, factor: float):
        camera = self.plotter.camera
        if hasattr(camera, "Zoom"):
            camera.Zoom(factor)
        self.plotter.render()

    def _update_fix_view_bounds(self):
        if hasattr(self, "fix_view_actors"):
            for actor in self.fix_view_actors:
                self.plotter.remove_actor(actor)
        self.fix_view_actors = []

        if self.fix_view_checkbox.isChecked():
            invisible_bounds = np.array([[-100, -100, -100], [600, 600, 600]], dtype=np.float32)
            origin = np.array([[0, 0, 0]], dtype=np.float32)

            bounds_actor = self.plotter.add_points(
                invisible_bounds, color="white", opacity=0.0, point_size=1, render_points_as_spheres=True
            )
            origin_actor = self.plotter.add_points(
                origin, color="black", point_size=10, render_points_as_spheres=True
            )
            self.fix_view_actors.extend([bounds_actor, origin_actor])

        self.plotter.render()

    def _save_recording(self):
        self.recorder.stop()
        self.record_timer.stop()
        self.recording_timer.stop()
        self.record_indicator.hide()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Recording", "", "JSON Files (*.json)")
        if path:
            self.recorder.save(path)
            self.last_recorded_time = self.recording_duration
            self.last_time_display.setText(_format_time(self.last_recorded_time))
        self.start_recording_button.setEnabled(True)
        self.stop_recording_button.setEnabled(False)

    def _start_recording(self):
        self.recorder.start()
        self.record_indicator.show()
        self.start_recording_button.setEnabled(False)
        self.stop_recording_button.setEnabled(True)
        self.recording_duration = 0.0
        self.timer_label.setText("00:00.000")
        self.record_start_time = time.time()
        self.recording_timer.start(50)
        self.record_timer.start()

    def _log_current_frame(self):
        if not self.recorder.recording or self.latest_gaze_line is None:
            return
        gaze_dict = {
            "gaze_line": self.latest_gaze_line.tolist(),
            "roi": self.latest_roi,
            "intercept": self.latest_intercept,
            "gaze_distance": self.latest_gaze_distance,
        }
        peg_list = (
            self.latest_pegs.tolist()
            if isinstance(self.latest_pegs, np.ndarray)
            else self.latest_pegs
        )
        self.recorder.log_frame(gaze_dict, peg_list, self.matrix_group.checkedId())

    def closeEvent(self, event):
        if self.receiver:
            self.receiver.stop()
            self.receiver.wait()
        super().closeEvent(event)


# =====================================
# ======== PLAYBACK WINDOW =============
# =====================================

class PlaybackWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Playback Mode")
        self.setMinimumSize(980, 720)

        self.frames = []
        self.fixation_points = []
        self.current_index = 0
        self.is_playing = False
        self.intercept_history = []

        self.plotter = QtInteractor(self)
        self.gaze_line_actor = None
        self.disc_actor = None
        self.cone_actor = None
        self.peg_actor = None

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(60)
        self.timer.timeout.connect(self.advance_frame)

        # Viewer Controls
        self.reset_button = QtWidgets.QPushButton("Reset View")
        self.zoom_in_button = QtWidgets.QPushButton("Zoom In")
        self.zoom_out_button = QtWidgets.QPushButton("Zoom Out")
        self.fix_view_checkbox = QtWidgets.QCheckBox("Fix View")
        self.fix_view_checkbox.setChecked(False)

        # Heatmap buttons (updated)
        self.perpeg_heatmaps_button = QtWidgets.QPushButton("Save Per-Peg Gaze Heatmaps")
        self.perpeg_heatmaps_button.setToolTip("Saves 6 heatmaps: gaze_heatmap_1.png … gaze_heatmap_6.png")
        self.perpeg_heatmaps_button.clicked.connect(self.save_gaze_heatmaps_per_peg)

        self.gaze_heatmap_button = QtWidgets.QPushButton("Save All-Peg Gaze Heatmap")
        self.gaze_heatmap_button.clicked.connect(self.save_gaze_heatmap_all)

        # Fixation heatmap kept as-is
        self.heatmap_button = QtWidgets.QPushButton("Show Fixation Heatmap")
        self.heatmap_button.clicked.connect(self.generate_fixation_heatmap)

        # Trajectories
        self.trajectory_checkbox = QtWidgets.QCheckBox("Show Peg Trajectories")
        self.trajectory_checkbox.setChecked(False)

        # NEW: Trajectory filter radios (Peg 1–6 + All)
        self.trajectory_group = QtWidgets.QGroupBox("Trajectory Filter")
        traj_layout = QtWidgets.QVBoxLayout()
        self.traj_button_group = QtWidgets.QButtonGroup(self)
        self.traj_radios = []
        for i in range(6):
            rb = QtWidgets.QRadioButton(f"Peg {i+1}")
            self.traj_button_group.addButton(rb, i)
            traj_layout.addWidget(rb)
            self.traj_radios.append(rb)
        rb_all = QtWidgets.QRadioButton("All")
        rb_all.setChecked(True)
        self.traj_button_group.addButton(rb_all, 6)
        traj_layout.addWidget(rb_all)
        self.trajectory_group.setLayout(traj_layout)
        self.trajectory_group.setEnabled(False)

        # Scrubber
        self.scrubber = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scrubber.setMinimum(0)
        self.scrubber.valueChanged.connect(self.update_frame)

        # Playback Buttons
        self.play_button = QtWidgets.QPushButton("▶ Play")
        self.pause_button = QtWidgets.QPushButton("⏸ Pause")
        self.restart_button = QtWidgets.QPushButton("↺ Restart")
        self.play_button.clicked.connect(self.play)
        self.pause_button.clicked.connect(self.pause)
        self.restart_button.clicked.connect(self.restart)

        # Load Recording
        self.load_button = QtWidgets.QPushButton("Load Recording")
        self.load_button.clicked.connect(self.load_recording)

        # Layouts
        controls_layout = QtWidgets.QVBoxLayout()
        controls_layout.addWidget(self.reset_button)
        controls_layout.addWidget(self.zoom_in_button)
        controls_layout.addWidget(self.zoom_out_button)
        controls_layout.addWidget(self.fix_view_checkbox)
        controls_layout.addSpacing(8)
        controls_layout.addWidget(self.perpeg_heatmaps_button)
        controls_layout.addWidget(self.gaze_heatmap_button)
        controls_layout.addWidget(self.heatmap_button)
        controls_layout.addSpacing(8)
        controls_layout.addWidget(self.trajectory_checkbox)
        controls_layout.addWidget(self.trajectory_group)
        controls_layout.addStretch()

        controls_group = QtWidgets.QGroupBox("Controls")
        controls_group.setLayout(controls_layout)

        right_panel = QtWidgets.QVBoxLayout()
        right_panel.addWidget(self.load_button)
        right_panel.addWidget(controls_group)
        right_panel.addStretch()

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addWidget(self.plotter.interactor, stretch=4)
        main_layout.addLayout(right_panel, stretch=1)

        playback_controls = QtWidgets.QHBoxLayout()
        playback_controls.addWidget(self.play_button)
        playback_controls.addWidget(self.pause_button)
        playback_controls.addWidget(self.restart_button)

        container_layout = QtWidgets.QVBoxLayout()
        container_layout.addWidget(self.scrubber)
        container_layout.addLayout(playback_controls)
        container_layout.addLayout(main_layout)
        self.setLayout(container_layout)

        # Final setup
        self.trajectory_checkbox.stateChanged.connect(self._toggle_trajectory_filter_enabled)
        self.traj_button_group.buttonClicked[int].connect(lambda _id: self.update_frame(self.current_index if self.frames else None))
        self.reset_button.clicked.connect(self.reset_view)
        self.zoom_in_button.clicked.connect(lambda: self._zoom(1.2))
        self.zoom_out_button.clicked.connect(lambda: self._zoom(0.8))
        self.fix_view_checkbox.stateChanged.connect(self.fix_view_bounds)

        self.plotter.show_axes()
        self.plotter.show_grid()

    # ===== heatmap helpers =====

    def _collect_gaze_endpoints(self):
        """Return list of gaze end points for all frames."""
        points = []
        for fr in self.frames:
            gl = fr["gaze_data"]["gaze_line"]
            if len(gl) >= 2:
                points.append(np.array(gl[1]))
        return points

    def _assign_points_to_nearest_peg(self, end_points):
        """Map each gaze endpoint to nearest peg index for its frame."""
        per_peg = [[] for _ in range(6)]
        for idx, fr in enumerate(self.frames):
            if idx >= len(end_points):
                break
            B = np.array(fr["gaze_data"]["gaze_line"][1]) if len(fr["gaze_data"]["gaze_line"]) >= 2 else None
            peg_data = np.array(fr["peg_data"])
            if B is None or peg_data.shape != (6, 3):
                continue
            dists = np.linalg.norm(peg_data - B, axis=1)
            peg_idx = int(np.argmin(dists))
            per_peg[peg_idx].append(B)
        return [np.array(lst) if len(lst) else np.empty((0, 3)) for lst in per_peg]

    def save_gaze_heatmaps_per_peg(self):
        if not self.frames:
            QtWidgets.QMessageBox.information(self, "No Data", "No gaze data available.")
            return

        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Folder for Per-Peg Heatmaps", "")
        if not out_dir:
            return

        end_points = self._collect_gaze_endpoints()
        per_peg_points = self._assign_points_to_nearest_peg(end_points)

        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde

        saved = []
        for i in range(6):
            pts = per_peg_points[i]
            path = os.path.join(out_dir, f"gaze_heatmap_{i+1}.png")
            if pts.shape[0] >= 2:
                x, y = pts[:, 0], pts[:, 1]
                xy = np.vstack([x, y])
                try:
                    kde = gaussian_kde(xy)
                    xi, yi = np.mgrid[x.min():x.max():500j, y.min():y.max():500j]
                    zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
                    plt.figure(figsize=(8, 6))
                    plt.title(f"Gaze Heatmap - Peg {i+1}")
                    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading="auto", cmap="viridis")
                    plt.colorbar(label="Gaze Density")
                    plt.xlabel("X"); plt.ylabel("Y"); plt.tight_layout()
                    plt.savefig(path); plt.close()
                    saved.append(path)
                except Exception as e:
                    # Fallback scatter if KDE fails
                    plt.figure(figsize=(8, 6))
                    plt.title(f"Gaze Heatmap (scatter) - Peg {i+1}")
                    if pts.shape[0] > 0:
                        plt.scatter(pts[:, 0], pts[:, 1], s=5)
                    plt.xlabel("X"); plt.ylabel("Y"); plt.tight_layout()
                    plt.savefig(path); plt.close()
                    saved.append(path)
            else:
                # Not enough points: save informative placeholder scatter
                plt.figure(figsize=(8, 6))
                plt.title(f"Gaze Heatmap (insufficient data) - Peg {i+1}")
                if pts.shape[0] == 1:
                    plt.scatter(pts[:, 0], pts[:, 1], s=10)
                plt.xlabel("X"); plt.ylabel("Y"); plt.tight_layout()
                plt.savefig(path); plt.close()
                saved.append(path)

        QtWidgets.QMessageBox.information(self, "Saved", f"Saved per-peg heatmaps:\n" + "\n".join(saved))

    def save_gaze_heatmap_all(self):
        if not self.frames:
            QtWidgets.QMessageBox.information(self, "No Data", "No gaze data available.")
            return

        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde

        # All endpoints (same behavior as before)
        gaze_points = self._collect_gaze_endpoints()
        if not gaze_points:
            QtWidgets.QMessageBox.information(self, "No Data", "No valid gaze points found.")
            return

        points = np.array(gaze_points)
        x, y = points[:, 0], points[:, 1]
        xy = np.vstack([x, y])
        try:
            kde = gaussian_kde(xy)
            xi, yi = np.mgrid[x.min():x.max():500j, y.min():y.max():500j]
            zi = kde(np.vstack([xi.flatten(), yi.flatten()]))

            plt.figure(figsize=(8, 6))
            plt.title("Gaze Position Heatmap (All Pegs, X-Y Plane)")
            plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading="auto", cmap="viridis")
            plt.colorbar(label="Gaze Density")
            plt.xlabel("X"); plt.ylabel("Y"); plt.tight_layout()
            save_path = "gaze_heatmap.png"  # keep prior filename
            plt.savefig(save_path); plt.close()
            QtWidgets.QMessageBox.information(self, "Saved", f"Saved: {os.path.abspath(save_path)}")
        except Exception:
            # Fallback scatter
            plt.figure(figsize=(8, 6))
            plt.title("Gaze Position Heatmap (scatter)")
            plt.scatter(x, y, s=5)
            plt.xlabel("X"); plt.ylabel("Y"); plt.tight_layout()
            save_path = "gaze_heatmap.png"
            plt.savefig(save_path); plt.close()
            QtWidgets.QMessageBox.information(self, "Saved", f"Saved (scatter): {os.path.abspath(save_path)}")

    def generate_fixation_heatmap(self):
        if not self.fixation_points:
            QtWidgets.QMessageBox.information(self, "No Data", "No fixation points recorded.")
            return
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde

        points = np.array(self.fixation_points)
        x, y = points[:, 0], points[:, 1]
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy)
        xi, yi = np.mgrid[x.min():x.max():500j, y.min():y.max():500j]
        zi = kde(np.vstack([xi.flatten(), yi.flatten()]))

        plt.figure(figsize=(8, 6))
        plt.title("Gaze Fixation Heatmap")
        plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading="auto", cmap="hot")
        plt.colorbar(label="Fixation Density")
        plt.xlabel("X"); plt.ylabel("Y"); plt.tight_layout()
        save_path = "fixation_heatmap.png"
        plt.savefig(save_path); plt.close()
        QtWidgets.QMessageBox.information(self, "Saved", f"Saved: {os.path.abspath(save_path)}")

    # ===== trajectories =====

    def _toggle_trajectory_filter_enabled(self, state):
        enabled = state == QtCore.Qt.Checked
        self.trajectory_group.setEnabled(enabled)
        self.update_frame(self.current_index if self.frames else None)

    def load_recording(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Recording", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "r") as f:
                self.frames = json.load(f)
            self.scrubber.setMaximum(len(self.frames) - 1 if self.frames else 0)
            self.current_index = 0
            self.update_frame()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load Error", f"Failed to load recording: {e}")

    def update_frame(self, idx=None):
        if not self.frames:
            return

        if idx is None or not (0 <= idx < len(self.frames)):
            idx = 0
        self.current_index = idx
        frame = self.frames[idx]

        # Fixation accumulation (3 consecutive intercept frames)
        current_intercept = frame["gaze_data"]["intercept"] == 1
        self.intercept_history.append(current_intercept)
        if len(self.intercept_history) > 3:
            self.intercept_history.pop(0)
        if len(self.intercept_history) >= 3 and all(self.intercept_history[-3:]):
            end_point = np.array(frame["gaze_data"]["gaze_line"][1])
            if not self.fixation_points or not np.array_equal(self.fixation_points[-1], end_point):
                self.fixation_points.append(end_point)

        gaze_line = np.array(frame["gaze_data"]["gaze_line"])
        intercept = frame["gaze_data"]["intercept"]
        peg_data = np.array(frame["peg_data"])
        A, B = gaze_line

        if not hasattr(self, "_actors_initialized"):
            self.plotter.clear()
            self.plotter.show_axes()
            self.plotter.show_grid()

            self.gaze_line_mesh = pv.Line(A, B)
            self.gaze_line_actor = self.plotter.add_mesh(self.gaze_line_mesh, color="green", line_width=3)

            direction = B - A
            self.last_direction = direction / np.linalg.norm(direction)
            self.disc_mesh = pv.Disc(center=B, inner=0.0, outer=DISC_RADIUS, normal=self.last_direction)
            self.disc_actor = self.plotter.add_mesh(self.disc_mesh, color="cyan", opacity=0.5)

            self.cone_center = A + 0.5 * (B - A)
            self.cone_mesh = pv.Cone(center=self.cone_center, direction=-self.last_direction,
                                     height=float(np.linalg.norm(direction)), radius=DISC_RADIUS)
            self.cone_actor = self.plotter.add_mesh(
                self.cone_mesh, color="green" if intercept else "red", opacity=0.3
            )

            if peg_data.shape == (6, 3):
                self.peg_mesh = pv.PolyData(peg_data)
                self.peg_actor = self.plotter.add_mesh(
                    self.peg_mesh, color="purple", point_size=12, render_points_as_spheres=True
                )

            # Trajectory actors (one per peg)
            self.trajectory_meshes = []
            self.trajectory_actors = []
            peg_count = len(self.frames[0]["peg_data"])
            for _ in range(peg_count):
                mesh = pv.PolyData()
                self.trajectory_meshes.append(mesh)
                actor = self.plotter.add_mesh(mesh, color="blue", line_width=3)
                self.trajectory_actors.append(actor)

            self._actors_initialized = True
        else:
            # Update existing
            self.gaze_line_mesh.points = pv.Line(A, B).points

            direction = B - A
            self.last_direction = direction / np.linalg.norm(direction)
            self.disc_mesh.points = pv.Disc(center=B, inner=0.0, outer=DISC_RADIUS,
                                            normal=self.last_direction).points

            self.cone_center = A + 0.5 * (B - A)
            self.cone_mesh.points = pv.Cone(
                center=self.cone_center, direction=-self.last_direction,
                height=float(np.linalg.norm(direction)), radius=DISC_RADIUS
            ).points

            color = 'green' if intercept else 'red'
            self.cone_actor.prop.color = color

            if peg_data.shape == (6, 3) and hasattr(self, 'peg_mesh'):
                self.peg_mesh.points = peg_data

        # Trajectories (respect filter)
        if hasattr(self, "trajectory_checkbox") and self.trajectory_checkbox.isChecked():
            peg_count = len(self.frames[0]["peg_data"])
            peg_trails = [[] for _ in range(peg_count)]
            for fr in self.frames[:idx + 1]:
                for i, peg in enumerate(fr["peg_data"]):
                    peg_trails[i].append(peg)

            selected = 6  # All by default
            if self.trajectory_group.isEnabled():
                selected = self.traj_button_group.checkedId()  # 0..5 or 6 (All)

            for i, trail in enumerate(peg_trails):
                show_this = (selected == 6) or (i == selected)
                # toggle visibility
                try:
                    self.trajectory_actors[i].SetVisibility(bool(show_this))
                except Exception:
                    pass
                if not show_this:
                    continue
                if len(trail) >= 2 and i < len(self.trajectory_meshes):
                    arr = np.array(trail)
                    if arr.ndim == 2 and arr.shape[1] == 3:
                        spline = pv.Spline(arr, max(len(arr) * 10, 20))
                        if len(spline.points) > 0 and len(spline.lines) > 0:
                            self.trajectory_meshes[i].points = spline.points
                            self.trajectory_meshes[i].lines = spline.lines
                            self.trajectory_actors[i].mapper.dataset = self.trajectory_meshes[i]
        else:
            # hide all trajectory actors when unchecked
            if hasattr(self, "trajectory_actors"):
                for a in self.trajectory_actors:
                    try:
                        a.SetVisibility(False)
                    except Exception:
                        pass

        if hasattr(self, '_actors_initialized'):
            self.plotter.render()

    def reset_view(self):
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        self.plotter.render()

    def _zoom(self, factor: float):
        if hasattr(self.plotter.camera, "Zoom"):
            self.plotter.camera.Zoom(factor)
        self.plotter.render()

    def fix_view_bounds(self):
        if hasattr(self, "fix_view_actors"):
            for actor in self.fix_view_actors:
                self.plotter.remove_actor(actor)
            self.fix_view_actors = []
        else:
            self.fix_view_actors = []

        if self.fix_view_checkbox.isChecked():
            bounds = np.array([[-100, -100, -100], [600, 600, 600]], dtype=np.float32)
            origin = np.array([[0, 0, 0]], dtype=np.float32)

            bounds_actor = self.plotter.add_points(
                bounds, color="white", opacity=0.0, point_size=1, render_points_as_spheres=True
            )
            origin_actor = self.plotter.add_points(
                origin, color="black", point_size=10, render_points_as_spheres=True
            )
            self.fix_view_actors.extend([bounds_actor, origin_actor])

        self.plotter.render()

    def play(self):
        if not self.frames:
            return
        self.is_playing = True
        self.timer.start()

    def pause(self):
        self.is_playing = False
        self.timer.stop()

    def restart(self):
        self.current_index = 0
        self.scrubber.setValue(0)
        self.update_frame()
        self.play()

    def advance_frame(self):
        if not self.frames or not self.is_playing:
            return
        self.current_index += 1
        if self.current_index >= len(self.frames):
            self.timer.stop()
            self.is_playing = False
            return
        self.scrubber.setValue(self.current_index)


# =====================================
# ======== CALIB/VALIDATION ===========
# =====================================

class MainWindow(QtWidgets.QWidget):
    """Calibration / Validation Window."""
    trigger_matrix_mode = QtCore.pyqtSignal()

    def __init__(self, enable_receiver=True):
        super().__init__()
        self.setWindowTitle("Real-Time Point Viewer")
        self.setMinimumSize(980, 680)
        self.trigger_matrix_mode.connect(self.handle_matrix_mode)

        # Data storage
        self.camera_points: List[np.ndarray] = []
        self.holo_points: List[np.ndarray] = []
        self.matrix_transform_points: List[np.ndarray] = []
        self.manual_points: List[np.ndarray] = []
        self.manual_transformed_points: List[np.ndarray] = []
        self.peg_validation_point = None
        self.peg_validation_actor = None
        self.transform_matrices = []
        self.fix_view_actors = []
        self.receiver = None

        if enable_receiver:
            self.receiver = DataReceiver(parent=self)
            self.receiver.updated_points.connect(self.set_all_pairs)
            self.receiver.peg_point_received.connect(self.set_peg_validation_point)
            self.receiver.validation_gaze_received.connect(self.update_validation_gaze)
            self.receiver.full_gaze_received.connect(self.receive_gaze_data)
            self.receiver.start()

        # UI setup
        self.plotter = QtInteractor(self)
        self.plotter.show_axes()
        self.plotter.show_grid()
        self.plotter.enable_point_picking(
            callback=self._on_point_picked, use_picker=True,
            show_message=False, left_clicking=True, show_point=True
        )

        self.render_timer = QtCore.QTimer(self)
        self.render_timer.setInterval(60)
        self.render_timer.timeout.connect(self.plotter.render)
        self.render_timer.start()

        # Dashed line (validation peg -> gaze)
        self.dashed_segments = 10
        self.validation_dashed_meshes = []
        self.validation_dashed_actors = []
        for _ in range(self.dashed_segments):
            mesh = pv.PolyData()
            mesh.points = pv.pyvista_ndarray([[0, 0, 0], [0, 0, 0]])
            mesh.lines = np.array([2, 0, 1])
            actor = self.plotter.add_mesh(mesh, color="red", line_width=2)
            self.validation_dashed_meshes.append(mesh)
            self.validation_dashed_actors.append(actor)

        # Options
        self.cam_checkbox = QtWidgets.QCheckBox("Show Camera Points"); self.cam_checkbox.setChecked(True)
        self.holo_checkbox = QtWidgets.QCheckBox("Show HoloLens Points"); self.holo_checkbox.setChecked(True)

        self.reset_button = QtWidgets.QPushButton("Reset View")
        self.zoom_in_button = QtWidgets.QPushButton("Zoom In")
        self.zoom_out_button = QtWidgets.QPushButton("Zoom Out")
        self.fix_view_checkbox = QtWidgets.QCheckBox("Fix View"); self.fix_view_checkbox.setChecked(False)

        self.gaze_distance_label = QtWidgets.QLabel("Gaze Distance: N/A")
        self.gaze_distance_label.setProperty("kind", "meter")

        options_group = QtWidgets.QGroupBox("Options")
        options_layout = QtWidgets.QVBoxLayout()
        options_layout.addWidget(self.cam_checkbox)
        options_layout.addWidget(self.holo_checkbox)
        options_group.setLayout(options_layout)

        # Transform config
        transform_group = QtWidgets.QGroupBox("Transform")
        transform_layout = QtWidgets.QVBoxLayout()
        self.transform_checkbox = QtWidgets.QCheckBox("Show Transform Point"); self.transform_checkbox.setChecked(True)

        fields_layout = QtWidgets.QHBoxLayout()
        self.transform_x = QtWidgets.QLineEdit(); self.transform_x.setPlaceholderText("X"); self.transform_x.setMaximumWidth(80)
        self.transform_y = QtWidgets.QLineEdit(); self.transform_y.setPlaceholderText("Y"); self.transform_y.setMaximumWidth(80)
        self.transform_z = QtWidgets.QLineEdit(); self.transform_z.setPlaceholderText("Z"); self.transform_z.setMaximumWidth(80)
        fields_layout.addWidget(self.transform_x); fields_layout.addWidget(self.transform_y); fields_layout.addWidget(self.transform_z)

        self.transform_apply = QtWidgets.QPushButton("Create Point")
        self.clear_transform_button = QtWidgets.QPushButton("Clear Transform Points")

        self.load_matrices_button = QtWidgets.QPushButton("Load Matrices")
        self.load_matrices_button.clicked.connect(self.load_matrices_from_file)

        self.matrix_buttons = []
        self.matrix_group = QtWidgets.QButtonGroup()
        for i in range(4):
            btn = QtWidgets.QRadioButton(MATRIX_BUTTON_LABELS[i])
            self.matrix_buttons.append(btn)
            self.matrix_group.addButton(btn, i)
            transform_layout.addWidget(btn)
        self.matrix_buttons[0].setChecked(True)

        self.matrix_apply_button = QtWidgets.QPushButton("Transform")

        transform_layout.addWidget(self.transform_checkbox)
        transform_layout.addWidget(self.clear_transform_button)
        transform_layout.addWidget(self.load_matrices_button)
        transform_layout.addWidget(self.matrix_apply_button)
        transform_layout.addLayout(fields_layout)
        transform_layout.addWidget(self.transform_apply)
        transform_group.setLayout(transform_layout)

        # View group
        view_group = QtWidgets.QGroupBox("View")
        view_layout = QtWidgets.QHBoxLayout()
        view_layout.addWidget(self.reset_button)
        view_layout.addWidget(self.zoom_in_button)
        view_layout.addWidget(self.zoom_out_button)
        view_layout.addWidget(self.fix_view_checkbox)
        view_group.setLayout(view_layout)

        # Right panel
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(options_group)
        right_layout.addWidget(transform_group)
        right_layout.addWidget(view_group)
        right_layout.addStretch()
        right_layout.addWidget(self.gaze_distance_label)

        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.plotter.interactor, 1)
        layout.addLayout(right_layout)

        # Signals
        self.cam_checkbox.stateChanged.connect(self.update_scene)
        self.holo_checkbox.stateChanged.connect(self.update_scene)
        self.transform_checkbox.stateChanged.connect(self.update_scene)
        self.reset_button.clicked.connect(self.reset_view)
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.transform_apply.clicked.connect(self.apply_transform)
        self.clear_transform_button.clicked.connect(self.clear_transform_points)
        self.matrix_apply_button.clicked.connect(self.apply_matrix_transform)
        self.fix_view_checkbox.stateChanged.connect(self.update_scene)

        self.latest_validation_gaze = None
        self.latest_validation_roi = 0.0
        self.latest_validation_intercept = 0

    def closeEvent(self, event):
        if self.receiver:
            self.receiver.stop()
            self.receiver.wait()
        super().closeEvent(event)

    @QtCore.pyqtSlot(list, list)
    def set_all_pairs(self, camera_points, holo_points):
        self.camera_points = camera_points
        self.holo_points = holo_points
        self.update_scene()

    @QtCore.pyqtSlot(np.ndarray)
    def set_peg_validation_point(self, point):
        self.peg_validation_point = point

    @QtCore.pyqtSlot(object, float, int, float)
    def update_validation_gaze(self, gaze_line, roi, intercept, gaze_distance):
        try:
            gaze_arr = _safe_array(gaze_line, (2, 3), "MainWindow.validation_gaze")
            if gaze_arr is None:
                return
            origin = gaze_arr[0]
            direction = gaze_arr[1]
            length = np.linalg.norm(direction)
            if length == 0:
                target = origin
            else:
                gaze_length = GAZE_LINE_LENGTH
                if self.peg_validation_point is not None:
                    idx = self.matrix_group.checkedId()
                    if 0 <= idx < len(self.transform_matrices):
                        pt_h = np.append(self.peg_validation_point, 1.0)
                        transformed = (self.transform_matrices[idx] @ pt_h)[:3]
                        gaze_length = np.linalg.norm(transformed)
                target = origin + (direction / max(length, 1e-9)) * gaze_length

            self.latest_validation_gaze = np.array([origin, target])
            self.latest_validation_roi = float(roi)
            self.latest_validation_intercept = int(intercept)
            self.gaze_distance_label.setText(f"Gaze Distance: {gaze_distance:.2f} mm")
            self._update_validation_gaze_line()
        except Exception as e:
            log.warning("[MainWindow] Failed to update validation gaze visuals: %s", e)

    def _update_validation_gaze_line(self):
        if self.latest_validation_gaze is None:
            return

        A, B = self.latest_validation_gaze
        direction = B - A
        length = np.linalg.norm(direction)
        if length == 0:
            return
        norm_direction = direction / length
        cone_center = A + 0.5 * (B - A)
        roi = float(self.latest_validation_roi)

        line = pv.Line(A, B)
        if hasattr(self, "validation_line_mesh"):
            self.validation_line_mesh.deep_copy(line)
            self.validation_line_mesh.Modified()
        else:
            self.validation_line_mesh = line
            self.validation_gaze_line_actor = self.plotter.add_mesh(
                self.validation_line_mesh, color="green", line_width=3
            )

        disc = pv.Disc(center=B, inner=0.0, outer=DISC_RADIUS, normal=norm_direction, r_res=1, c_res=10)
        if hasattr(self, "validation_disc_mesh"):
            self.validation_disc_mesh.deep_copy(disc)
            self.validation_disc_mesh.Modified()
        else:
            self.validation_disc_mesh = disc
            self.validation_disc_actor = self.plotter.add_mesh(self.validation_disc_mesh, color="cyan", opacity=0.5)

        cone_color = "green" if self.latest_validation_intercept else "red"
        cone = pv.Cone(center=cone_center, direction=-norm_direction, height=float(length), radius=DISC_RADIUS)
        if hasattr(self, "validation_cone_mesh"):
            self.validation_cone_mesh.deep_copy(cone)
            self.validation_cone_mesh.Modified()
            if hasattr(self, "validation_cone_actor"):
                self.validation_cone_actor.GetProperty().SetColor(Color(cone_color).float_rgb)
        else:
            self.validation_cone_mesh = cone
            self.validation_cone_actor = self.plotter.add_mesh(self.validation_cone_mesh, color=cone_color, opacity=0.3)

        if self.peg_validation_point is not None:
            point = self.peg_validation_point

            if not hasattr(self, "peg_validation_mesh"):
                self.peg_validation_mesh = pv.PolyData(point.reshape(1, 3))
                self.peg_validation_actor = self.plotter.add_mesh(
                    self.peg_validation_mesh, color="purple", point_size=12, render_points_as_spheres=True
                )
            else:
                self.peg_validation_mesh.points = point.reshape(1, 3)
                self.peg_validation_mesh.Modified()

            idx = self.matrix_group.checkedId()
            if 0 <= idx < len(self.transform_matrices):
                matrix = self.transform_matrices[idx]
                pt_h = np.append(point, 1.0)
                transformed = (matrix @ pt_h)[:3]

                if not hasattr(self, "peg_transformed_mesh"):
                    self.peg_transformed_mesh = pv.PolyData(transformed.reshape(1, 3))
                    self.peg_transformed_actor = self.plotter.add_mesh(
                        self.peg_transformed_mesh, color="#300053", point_size=12, render_points_as_spheres=True
                    )
                else:
                    self.peg_transformed_mesh.points = transformed.reshape(1, 3)
                    self.peg_transformed_mesh.Modified()

                sphere = pv.Sphere(radius=roi, center=transformed)
                if hasattr(self, "validation_sphere_mesh"):
                    self.validation_sphere_mesh.deep_copy(sphere)
                    self.validation_sphere_mesh.Modified()
                else:
                    self.validation_sphere_mesh = sphere
                    self.validation_sphere_actor = self.plotter.add_mesh(
                        self.validation_sphere_mesh, color="green", opacity=0.2
                    )

        if self.peg_validation_point is not None and self.latest_validation_gaze is not None:
            peg = self.peg_validation_point

            def closest_point_on_line(p, a, b):
                ab = b - a
                denom = np.dot(ab, ab)
                if denom == 0:
                    return a
                t = np.dot(p - a, ab) / denom
                return a + np.clip(t, 0, 1) * ab

            closest_point = closest_point_on_line(peg, A, B)
            pairs = np.linspace(closest_point, peg, self.dashed_segments * 2).reshape(-1, 2, 3)

            dash_idx = 0
            for i in range(0, len(pairs), 2):
                start, end = pairs[i]
                mesh = self.validation_dashed_meshes[dash_idx]
                mesh.points = np.array([start, end])
                mesh.lines = np.array([2, 0, 1])
                mesh.Modified()
                dash_idx += 1
                if dash_idx >= len(self.validation_dashed_meshes):
                    break
            for j in range(dash_idx, len(self.validation_dashed_meshes)):
                self.validation_dashed_meshes[j].points = np.array([[0, 0, 0], [0, 0, 0]])
                self.validation_dashed_meshes[j].Modified()

    @QtCore.pyqtSlot(object, float, int, float, object)
    def receive_gaze_data(self, gaze_line, roi, intercept, gaze_distance, pegs):
        if hasattr(self, "gaze_window") and self.gaze_window:
            self.gaze_window.update_gaze_data(gaze_line, roi, intercept, gaze_distance, pegs)

    # ===== UI actions =====

    def reset_view(self):
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        self.plotter.render()

    def _on_point_picked(self, picked_point, picker):
        if picked_point is not None:
            coord_str = f"Picked Point: ({picked_point[0]:.2f}, {picked_point[1]:.2f}, {picked_point[2]:.2f})"
            QtWidgets.QMessageBox.information(self, "Point Coordinates", coord_str)
            log.info(coord_str)

    def zoom_in(self):
        self._zoom(1.2)

    def zoom_out(self):
        self._zoom(0.8)

    def handle_matrix_mode(self):
        matrix_path = HARD_PATH_GAZE
        try:
            result = load_transform_matrices_from_file(matrix_path)
            self.transform_matrices = result["matrices"]

            if result["camera_points"] and result["hololens_points"]:
                self.set_all_pairs(result["camera_points"], result["hololens_points"])

            self.matrix_info_window = MatrixInfoWindow(
                self.transform_matrices, result["rows"], result["rmse"], result["repro"]
            )
            self.matrix_info_window.show()
            log.info("[MainWindow] Transform matrices loaded successfully.")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Matrix Load Failed", str(e))
            log.warning("[MainWindow] Error loading matrices: %s", e)

    def apply_matrix_transform(self):
        idx = self.matrix_group.checkedId()
        if idx < 0 or idx >= len(self.transform_matrices):
            log.warning("[MainWindow] Invalid matrix index selected.")
            return

        matrix = self.transform_matrices[idx]
        transformed = []
        for pt in self.camera_points:
            pt_h = np.append(pt, 1.0)
            new_pt = matrix @ pt_h
            transformed.append(new_pt[:3])

        self.matrix_transform_points = transformed
        self.update_scene()

    def apply_transform(self):
        try:
            x = float(self.transform_x.text())
            y = float(self.transform_y.text())
            z = float(self.transform_z.text())
        except ValueError:
            return

        manual_point = np.array([x, y, z])
        self.manual_points.append(manual_point)

        idx = self.matrix_group.checkedId()
        if 0 <= idx < len(self.transform_matrices):
            matrix = self.transform_matrices[idx]
            pt_h = np.append(manual_point, 1.0)
            transformed = matrix @ pt_h
            transformed_point = transformed[:3]
            self.manual_transformed_points.append(transformed_point)

        self.update_scene()

    def clear_transform_points(self):
        self.matrix_transform_points.clear()
        self.manual_points.clear()
        self.manual_transformed_points.clear()
        self.update_scene()

    def load_matrices_from_file(self):
        path = HARD_PATH_CALIB
        if not os.path.isfile(path):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Transform JSON", "", "JSON (*.txt *.json)")
        if not path:
            QtWidgets.QMessageBox.warning(self, "Load Failed", "No file selected.")
            return

        try:
            result = load_transform_matrices_from_file(path)
            self.transform_matrices = result["matrices"]

            if result["camera_points"] and result["hololens_points"]:
                self.set_all_pairs(result["camera_points"], result["hololens_points"])

            self.matrix_info_window = MatrixInfoWindow(
                self.transform_matrices, result["rows"], result["rmse"], result["repro"]
            )
            self.matrix_info_window.show()
            QtWidgets.QMessageBox.information(self, "Matrices Loaded", "Successfully loaded matrices.")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load Failed", str(e))
            log.warning("[MainWindow] Failed to load matrices: %s", e)

    def _zoom(self, factor: float):
        camera = self.plotter.camera
        if hasattr(camera, "Zoom"):
            camera.Zoom(factor)
        self.plotter.render()

    def draw_dashed_line(self, p1, p2, segments=30):
        points = np.linspace(p1, p2, segments * 2).reshape(-1, 2, 3)
        for i, (start, end) in enumerate(points):
            if i % 2 == 0:
                self.plotter.add_lines(np.array([start, end]), color="gray", width=4)

    def update_scene(self):
        self.plotter.clear()
        self.plotter.show_axes()
        self.plotter.show_grid()

        if self.camera_points:
            self.plotter.add_points(
                np.vstack(self.camera_points), color="red", point_size=12,
                opacity=1.0 if self.cam_checkbox.isChecked() else 0.0,
                render_points_as_spheres=True, pickable=True
            )

        if self.holo_points:
            self.plotter.add_points(
                np.vstack(self.holo_points), color="blue", point_size=12,
                opacity=1.0 if self.holo_checkbox.isChecked() else 0.0,
                render_points_as_spheres=True, pickable=True
            )

        if self.transform_checkbox.isChecked():
            if self.matrix_transform_points:
                self.plotter.add_points(
                    np.vstack(self.matrix_transform_points),
                    color="green", point_size=12,
                    render_points_as_spheres=True, pickable=True
                )
            if self.manual_points:
                self.plotter.add_points(
                    np.vstack(self.manual_points),
                    color="#CBCB00", point_size=12,
                    render_points_as_spheres=True, pickable=True
                )
            if self.manual_transformed_points:
                self.plotter.add_points(
                    np.vstack(self.manual_transformed_points),
                    color="#C48300", point_size=12,
                    render_points_as_spheres=True, pickable=True
                )

        if self.camera_points and self.holo_points and self.cam_checkbox.isChecked() and self.holo_checkbox.isChecked():
            for cam, holo in zip(self.camera_points, self.holo_points):
                self.draw_dashed_line(cam, holo)

        for actor in self.fix_view_actors:
            self.plotter.remove_actor(actor)
        self.fix_view_actors.clear()

        if self.fix_view_checkbox.isChecked():
            invisible_bounds = np.array([[-100, -100, -100], [600, 600, 600]], dtype=np.float32)
            origin = np.array([[0, 0, 0]], dtype=np.float32)

            bounds_actor = self.plotter.add_points(
                invisible_bounds, color="white", opacity=0.0, point_size=1, render_points_as_spheres=True
            )
            origin_actor = self.plotter.add_points(
                origin, color="black", point_size=10, render_points_as_spheres=True
            )
            self.fix_view_actors.extend([bounds_actor, origin_actor])

        self.plotter.render()


# =====================================
# ============ MAIN MENU ==============
# =====================================

class MainMenuWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mixed Reality Surgical Training System 3D")
        self.setMinimumSize(520, 360)

        layout = QtWidgets.QVBoxLayout()

        info_label = QtWidgets.QLabel(
            "<h2>Welcome to the Surgical Training GUI 3D</h2>"
            "<p>Select a mode to begin:</p>"
        )
        info_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(info_label)

        btn_layout = QtWidgets.QVBoxLayout()
        calibration_btn = QtWidgets.QPushButton("Calibration / Validation Mode")
        gaze_tracking_btn = QtWidgets.QPushButton("Gaze Tracking Mode")
        playback_btn = QtWidgets.QPushButton("Playback Mode")
        calibration_btn.setMinimumHeight(36)
        gaze_tracking_btn.setMinimumHeight(36)
        playback_btn.setMinimumHeight(36)

        calibration_btn.clicked.connect(self.launch_calibration)
        gaze_tracking_btn.clicked.connect(self.launch_gaze_tracking)
        playback_btn.clicked.connect(self.launch_playback)

        btn_layout.addWidget(calibration_btn)
        btn_layout.addWidget(gaze_tracking_btn)
        btn_layout.addWidget(playback_btn)

        layout.addStretch()
        layout.addLayout(btn_layout)
        layout.addStretch()

        self.setLayout(layout)

    def launch_calibration(self):
        self.calibration_window = MainWindow()
        self.calibration_window.show()

    def launch_gaze_tracking(self):
        self.gaze_window = GazeTrackingWindow(enable_receiver=True)
        self.gaze_window.show()

    def launch_playback(self):
        self.playback_window = PlaybackWindow()
        self.playback_window.show()


# =====================================
# =============== MAIN ================
# =====================================

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    apply_app_style(app)
    main_menu = MainMenuWindow()
    main_menu.resize(560, 400)
    main_menu.show()
    sys.exit(app.exec_())
