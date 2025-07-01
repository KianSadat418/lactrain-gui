import sys
import socket
from datetime import datetime
import json
from typing import List
import numpy as np

import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets
import pyvista as pv
from pyvista import Color
pv.global_theme.allow_empty_mesh = True
from pyvistaqt import QtInteractor

MATRIX_BUTTON_LABELS = [
            "Similarity Xform",
            "Affine Xform",
            "Similarity Xform RANSAC",
            "Affine Xform RANSAC"
        ]

# Length of the gaze ray and radius of the disc at the end of the line
GAZE_LINE_LENGTH = 500.0
DISC_RADIUS = 40.0

def load_transform_matrices_from_file(path: str):
    import os

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


class Recorder:
    def __init__(self):
        self.frames = []
        self.recording = False

    def start(self):
        self.frames = []
        self.recording = True
        print("[Recorder] Started recording.")

    def stop(self):
        self.recording = False
        print("[Recorder] Stopped recording.")

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
        print(f"[Recorder] Saved {len(self.frames)} frames to {filepath}")


class DataReceiver(QtCore.QThread):
    updated_points = QtCore.pyqtSignal(list, list)
    peg_point_received = QtCore.pyqtSignal(np.ndarray)
    validation_gaze_received = QtCore.pyqtSignal(object, float, int, float)
    full_gaze_received = QtCore.pyqtSignal(object, float, int, float, object)

    def __init__(self, host="0.0.0.0", port=9991, parent=None):
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
                print(f"[Receiver] Error closing socket: {e}")
            self.server_socket = None

    def run(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.setblocking(False)
        except Exception as e:
            print(f"[Receiver] Failed to start server: {e}")
            return

        while self._running:
            try:
                data, addr = self.server_socket.recvfrom(650000)
            except BlockingIOError:
                QtCore.QThread.msleep(1)
                continue
            except Exception as e:
                print(f"[Receiver] Socket error: {e}")
                break

            if not data:
                continue

            try:
                line = data.decode("utf-8").strip()

                if line.startswith("M"):
                    QtCore.QMetaObject.invokeMethod(self.parent(), "trigger_matrix_mode", QtCore.Qt.QueuedConnection)

                elif line.startswith("V"):
                    data = json.loads(line[1:])
                    point = np.array([float(data["x"]), float(data["y"]), float(data["z"])])
                    self.peg_point_received.emit(point)
                    self.validation_gaze_received.emit(
                        np.array(data.get("gaze_line")),
                        float(data.get("roi", 0.0)),
                        int(data.get("intercept", 0)),
                        float(data.get("gaze_distance", 0.0))
                    )

                elif line.startswith("G"):
                    data = json.loads(line[1:])
                    self.full_gaze_received.emit(
                        np.array(data.get("gaze_line")),
                        float(data.get("roi", 0.0)),
                        int(data.get("intercept", 0)),
                        float(data.get("gaze_distance", 0.0)),
                        np.array(data.get("pegs", []))
                    )

                else:
                    raw = json.loads(line)
                    camera_points = [np.array(v[0]) for v in raw.values()]
                    holo_points = [np.array(v[1]) for v in raw.values()]
                    self.updated_points.emit(camera_points, holo_points)

            except Exception as e:
                print(f"[Receiver] Failed to process message: {e}")

        print("[Receiver] Shutting down cleanly.")


class MatrixInfoWindow(QtWidgets.QWidget):
    def __init__(self, matrix_data, point_rows, rmse_values, repro_values, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Transformation Matrix Info")
        self.setMinimumWidth(1000)
        layout = QtWidgets.QVBoxLayout()

        # Table setup
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
        layout.addWidget(table)

        # Matrices and RMSE
        matrix_layout = QtWidgets.QFormLayout()
        for name, mat, rmse, repro in zip(
            ["Similarity", "Affine", "Similarity RANSAC", "Affine RANSAC"],
            matrix_data,
            rmse_values,
            repro_values
        ):
            mat_str = "\n".join("  ".join(f"{v:.4f}" for v in row) for row in mat)
            matrix_label = QtWidgets.QLabel(f"<pre>{mat_str}</pre>")
            matrix_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            matrix_layout.addRow(f"{name} Matrix (RMSE: {rmse:.4f}, Repro Error: {repro:.4f})", matrix_label)

        layout.addLayout(matrix_layout)
        self.setLayout(layout)


class GazeTrackingWindow(QtWidgets.QWidget):
    def __init__(self, transform_matrices=None, enable_receiver=True, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gaze Tracking")
        self.setMinimumSize(800, 600)

        # === Core Attributes ===
        self.recorder = Recorder()
        self.transform_matrices = transform_matrices or []
        self.latest_gaze_line = None
        self.latest_roi = 0.0
        self.latest_intercept = 0
        self.latest_pegs = []
        self.latest_gaze_distance = 0.0
        self.fix_view_actors = []

        self.receiver = None
        if enable_receiver:
            self.receiver = DataReceiver(parent=self)
            self.receiver.full_gaze_received.connect(self.update_gaze_data)
            self.receiver.start()

        # === Plotter ===
        self.plotter = QtInteractor(self)

        # Original Peg Mesh (hidden)
        self.peg_mesh = pv.PolyData(np.zeros((6, 3)))
        self.peg_mesh_actor = self.plotter.add_mesh(self.peg_mesh, color="purple", point_size=12, render_points_as_spheres=True)
        self.peg_mesh_actor.SetVisibility(False)

        # Transformed Peg Mesh (all 6)
        self.transformed_peg_mesh = pv.PolyData(np.zeros((6, 3)))
        self.transformed_peg_actor = self.plotter.add_mesh(self.transformed_peg_mesh, color="#300053", point_size=12, render_points_as_spheres=True)

        # === Dashed Lines ===
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
        self.zoom_in_button = QtWidgets.QPushButton("+")
        self.zoom_out_button = QtWidgets.QPushButton("-")
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

         # === Load Matrices Button ===
        self.load_matrices_button = QtWidgets.QPushButton("Load Matrices")
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

        # === Peg of Interest Selection ===
        self.peg_of_interest_index = 0
        self.peg_selector_group = QtWidgets.QGroupBox("Peg of Interest")
        peg_radio_layout = QtWidgets.QVBoxLayout()
        self.peg_radio_buttons = []
        self.peg_button_group = QtWidgets.QButtonGroup()

        for i in range(6):
            btn = QtWidgets.QRadioButton(f"Peg {i}")
            self.peg_button_group.addButton(btn, i)
            self.peg_radio_buttons.append(btn)
            peg_radio_layout.addWidget(btn)

        self.peg_radio_buttons[0].setChecked(True)
        self.peg_button_group.buttonClicked[int].connect(self.set_peg_of_interest)

        self.peg_selector_group.setLayout(peg_radio_layout)

        # === Gaze Distance Label ===
        self.gaze_distance_label = QtWidgets.QLabel("Gaze Distance: N/A")

        # === Right Panel ===
        right_panel = QtWidgets.QVBoxLayout()
        right_panel.addWidget(transform_group)
        right_panel.addWidget(self.peg_selector_group)
        right_panel.addWidget(view_group)
        right_panel.addWidget(self.gaze_distance_label)
        right_panel.addStretch()

        # === Recording Controls ===
        self.start_recording_button = QtWidgets.QPushButton("Start Recording")
        self.stop_recording_button = QtWidgets.QPushButton("Stop & Save Recording")

        # Recording indicator
        self.record_indicator = QtWidgets.QLabel("● REC")
        self.record_indicator.setStyleSheet("color: red; font-weight: bold;")
        self.record_indicator.hide()

        right_panel.addWidget(self.start_recording_button)
        right_panel.addWidget(self.stop_recording_button)
        right_panel.addWidget(self.record_indicator)

        self.start_recording_button.clicked.connect(self._start_recording)
        self.stop_recording_button.clicked.connect(self._save_recording)

        # === Layouts ===
        viewer_layout = QtWidgets.QVBoxLayout()
        viewer_layout.addWidget(self.plotter.interactor)

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(viewer_layout, stretch=4)
        main_layout.addLayout(right_panel, stretch=1)
        self.setLayout(main_layout)

        # === Plotter Setup ===
        QtCore.QTimer.singleShot(0, self.plotter.show_axes)
        QtCore.QTimer.singleShot(0, self.plotter.show_grid)

        self.plotter.enable_point_picking(
            callback=self._on_point_picked,
            use_picker=True,
            show_message=False,
            left_clicking=True,
            show_point=True
        )

        # === Throttled Render Timer ===
        self.render_timer = QtCore.QTimer()
        self.render_timer.setInterval(100)  # 10 FPS
        self.render_timer.timeout.connect(self.plotter.render)
        self.render_timer.start()

        # Timer used to log frames while recording
        self.record_timer = QtCore.QTimer()
        self.record_timer.setInterval(100)
        self.record_timer.timeout.connect(self._log_current_frame)


    @QtCore.pyqtSlot(object, float, int, float, object)
    def update_gaze_data(self, gaze_line, roi, intercept, gaze_distance, pegs):
        try:
            gaze_arr = np.array(gaze_line)
            if gaze_arr.shape != (2, 3):
                print(f"[GazeTracking] Invalid gaze shape: {gaze_arr.shape}")
                return

            origin = gaze_arr[0]
            direction = gaze_arr[1]
            length = np.linalg.norm(direction)

            if length == 0:
                target = origin
            else:
                # Approximate dynamic line length using distance to transformed first peg
                gaze_length = GAZE_LINE_LENGTH
                if hasattr(self, "latest_pegs") and len(self.latest_pegs) > 0:
                    idx = self.matrix_group.checkedId()
                    if 0 <= idx < len(self.transform_matrices):
                        pt_h = np.append(self.latest_pegs[0], 1.0)
                        transformed = (self.transform_matrices[idx] @ pt_h)[:3]
                        gaze_length = np.linalg.norm(transformed)

                target = origin + (direction / length) * gaze_length

            self.latest_gaze_line = np.array([origin, target])
            self.latest_roi = roi
            self.latest_intercept = intercept
            self.latest_gaze_distance = gaze_distance
            peg_arr = np.array(pegs).reshape(-1, 3)
            self.latest_pegs = peg_arr
            self.peg_mesh.points = peg_arr
            self.peg_mesh.Modified()
            self.peg_mesh_actor.SetVisibility(True)

            self.gaze_distance_label.setText(f"Gaze Distance: {gaze_distance:.2f} mm")
            self._update_gaze_line()
        except Exception as e:
            print(f"[GazeTracking] Failed to update visuals: {e}")

    def _update_gaze_line(self):
        if self.latest_gaze_line is None or not hasattr(self, "latest_pegs"):
            return

        A, B = self.latest_gaze_line
        cone_center = A + 0.5 * (B - A)
        direction = B - A
        length = np.linalg.norm(direction)
        if length == 0:
            return
        norm_direction = direction / length
        height = float(length)

        # === 1. Gaze Line ===
        line = pv.Line(A, B)
        if hasattr(self, "gaze_line_mesh"):
            self.gaze_line_mesh.deep_copy(line)
            self.gaze_line_mesh.Modified()
        else:
            self.gaze_line_mesh = line
            self.gaze_line_actor = self.plotter.add_mesh(self.gaze_line_mesh, color="green", line_width=3)

        # === 2. Disc at End ===
        disc = pv.Disc(center=B, inner=0.0, outer=DISC_RADIUS, normal=norm_direction, r_res=1, c_res=10)
        if hasattr(self, "disc_mesh"):
            self.disc_mesh.deep_copy(disc)
            self.disc_mesh.Modified()
        else:
            self.disc_mesh = disc
            self.disc_actor = self.plotter.add_mesh(self.disc_mesh, color="yellow", opacity=0.5)

        # === 3. Cone from Origin ===
        cone_color = "green" if self.latest_intercept else "red"
        cone = pv.Cone(center=cone_center, direction=-norm_direction, height=height, radius=DISC_RADIUS)
        if hasattr(self, "cone_mesh"):
            self.cone_mesh.deep_copy(cone)
            self.cone_mesh.Modified()
            if hasattr(self, "cone_actor"):
                self.cone_actor.GetProperty().SetColor(Color(cone_color).float_rgb)
        else:
            self.cone_mesh = cone
            self.cone_actor = self.plotter.add_mesh(self.cone_mesh, color=cone_color, opacity=0.3)

        pegs = np.array(self.latest_pegs).reshape(-1, 3)

        # === 4. Transformed Pegs (all 6) ===
        idx = self.matrix_group.checkedId()
        if 0 <= idx < len(self.transform_matrices) and pegs.size >= 3:
            matrix = self.transform_matrices[idx]
            transformed_pegs = np.array([(matrix @ np.append(p, 1.0))[:3] for p in pegs])

            self.transformed_peg_mesh.deep_copy(pv.PolyData(transformed_pegs))
            self.transformed_peg_mesh.Modified()

            # === 5. ROI Sphere + Animated Line (peg of interest) ===
            peg = transformed_pegs[self.peg_of_interest_index] # peg of interest
            if hasattr(self, "roi_sphere_mesh"):
                sphere = pv.Sphere(radius=self.latest_roi, center=peg)
                self.roi_sphere_mesh.deep_copy(sphere)
                self.roi_sphere_mesh.Modified()
            else:
                self.roi_sphere_mesh = pv.Sphere(radius=self.latest_roi, center=peg)
                self.roi_sphere_actor = self.plotter.add_mesh(self.roi_sphere_mesh, color="green", opacity=0.2)

            def closest_point_on_line(p, a, b):
                ab = b - a
                t = np.dot(p - a, ab) / np.dot(ab, ab)
                t = np.clip(t, 0, 1)
                return a + t * ab

            closest_point = closest_point_on_line(peg, A, B)
            dashed_pairs = np.linspace(closest_point, peg, self.dashed_segments * 2).reshape(-1, 2, 3)

            for i in range(self.dashed_segments):
                if i < len(dashed_pairs) and i % 2 == 0:
                    start, end = dashed_pairs[i]
                    mesh = self.dashed_meshes[i // 2]
                    mesh.points = np.array([start, end])
                    mesh.lines = np.array([2, 0, 1])
                    mesh.Modified()
                else:
                    self.dashed_meshes[i // 2].points = np.array([[0, 0, 0], [0, 0, 0]])
                    self.dashed_meshes[i // 2].Modified()

    def set_peg_of_interest(self, index: int):
        self.peg_of_interest_index = index
        print(f"[GazeTracking] Peg of interest set to: {index}")
        self._update_gaze_line()

    def load_matrices_from_file(self):
        path = "Assets/Scripts/GUI/transform_data.txt"

        if not path:  # User cancelled or didn't choose anything
            QtWidgets.QMessageBox.warning(self, "Load Failed", "No file selected.")
            return

        try:
            result = load_transform_matrices_from_file(path)
            self.transform_matrices = result["matrices"]
            QtWidgets.QMessageBox.information(self, "Matrices Loaded", "Successfully loaded matrices.")
            self._update_gaze_line()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load Failed", str(e))
            print(f"[GazeTracking] Failed to load matrices: {e}")

    def reset_view(self):
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        self.plotter.render()

    def _on_point_picked(self, picked_point, picker):
        if picked_point is not None:
            coord_str = f"Picked Point: ({picked_point[0]:.2f}, {picked_point[1]:.2f}, {picked_point[2]:.2f})"
            QtWidgets.QMessageBox.information(self, "Point Coordinates", coord_str)
            print(coord_str)

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
        # Remove previous fix view actors if any
        if hasattr(self, "fix_view_actors"):
            for actor in self.fix_view_actors:
                self.plotter.remove_actor(actor)
        self.fix_view_actors = []

        # If checkbox is checked, add invisible bounds and origin
        if self.fix_view_checkbox.isChecked():
            invisible_bounds = np.array([[-100, -100, -100], [600, 600, 600]], dtype=np.float32)
            origin = np.array([[0, 0, 0]], dtype=np.float32)

            bounds_actor = self.plotter.add_points(
                invisible_bounds,
                color="white",
                opacity=0.0,
                point_size=1,
                render_points_as_spheres=True
            )
            origin_actor = self.plotter.add_points(
                origin,
                color="black",
                point_size=10,
                render_points_as_spheres=True
            )

            self.fix_view_actors.extend([bounds_actor, origin_actor])

        # Trigger re-render
        self.plotter.render()

    def refresh_transform_and_redraw(self):
        if self.current_gaze_data:
            self.update_gaze_visual(self.current_gaze_data)

    def _save_recording(self):
        self.recorder.stop()
        self.record_timer.stop()
        self.record_indicator.hide()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Recording", "", "JSON Files (*.json)")
        if path:
            self.recorder.save(path)

    def _start_recording(self):
        self.recorder.start()
        self.record_indicator.show()
        self.record_timer.start()

    def _log_current_frame(self):
        if not self.recorder.recording:
            return
        if self.latest_gaze_line is None:
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


class PlaybackWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Playback Mode")
        self.setMinimumSize(800, 600)

        # === Core Data and Playback State ===
        self.frames = []
        self.fixation_points = []
        self.current_index = 0
        self.is_playing = False

        # === PyVista Plotter ===
        self.plotter = QtInteractor(self)
        self.gaze_line_mesh = pv.Line([0, 0, 0], [0, 0, 0])
        self.gaze_line_actor = self.plotter.add_mesh(
            self.gaze_line_mesh, color="green", line_width=3
        )

        self.disc_mesh = pv.Disc(center=[0, 0, 0], inner=0.0, outer=DISC_RADIUS)
        self.disc_actor = self.plotter.add_mesh(
            self.disc_mesh, color="yellow", opacity=0.5
        )

        self.cone_mesh = pv.Cone(direction=[0, 0, 1], height=GAZE_LINE_LENGTH,
                                 radius=DISC_RADIUS)
        self.cone_actor = self.plotter.add_mesh(
            self.cone_mesh, color="red", opacity=0.3
        )

        self.peg_mesh = pv.PolyData(np.zeros((6, 3)))
        self.peg_actor = self.plotter.add_mesh(
            self.peg_mesh,
            color="purple",
            point_size=12,
            render_points_as_spheres=True,
        )

        # === Timer for Playback ===
        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)  # 100 ms = 10 FPS
        self.timer.timeout.connect(self.advance_frame)

        self.trajectory_actors = []

        # === Viewer Controls ===
        self.reset_button = QtWidgets.QPushButton("Reset View")
        self.zoom_in_button = QtWidgets.QPushButton("+")
        self.zoom_out_button = QtWidgets.QPushButton("-")
        self.fix_view_checkbox = QtWidgets.QCheckBox("Fix View")
        self.fix_view_checkbox.setChecked(False)

        # === Analysis Buttons (stubs for now) ===
        self.heatmap_button = QtWidgets.QPushButton("Show Heatmap")
        self.heatmap_button.clicked.connect(self.generate_heatmap)

        self.trajectory_checkbox = QtWidgets.QCheckBox("Show Peg Trajectories")
        self.trajectory_checkbox.setChecked(False)

        # === Scrubber (Frame Slider) ===
        self.scrubber = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scrubber.setMinimum(0)
        self.scrubber.valueChanged.connect(self.update_frame)

        # === Playback Buttons ===
        self.play_button = QtWidgets.QPushButton("▶ Play")
        self.pause_button = QtWidgets.QPushButton("⏸ Pause")
        self.restart_button = QtWidgets.QPushButton("<- Restart")
        self.play_button.clicked.connect(self.play)
        self.pause_button.clicked.connect(self.pause)
        self.restart_button.clicked.connect(self.restart)

        # === Load Recording Button ===
        self.load_button = QtWidgets.QPushButton("Load Recording")
        self.load_button.clicked.connect(self.load_recording)

        # === Layouts ===
        # Viewer Controls Section
        controls_layout = QtWidgets.QVBoxLayout()
        controls_layout.addWidget(self.reset_button)
        controls_layout.addWidget(self.zoom_in_button)
        controls_layout.addWidget(self.zoom_out_button)
        controls_layout.addWidget(self.fix_view_checkbox)
        controls_layout.addWidget(self.heatmap_button)
        controls_layout.addWidget(self.trajectory_checkbox)
        controls_layout.addStretch()

        controls_group = QtWidgets.QGroupBox("Controls")
        controls_group.setLayout(controls_layout)

        # Right Panel with Load + Controls
        right_panel = QtWidgets.QVBoxLayout()
        right_panel.addWidget(self.load_button)
        right_panel.addWidget(controls_group)
        right_panel.addStretch()

        # Central Area (Viewer + Right Panel)
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addWidget(self.plotter.interactor, stretch=4)
        main_layout.addLayout(right_panel, stretch=1)

        # Playback Button Row
        playback_controls = QtWidgets.QHBoxLayout()
        playback_controls.addWidget(self.play_button)
        playback_controls.addWidget(self.pause_button)
        playback_controls.addWidget(self.restart_button)

        # Final Container Layout
        container_layout = QtWidgets.QVBoxLayout()
        container_layout.addWidget(self.scrubber)
        container_layout.addLayout(playback_controls)
        container_layout.addLayout(main_layout)

        self.setLayout(container_layout)

        # === Final Setup ===
        self.trajectory_checkbox.stateChanged.connect(lambda _: self.update_frame(self.current_index))
        self.reset_button.clicked.connect(self.reset_view)
        self.zoom_in_button.clicked.connect(lambda: self._zoom(1.2))
        self.zoom_out_button.clicked.connect(lambda: self._zoom(0.8))
        self.fix_view_checkbox.stateChanged.connect(self.fix_view_bounds)

        self.plotter.show_axes()
        self.plotter.show_grid()

    def load_recording(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Recording", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "r") as f:
                self.frames = json.load(f)
            self.scrubber.setMaximum(len(self.frames) - 1)
            self.current_index = 0
            self.update_frame()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load Error", f"Failed to load recording: {e}")

    def update_frame(self, idx=None):
        if not self.frames:
            return
        if idx is None:
            idx = self.current_index
        else:
            self.current_index = idx

        frame = self.frames[idx]
        # Track fixation points only if intercept is true
        if frame["gaze_data"]["intercept"] == 1:
            end_point = np.array(frame["gaze_data"]["gaze_line"][1])  # Point B
            self.fixation_points.append(end_point)

        gaze_line = np.array(frame["gaze_data"]["gaze_line"])
        roi = frame["gaze_data"]["roi"]
        intercept = frame["gaze_data"]["intercept"]
        peg_data = np.array(frame["peg_data"])

        A, B = gaze_line
        line = pv.Line(A, B)
        self.gaze_line_mesh.deep_copy(line)
        self.gaze_line_mesh.Modified()

        direction = B - A
        norm_direction = direction / np.linalg.norm(direction)

        disc = pv.Disc(center=B, inner=0.0, outer=DISC_RADIUS, normal=norm_direction)
        self.disc_mesh.deep_copy(disc)
        self.disc_mesh.Modified()

        center = A + 0.5 * direction
        cone = pv.Cone(center=center, direction=-norm_direction, height=GAZE_LINE_LENGTH, radius=DISC_RADIUS)
        self.cone_mesh.deep_copy(cone)
        self.cone_mesh.Modified()
        col = Color("green" if intercept else "red").float_rgb
        self.cone_actor.GetProperty().SetColor(col)

        if peg_data.size >= 3:
            pegs = peg_data.reshape(-1, 3)
            self.peg_mesh.points = pegs
            self.peg_mesh.Modified()

        # === Draw peg trajectories if checkbox is checked ===
        for actor in self.trajectory_actors:
            self.plotter.remove_actor(actor)
        self.trajectory_actors = []
        if hasattr(self, "trajectory_checkbox") and self.trajectory_checkbox.isChecked():
            peg_count = len(self.frames[0]["peg_data"])
            peg_trails = [[] for _ in range(peg_count)]

            for fr in self.frames:
                for i, peg in enumerate(fr["peg_data"]):
                    peg_trails[i].append(peg)

            for trail in peg_trails:
                trail = np.array(trail)
                if len(trail) < 2:
                    continue
                line = pv.Spline(trail, len(trail) * 10)
                actor = self.plotter.add_mesh(line, color="blue", line_width=3)
                self.trajectory_actors.append(actor)

        self.plotter.render()

    def generate_heatmap(self):
        if not self.fixation_points and self.frames:
            for fr in self.frames:
                if fr["gaze_data"]["intercept"] == 1:
                    end_point = np.array(fr["gaze_data"]["gaze_line"][1])
                    self.fixation_points.append(end_point)

        if not self.fixation_points:
            QtWidgets.QMessageBox.information(self, "No Data", "No fixation points recorded.")
            return

        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde
        import os

        points = np.array(self.fixation_points)
        x = points[:, 0]
        y = points[:, 1]  # Or use z instead of y depending on view plane

        # Kernel density estimate for smooth heatmap
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy)
        xi, yi = np.mgrid[x.min():x.max():500j, y.min():y.max():500j]
        zi = kde(np.vstack([xi.flatten(), yi.flatten()]))

        plt.figure(figsize=(8, 6))
        plt.title("Gaze Fixation Heatmap")
        plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading="auto", cmap="hot")
        plt.colorbar(label="Fixation Density")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.tight_layout()

        save_path = "fixation_heatmap.png"
        plt.savefig(save_path)
        plt.show()

        print(f"[Heatmap] Heatmap saved to {os.path.abspath(save_path)}")

    def reset_view(self):
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        self.plotter.render()

    def _zoom(self, factor: float):
        if hasattr(self.plotter.camera, "Zoom"):
            self.plotter.camera.Zoom(factor)
        self.plotter.render()

    def fix_view_bounds(self):
        # Remove any previously added actors
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
                bounds,
                color="white",
                opacity=0.0,
                point_size=1,
                render_points_as_spheres=True
            )
            origin_actor = self.plotter.add_points(
                origin,
                color="black",
                point_size=10,
                render_points_as_spheres=True
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


class MainWindow(QtWidgets.QWidget):
    """Main application window."""
    trigger_matrix_mode = QtCore.pyqtSignal()
    
    def __init__(self, enable_receiver=True):
        super().__init__()
        self.setWindowTitle("Real-Time Point Viewer")
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
            callback=self._on_point_picked,
            use_picker=True,
            show_message=False,
            left_clicking=True,
            show_point=True
        )

        self.render_timer = QtCore.QTimer()
        self.render_timer.setInterval(100)  # Adjust to ~10 FPS (100 ms)
        self.render_timer.timeout.connect(self.plotter.render)
        self.render_timer.start()

        # Dashed line setup (20 segments = 10 dashes)
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

        self.cam_checkbox = QtWidgets.QCheckBox("Show Camera Points")
        self.cam_checkbox.setChecked(True)
        self.holo_checkbox = QtWidgets.QCheckBox("Show HoloLens Points")
        self.holo_checkbox.setChecked(True)
        self.reset_button = QtWidgets.QPushButton("Reset View")
        self.zoom_in_button = QtWidgets.QPushButton("+")
        self.fix_view_checkbox = QtWidgets.QCheckBox("Fix View")
        self.fix_view_checkbox.setChecked(False)
        self.zoom_out_button = QtWidgets.QPushButton("-")

        self.gaze_distance_label = QtWidgets.QLabel("Gaze Distance: N/A")

        self.latest_validation_gaze = None
        self.latest_validation_roi = 0.0
        self.latest_validation_intercept = 0

        # Right side configuration panel
        options_group = QtWidgets.QGroupBox("Options")
        options_layout = QtWidgets.QVBoxLayout()
        options_layout.addWidget(self.cam_checkbox)
        options_layout.addWidget(self.holo_checkbox)
        options_group.setLayout(options_layout)

        # Transform configuration
        transform_group = QtWidgets.QGroupBox("Transform")
        transform_layout = QtWidgets.QVBoxLayout()
        self.transform_checkbox = QtWidgets.QCheckBox("Show Transform Point")
        self.transform_checkbox.setChecked(True)
        fields_layout = QtWidgets.QHBoxLayout()
        self.transform_x = QtWidgets.QLineEdit()
        self.transform_x.setPlaceholderText("X")
        self.transform_x.setMaximumWidth(50)
        self.transform_y = QtWidgets.QLineEdit()
        self.transform_y.setPlaceholderText("Y")
        self.transform_y.setMaximumWidth(50)
        self.transform_z = QtWidgets.QLineEdit()
        self.transform_z.setPlaceholderText("Z")
        self.transform_z.setMaximumWidth(50)
        fields_layout.addWidget(self.transform_x)
        fields_layout.addWidget(self.transform_y)
        fields_layout.addWidget(self.transform_z)
        self.transform_apply = QtWidgets.QPushButton("Create Point")
        transform_layout.addWidget(self.transform_checkbox)
        self.clear_transform_button = QtWidgets.QPushButton("Clear Transform Points")
        transform_layout.addWidget(self.clear_transform_button)
        self.load_matrices_button = QtWidgets.QPushButton("Load Matrices")
        transform_layout.addWidget(self.load_matrices_button)
        self.load_matrices_button.clicked.connect(self.load_matrices_from_file)

        self.matrix_buttons = []
        self.matrix_group = QtWidgets.QButtonGroup()
        for i in range(4):
            btn = QtWidgets.QRadioButton(f"{MATRIX_BUTTON_LABELS[i]}")
            self.matrix_buttons.append(btn)
            self.matrix_group.addButton(btn, i)
            transform_layout.addWidget(btn)
        self.matrix_buttons[0].setChecked(True)

        self.matrix_apply_button = QtWidgets.QPushButton("Transform")
        transform_layout.addWidget(self.matrix_apply_button)

        transform_layout.addLayout(fields_layout)
        transform_layout.addWidget(self.transform_apply)
        transform_group.setLayout(transform_layout)

        view_group = QtWidgets.QGroupBox("View")
        view_layout = QtWidgets.QHBoxLayout()
        view_layout.addWidget(self.reset_button)
        view_layout.addWidget(self.zoom_in_button)
        view_layout.addWidget(self.zoom_out_button)
        view_layout.addWidget(self.fix_view_checkbox)
        view_group.setLayout(view_layout)

        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(options_group)
        right_layout.addWidget(transform_group)
        right_layout.addWidget(view_group)
        right_layout.addStretch()
        right_layout.addWidget(self.gaze_distance_label)

        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.plotter.interactor, 1)
        layout.addLayout(right_layout)

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
        """Store the latest peg validation point for rendering via timer."""
        self.peg_validation_point = point
        # Mesh updates are now handled in the _update_validation_gaze_line method via timer.

    @QtCore.pyqtSlot(object, float, int, float)
    def update_validation_gaze(self, gaze_line, roi, intercept, gaze_distance):
        """Receive latest validation gaze data from the socket."""

        try:
            # Store latest data for timer based update
            gaze_arr = np.array(gaze_line)
            if gaze_arr.shape != (2, 3):
                print(f"[MainWindow] Invalid validation gaze shape: {gaze_arr.shape}")
                return
            origin = gaze_arr[0]
            direction = gaze_arr[1]
            length = np.linalg.norm(direction)
            if length == 0:
                target = origin
            else:
                # Determine distance of transformed validation peg from origin
                gaze_length = GAZE_LINE_LENGTH  # fallback
                if self.peg_validation_point is not None:
                    idx = self.matrix_group.checkedId()
                    if 0 <= idx < len(self.transform_matrices):
                        pt_h = np.append(self.peg_validation_point, 1.0)
                        transformed = (self.transform_matrices[idx] @ pt_h)[:3]
                        gaze_length = np.linalg.norm(transformed)

                # Use dynamic length for target
                target = origin + (direction / length) * gaze_length

            self.latest_validation_gaze = np.array([origin, target])
            self.latest_validation_roi = float(roi)
            self.latest_validation_intercept = int(intercept)

            self.gaze_distance_label.setText(f"Gaze Distance: {gaze_distance:.2f} mm")

            self._update_validation_gaze_line()
        except Exception as e:
            print(f"[MainWindow] Failed to update validation gaze visuals: {e}")

    def _update_validation_gaze_line(self):
        if self.latest_validation_gaze is None:
            return

        A, B = self.latest_validation_gaze
        cone_center = A + 0.5 * (B - A)
        roi = float(self.latest_validation_roi)
        direction = B - A
        length = np.linalg.norm(direction)
        if length == 0:
            return
        norm_direction = direction / length

        # === 1. Create or update gaze line ===
        line = pv.Line(A, B)
        if hasattr(self, "validation_line_mesh"):
            self.validation_line_mesh.deep_copy(line)
            self.validation_line_mesh.Modified()
        else:
            self.validation_line_mesh = line
            self.validation_gaze_line_actor = self.plotter.add_mesh(
                self.validation_line_mesh,
                color="green",  # Will be updated dynamically
                line_width=3
            )

        # === 2. Disc at endpoint ===
        if hasattr(self, "validation_disc_mesh"):
            disc = pv.Disc(center=B, inner=0.0, outer=DISC_RADIUS, normal=norm_direction, r_res=1, c_res=10)
            self.validation_disc_mesh.deep_copy(disc)
            self.validation_disc_mesh.Modified()
        else:
            self.validation_disc_mesh = pv.Disc(center=B, inner=0.0, outer=DISC_RADIUS, normal=norm_direction, r_res=1, c_res=10)
            self.validation_disc_actor = self.plotter.add_mesh(self.validation_disc_mesh, color="cyan", opacity=0.5)

        # === 3. Cone from origin to disc ===
        cone_color = "green" if self.latest_validation_intercept else "red"

        cone = pv.Cone(center=cone_center, direction=-norm_direction, height=GAZE_LINE_LENGTH, radius=DISC_RADIUS)
        if hasattr(self, "validation_cone_mesh"):
            self.validation_cone_mesh.deep_copy(cone)
            self.validation_cone_mesh.Modified()
            # Update actor color (reuses actor)
            if hasattr(self, "validation_cone_actor"):
                self.validation_cone_actor.GetProperty().SetColor(Color(cone_color).float_rgb)
        else:
            self.validation_cone_mesh = cone
            self.validation_cone_actor = self.plotter.add_mesh(
                self.validation_cone_mesh,
                color=cone_color,
                opacity=0.3
            )

        # --- 4. Validation Peg (real and transformed) with ROI sphere ---
        if self.peg_validation_point is not None:
            point = self.peg_validation_point

            # Real peg point (purple)
            if not hasattr(self, "peg_validation_mesh"):
                self.peg_validation_mesh = pv.PolyData(point.reshape(1, 3))
                self.peg_validation_actor = self.plotter.add_mesh(
                    self.peg_validation_mesh,
                    color="purple",
                    point_size=12,
                    render_points_as_spheres=True
                )
            else:
                self.peg_validation_mesh.points = point.reshape(1, 3)
                self.peg_validation_mesh.Modified()

            # Transformed peg point (dark purple) and transparent ROI sphere
            idx = self.matrix_group.checkedId()
            if 0 <= idx < len(self.transform_matrices):
                matrix = self.transform_matrices[idx]
                pt_h = np.append(point, 1.0)
                transformed = (matrix @ pt_h)[:3]

                if not hasattr(self, "peg_transformed_mesh"):
                    self.peg_transformed_mesh = pv.PolyData(transformed.reshape(1, 3))
                    self.peg_transformed_actor = self.plotter.add_mesh(
                        self.peg_transformed_mesh,
                        color="#300053",  # dark purple
                        point_size=12,
                        render_points_as_spheres=True
                    )
                else:
                    self.peg_transformed_mesh.points = transformed.reshape(1, 3)
                    self.peg_transformed_mesh.Modified()

                # Transparent ROI sphere at transformed peg
                if hasattr(self, "validation_sphere_mesh"):
                    sphere = pv.Sphere(radius=roi, center=transformed)
                    self.validation_sphere_mesh.deep_copy(sphere)
                    self.validation_sphere_mesh.Modified()
                else:
                    self.validation_sphere_mesh = pv.Sphere(radius=roi, center=transformed)
                    self.validation_sphere_actor = self.plotter.add_mesh(
                        self.validation_sphere_mesh, color="green", opacity=0.2
                    )

        # === 5. Animated dashed line from closest peg to gaze line ===
        if self.peg_validation_point is not None and self.latest_validation_gaze is not None:
            peg = self.peg_validation_point
            A, B = self.latest_validation_gaze

            def closest_point_on_line(p, a, b):
                ab = b - a
                t = np.dot(p - a, ab) / np.dot(ab, ab)
                t = np.clip(t, 0, 1)
                return a + t * ab

            closest_point = closest_point_on_line(peg, A, B)
            dashed_pairs = np.linspace(closest_point, peg, self.dashed_segments * 2).reshape(-1, 2, 3)

            for i in range(self.dashed_segments):
                if i < len(dashed_pairs) and i % 2 == 0:
                    start, end = dashed_pairs[i]
                    mesh = self.validation_dashed_meshes[i // 2]
                    mesh.points = np.array([start, end])
                    mesh.lines = np.array([2, 0, 1])
                    mesh.Modified()
                else:
                    # Collapse unused segments
                    self.validation_dashed_meshes[i // 2].points = np.array([[0, 0, 0], [0, 0, 0]])
                    self.validation_dashed_meshes[i // 2].Modified()

    @QtCore.pyqtSlot(object, float, int, float, object)
    def receive_gaze_data(self, gaze_line, roi, intercept, gaze_distance, pegs):
        if hasattr(self, "gaze_window") and self.gaze_window:
            self.gaze_window.update_gaze_data(gaze_line, roi, intercept, gaze_distance, pegs)

    def reset_view(self):
        """Reset camera orientation with Z axis pointing up."""
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        self.plotter.render()

    def _on_point_picked(self, picked_point, picker):
        if picked_point is not None:
            coord_str = f"Picked Point: ({picked_point[0]:.2f}, {picked_point[1]:.2f}, {picked_point[2]:.2f})"
            QtWidgets.QMessageBox.information(self, "Point Coordinates", coord_str)
            print(coord_str)

    def zoom_in(self):
        """Zoom the view in."""
        self._zoom(1.2)

    def zoom_out(self):
        """Zoom the view out."""
        self._zoom(0.8)

    def handle_matrix_mode(self):
        matrix_path = "Assets/Scripts/GUI/transform_data.txt"
        try:
            result = load_transform_matrices_from_file(matrix_path)
            self.transform_matrices = result["matrices"]

            if result["camera_points"] and result["hololens_points"]:
                self.set_all_pairs(result["camera_points"], result["hololens_points"])

            self.matrix_info_window = MatrixInfoWindow(
                self.transform_matrices,
                result["rows"],
                result["rmse"],
                result["repro"]
            )
            self.matrix_info_window.show()
            print("[MainWindow] Transform matrices loaded successfully.")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Matrix Load Failed", str(e))
            print(f"[MainWindow] Error loading matrices: {e}")

    def apply_matrix_transform(self):
        idx = self.matrix_group.checkedId()
        if idx < 0 or idx >= len(self.transform_matrices):
            print("[MainWindow] Invalid matrix index selected.")
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
        """Read transform point from fields and store it."""
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
        path = "Assets/Scripts/GUI/transform_data.txt"

        if not path:
            QtWidgets.QMessageBox.warning(self, "Load Failed", "No file selected.")
            return

        try:
            result = load_transform_matrices_from_file(path)
            self.transform_matrices = result["matrices"]

            if result["camera_points"] and result["hololens_points"]:
                self.set_all_pairs(result["camera_points"], result["hololens_points"])

            self.matrix_info_window = MatrixInfoWindow(
                self.transform_matrices,
                result["rows"],
                result["rmse"],
                result["repro"]
            )
            self.matrix_info_window.show()

            QtWidgets.QMessageBox.information(self, "Matrices Loaded", "Successfully loaded matrices.")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load Failed", str(e))
            print(f"[MainWindow] Failed to load matrices: {e}")

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
                np.vstack(self.camera_points),
                color="red",
                point_size=12,
                opacity=1.0 if self.cam_checkbox.isChecked() else 0.0,
                render_points_as_spheres=True,
                pickable=True
            )

        if self.holo_points:
            self.plotter.add_points(
                np.vstack(self.holo_points),
                color="blue",
                point_size=12,
                opacity=1.0 if self.holo_checkbox.isChecked() else 0.0,
                render_points_as_spheres=True,
                pickable=True
            )

        if self.transform_checkbox.isChecked():
            if self.matrix_transform_points:
                self.plotter.add_points(
                    np.vstack(self.matrix_transform_points),
                    color="green",  # transformed camera points
                    point_size=12,
                    render_points_as_spheres=True,
                    pickable=True
                )
            if self.manual_points:
                self.plotter.add_points(
                    np.vstack(self.manual_points),
                    color="#CBCB00",  # manual point
                    point_size=12,
                    render_points_as_spheres=True,
                    pickable=True
                )
            if self.manual_transformed_points:
                self.plotter.add_points(
                    np.vstack(self.manual_transformed_points),
                    color="#C48300",  # dark yellow for manual transform
                    point_size=12,
                    render_points_as_spheres=True,
                    pickable=True
                )

        if (
            self.camera_points
            and self.holo_points
            and self.cam_checkbox.isChecked()
            and self.holo_checkbox.isChecked()
        ):
            for cam, holo in zip(self.camera_points, self.holo_points):
                self.draw_dashed_line(cam, holo)

        # Fix View Logic: Add bounding and origin points
        for actor in self.fix_view_actors:
            self.plotter.remove_actor(actor)
        self.fix_view_actors.clear()

        if self.fix_view_checkbox.isChecked():
            invisible_bounds = np.array([[-100, -100, -100], [600, 600, 600]], dtype=np.float32)
            origin = np.array([[0, 0, 0]], dtype=np.float32)

            # Invisible bounds (tiny and transparent)
            bounds_actor = self.plotter.add_points(
                invisible_bounds,
                color="white",
                opacity=0.0,
                point_size=1,
                render_points_as_spheres=True
            )
            # Visible origin
            origin_actor = self.plotter.add_points(
                origin,
                color="black",
                point_size=10,
                render_points_as_spheres=True
            )

            self.fix_view_actors.extend([bounds_actor, origin_actor])

        self.plotter.render()


class MainMenuWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mixed Reality Surgical Training System")
        self.setMinimumSize(400, 300)

        layout = QtWidgets.QVBoxLayout()

        # Info Label
        info_label = QtWidgets.QLabel(
            "<h2>Welcome to the Surgical Training GUI</h2>"
            "<p>Select a mode to begin:</p>"
        )
        info_label.setAlignment(QtCore.Qt.AlignCenter)

        # Buttons
        calibration_btn = QtWidgets.QPushButton("Calibration / Validation Mode")
        gaze_tracking_btn = QtWidgets.QPushButton("Gaze Tracking Mode")
        playback_btn = QtWidgets.QPushButton("Playback Mode")

        calibration_btn.clicked.connect(self.launch_calibration)
        gaze_tracking_btn.clicked.connect(self.launch_gaze_tracking)
        playback_btn.clicked.connect(self.launch_playback)

        layout.addWidget(info_label)
        layout.addStretch()
        layout.addWidget(calibration_btn)
        layout.addWidget(gaze_tracking_btn)
        layout.addWidget(playback_btn)
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


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_menu = MainMenuWindow()
    main_menu.resize(500, 400)
    main_menu.show()
    sys.exit(app.exec_())