import sys
import socket
import json
from typing import List
import numpy as np

import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets
import pyvista as pv
pv.global_theme.allow_empty_mesh = True
from pyvistaqt import QtInteractor

MATRIX_BUTTON_LABELS = [
            "Similarity Xform",
            "Affine Xform",
            "Similarity Xform RANSAC",
            "Affine Xform RANSAC"
        ]

class DataReceiver(QtCore.QThread):
    updated_points = QtCore.pyqtSignal(list, list)
    peg_point_received = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, host: str = "0.0.0.0", port: int = 9991, parent=None):
        super().__init__(parent)
        self.host = host
        self.port = port
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            server_socket.bind((self.host, self.port))
        except Exception as e:
            print(f"[Receiver] Failed to start server: {e}")

        while self._running:
            try:
                data, addr = server_socket.recvfrom(650000)
                if not data:
                    continue
                line = data.decode('utf-8').strip()
                if line.startswith("M"):
                    QtCore.QMetaObject.invokeMethod(
                        self.parent(), "trigger_matrix_mode", QtCore.Qt.QueuedConnection
                    )
                elif line.startswith("V"):
                    try:
                        data = json.loads(line[1:])
                        x, y, z = float(data["x"]), float(data["y"]), float(data["z"])
                        point = np.array([x, y, z])
                        self.peg_point_received.emit(point)

                        # Additional gaze-related fields
                        gaze_line = np.array(data.get("gaze_line"))
                        roi_radius = float(data.get("roi", 0.0))
                        intercept = int(data.get("intercept", 0))

                        # Send visual data to main window
                        QtCore.QMetaObject.invokeMethod(
                            self.parent(), "update_validation_gaze",
                            QtCore.Qt.QueuedConnection,
                            QtCore.Q_ARG(object, gaze_line),
                            QtCore.Q_ARG(float, roi_radius),
                            QtCore.Q_ARG(int, intercept)
                        )
                    except Exception as e:
                        print(f"[Receiver] Error parsing or handling V message: {e}")
                elif line.startswith("G"):
                    try:
                        gaze_data = json.loads(line[1:])
                        QtCore.QMetaObject.invokeMethod(
                            self.parent(), "receive_gaze_data", QtCore.Qt.QueuedConnection,
                            QtCore.Q_ARG(dict, gaze_data)
                        )
                    except Exception as e:
                        print(f"[Receiver] Error parsing gaze data: {e}")
                else:
                    data = json.loads(line)
                    camera_points = []
                    holo_points = []
                    for key, value in data.items():
                        cam = np.array(value[0])
                        holo = np.array(value[1])
                        camera_points.append(cam)
                        holo_points.append(holo)
                    self.updated_points.emit(camera_points, holo_points)
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"[Receiver] Error receiving data: {e}")
                break


class MatrixInfoWindow(QtWidgets.QWidget):
    def __init__(self, matrix_data, point_rows, rmse_values, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Transformation Matrix Info")
        self.setMinimumWidth(1000)
        layout = QtWidgets.QVBoxLayout()

        # Table setup
        table = QtWidgets.QTableWidget()
        table.setColumnCount(10)
        table.setHorizontalHeaderLabels([
            "Camera", "HoloLens",
            "Sim Xform", "Sim Error",
            "Affine Xform", "Affine Error",
            "Sim RANSAC", "Sim RANSAC Error",
            "Affine RANSAC", "Affine RANSAC Error"
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
        for name, mat, rmse in zip(
            ["Similarity", "Affine", "Similarity RANSAC", "Affine RANSAC"],
            matrix_data,
            rmse_values
        ):
            mat_str = "\n".join("  ".join(f"{v:.4f}" for v in row) for row in mat)
            matrix_label = QtWidgets.QLabel(f"<pre>{mat_str}</pre>")
            matrix_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            matrix_layout.addRow(f"{name} Matrix (RMSE: {rmse:.4f})", matrix_label)

        layout.addLayout(matrix_layout)
        self.setLayout(layout)


class GazeTrackingWindow(QtWidgets.QWidget):
    def __init__(self, transform_matrices=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gaze Tracking")
        self.setMinimumSize(600, 400)
        self.plotter = QtInteractor(self)

        self.latest_gaze_line = None
        self.gaze_line_actor = None
        self.line_mesh = pv.PolyData()
        self.line_actor = self.plotter.add_mesh(self.line_mesh, color="green", line_width=3, render=False)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_gaze_line_actor)

        self.current_gaze_data = None
        self.gaze_history = []
        self.max_history = 50
        self.selected_matrix_index = 0
        self.transform_matrices = transform_matrices or []

        viewer_layout = QtWidgets.QVBoxLayout()
        viewer_layout.addWidget(self.plotter.interactor)

        view_group = QtWidgets.QGroupBox("View")
        view_layout = QtWidgets.QVBoxLayout()

        self.reset_button = QtWidgets.QPushButton("Reset View")
        self.zoom_in_button = QtWidgets.QPushButton("+")
        self.zoom_out_button = QtWidgets.QPushButton("-")

        self.reset_button.clicked.connect(self.reset_view)
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_out_button.clicked.connect(self.zoom_out)

        view_layout.addWidget(self.reset_button)
        view_layout.addWidget(self.zoom_in_button)
        view_layout.addWidget(self.zoom_out_button)
        view_layout.addStretch()

        view_group.setLayout(view_layout)

        transform_group = QtWidgets.QGroupBox("Transform")
        transform_layout = QtWidgets.QVBoxLayout()

        self.matrix_group = QtWidgets.QButtonGroup()
        self.matrix_buttons = []

        for i in range(4):
            btn = QtWidgets.QRadioButton(f"{MATRIX_BUTTON_LABELS[i]}")
            self.matrix_group.addButton(btn, i)
            self.matrix_buttons.append(btn)
            transform_layout.addWidget(btn)

        self.matrix_buttons[0].setChecked(True)
        transform_group.setLayout(transform_layout)

        self.matrix_group.buttonClicked.connect(self.refresh_transform_and_redraw)

        right_panel = QtWidgets.QVBoxLayout()
        right_panel.addWidget(transform_group)
        right_panel.addWidget(view_group)
        right_panel.addStretch()

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(viewer_layout, stretch=4)
        main_layout.addLayout(right_panel, stretch=1)

        self.setLayout(main_layout)

        self.plotter.show_axes()
        self.plotter.show_grid()

    def update_gaze_visual(self, gaze_data: dict):
        try:
            self.current_gaze_data = gaze_data
            gaze_line = np.array(gaze_data["gaze_line"])
            if gaze_line.shape != (2, 3):
                print(f"[Gaze] Invalid gaze line shape: {gaze_line.shape}")
                return

            A, B = gaze_line[0], gaze_line[1]
            direction = B - A
            norm_direction = direction / np.linalg.norm(direction)

            # Update or create gaze line
            if hasattr(self, "line_mesh") and self.line_mesh is not None:
                self.line_mesh.points = pv.pyvista_ndarray(gaze_line)
                self.line_mesh.lines = np.array([2, 0, 1])
                self.line_mesh.Modified()
            else:
                self.line_mesh = pv.PolyData()
                self.line_mesh.points = pv.pyvista_ndarray(gaze_line)
                self.line_mesh.lines = np.array([2, 0, 1])
                self.gaze_line_actor = self.plotter.add_mesh(self.line_mesh, color="green", line_width=3)

            # Remove old disc/cone actors if they exist
            if hasattr(self, "roi_actor") and self.roi_actor:
                self.plotter.remove_actor(self.roi_actor)
            if hasattr(self, "cone_actor") and self.cone_actor:
                self.plotter.remove_actor(self.cone_actor)

            # ROI disc + cone
            if "roi" in gaze_data and gaze_data["roi"] is not None:
                roi_center, roi_radius = gaze_data["roi"]
                roi_radius = float(roi_radius)
                disc_center = A + 0.5 * direction
                disc = pv.Disc(center=disc_center, inner=0, outer=roi_radius, normal=norm_direction, r_res=1, c_res=100)
                self.roi_actor = self.plotter.add_mesh(disc, color="yellow", opacity=0.5)

                cone_length = float(np.linalg.norm(disc_center - A))
                cone = pv.Cone(center=A, direction=norm_direction, height=cone_length, radius=roi_radius)
                self.cone_actor = self.plotter.add_mesh(cone, color="orange", opacity=0.3)

            # Pegs
            if "pegs" in gaze_data:
                for actor in getattr(self, "peg_actors", []):
                    self.plotter.remove_actor(actor)
                self.peg_actors = []

                original_pegs = np.array(gaze_data["pegs"])
                idx = self.matrix_group.checkedId()
                if 0 <= idx < len(self.transform_matrices):
                    matrix = self.transform_matrices[idx]
                    pegs = [matrix @ np.append(p, 1.0) for p in original_pegs]
                    pegs = np.array([p[:3] for p in pegs])
                else:
                    pegs = original_pegs

                def point_line_distance(p, a, b):
                    ap = p - a
                    ab = b - a
                    t = max(0, min(1, np.dot(ap, ab) / np.dot(ab, ab)))
                    closest = a + t * ab
                    return np.linalg.norm(p - closest), closest

                closest_peg, closest_point, min_dist = None, None, float("inf")
                for peg in pegs:
                    dist, proj = point_line_distance(peg, A, B)
                    if dist < min_dist:
                        min_dist = dist
                        closest_peg = peg
                        closest_point = proj

                for peg in pegs:
                    is_closest = closest_peg is not None and np.allclose(peg, closest_peg)
                    color = "red" if is_closest else "blue"
                    actor = self.plotter.add_points(np.array([peg]), color=color, point_size=12, render_points_as_spheres=True)
                    self.peg_actors.append(actor)

                if closest_point is not None and closest_peg is not None:
                    segments = 20
                    dashed_points = np.linspace(closest_point, closest_peg, segments * 2).reshape(-1, 2, 3)
                    for i, (start, end) in enumerate(dashed_points):
                        if i % 2 == 0:
                            self.plotter.add_lines(np.array([start, end]), color="red", width=2)

            self.plotter.render()
            QtWidgets.QApplication.processEvents()

        except Exception as e:
            print(f"[GazeTracking] Failed to update: {e}")

    def _update_gaze_line_actor(self):
        if self.latest_gaze_line is None or len(self.latest_gaze_line) != 2:
            return

        points = np.array(self.latest_gaze_line)
        if points.shape != (2, 3):
            return

        # Update line geometry
        self.line_mesh.points = pv.pyvista_ndarray(points)
        self.line_mesh.lines = np.array([2, 0, 1])  # VTK line: n_points, i0, i1
        self.line_mesh.Modified()
        self.plotter.render()


    def reset_view(self):
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        self.plotter.render()

    def zoom_in(self):
        self._zoom(1.2)

    def zoom_out(self):
        self._zoom(0.8)

    def _zoom(self, factor: float):
        camera = self.plotter.camera
        if hasattr(camera, "Zoom"):
            camera.Zoom(factor)
        self.plotter.render()

    def refresh_transform_and_redraw(self):
        if self.current_gaze_data:
            self.update_gaze_visual(self.current_gaze_data)


class MainWindow(QtWidgets.QWidget):
    """Main application window."""
    trigger_matrix_mode = QtCore.pyqtSignal()
    
    def __init__(self):
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

        # UI setup
        self.plotter = QtInteractor(self)
        self.plotter.show_axes()
        self.plotter.show_grid()
        self.cam_checkbox = QtWidgets.QCheckBox("Show Camera Points")
        self.cam_checkbox.setChecked(True)
        self.holo_checkbox = QtWidgets.QCheckBox("Show HoloLens Points")
        self.holo_checkbox.setChecked(True)
        self.reset_button = QtWidgets.QPushButton("Reset View")
        self.zoom_in_button = QtWidgets.QPushButton("+")
        self.zoom_out_button = QtWidgets.QPushButton("-")
        self.rmse_label = QtWidgets.QLabel("RMSE: N/A")
        self.validation_line_mesh = pv.PolyData()
        self.validation_gaze_line_actor = self.plotter.add_mesh(self.validation_line_mesh, color="green", line_width=3, render=False)

        self.validation_gaze_timer = QtCore.QTimer()
        self.validation_gaze_timer.timeout.connect(self._update_validation_gaze_line)
        self.validation_gaze_timer.start(100)  # update every 100 ms

        # Right side configuration panel
        options_group = QtWidgets.QGroupBox("Options")
        options_layout = QtWidgets.QVBoxLayout()
        options_layout.addWidget(self.cam_checkbox)
        options_layout.addWidget(self.holo_checkbox)
        options_group.setLayout(options_layout)
        gaze_group = QtWidgets.QGroupBox("Gaze Tracking")
        gaze_layout = QtWidgets.QVBoxLayout()

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
        view_group.setLayout(view_layout)

        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(options_group)
        right_layout.addWidget(transform_group)
        right_layout.addWidget(view_group)
        right_layout.addStretch()
        self.launch_gaze_button = QtWidgets.QPushButton("Launch Gaze Tracking")
        gaze_layout.addWidget(self.launch_gaze_button)
        gaze_group.setLayout(gaze_layout)
        right_layout.addWidget(gaze_group)
        right_layout.addWidget(self.rmse_label)

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
        self.launch_gaze_button.clicked.connect(self.open_gaze_tracking)

        self.receiver = DataReceiver(parent=self)
        self.receiver.updated_points.connect(self.set_all_pairs)
        self.receiver.peg_point_received.connect(self.set_peg_validation_point)
        self.receiver.start()

    def closeEvent(self, event):
        self.receiver.stop()
        self.receiver.wait()
        super().closeEvent(event)

    @QtCore.pyqtSlot(list, list)
    def set_all_pairs(self, camera_points, holo_points):
        self.camera_points = camera_points
        self.holo_points = holo_points
        self.update_rmse()
        self.update_scene()

    @QtCore.pyqtSlot(np.ndarray)
    def set_peg_validation_point(self, point):
        """Store the latest peg validation point for rendering via timer."""
        self.peg_validation_point = point
        # Mesh updates are now handled in the _update_validation_gaze_line method via timer.

    @QtCore.pyqtSlot(object, float, int)
    def update_validation_gaze(self, gaze_line, roi_radius, intercept):
        try:
            color = "red" if intercept else "green"

            # Update gaze line in-place
            self.latest_validation_gaze = np.array(gaze_line)
            self.validation_line_mesh.lines = np.array([2, 0, 1])
            self.validation_line_mesh.Modified()
            self.plotter.update_scalars(None, render=False)
            self.validation_gaze_line_actor.prop.set_color(color)

            # Remove old disc if needed
            if hasattr(self, "validation_roi_actor") and self.validation_roi_actor:
                self.plotter.remove_actor(self.validation_roi_actor)

            # ROI disc
            A, B = gaze_line
            direction = B - A
            norm_direction = direction / np.linalg.norm(direction)
            disc_center = A + 0.5 * direction
            disc = pv.Disc(center=disc_center, inner=0, outer=roi_radius, normal=norm_direction, r_res=1, c_res=100)
            self.validation_roi_actor = self.plotter.add_mesh(disc, color="yellow", opacity=0.5)

            self.plotter.render()
        except Exception as e:
            print(f"[MainWindow] Failed to update validation gaze visuals: {e}")

    def _update_validation_gaze_line(self):
        # --- Update validation gaze line ---
        if hasattr(self, "latest_validation_gaze") and self.latest_validation_gaze is not None:
            points = self.latest_validation_gaze
            if isinstance(points, np.ndarray) and points.shape == (2, 3):
                self.validation_line_mesh.points = pv.pyvista_ndarray(points)
                self.validation_line_mesh.lines = np.array([2, 0, 1])
                self.validation_line_mesh.Modified()

        # --- Update validation peg point (real and transformed) ---
        if self.peg_validation_point is not None:
            point = self.peg_validation_point

            # Real point
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

            # Transformed point
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

        self.plotter.render()

    @QtCore.pyqtSlot(dict)
    def receive_gaze_data(self, gaze_data: dict):
        if hasattr(self, "gaze_window") and self.gaze_window:
            self.gaze_window.update_gaze_visual(gaze_data)

    def update_rmse(self):
        if not self.camera_points:
            self.rmse_label.setText("RMSE: N/A")
            return
        cams = np.vstack(self.camera_points)
        holos = np.vstack(self.holo_points)
        diff = cams - holos
        rmse = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
        self.rmse_label.setText(f"RMSE: {rmse:.4f}")

    def reset_view(self):
        """Reset camera orientation with Z axis pointing up."""
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        self.plotter.render()

    def zoom_in(self):
        """Zoom the view in."""
        self._zoom(1.2)

    def zoom_out(self):
        """Zoom the view out."""
        self._zoom(0.8)

    def handle_matrix_mode(self):
        matrix_path = "Assets/Scripts/GUI/transform_data.txt"
        try:
            with open(matrix_path, "r") as f:
                matrix_json =json.load(f)
            required_keys = [
                "similarity_transform", "affine_transform", 
                "similarity_transform_ransac", "affine_transform_ransac"
                ]
            self.transform_matrices = []
            for key in required_keys:
                if key not in matrix_json:
                    raise ValueError(f"Missing required key: {key}")
                matrix_data = matrix_json[key]
                matrix = np.array(matrix_data).reshape(4, 4)
                self.transform_matrices.append(matrix)

            rmse_values = matrix_json.get("rmse", [0, 0, 0, 0])
            row_dict = matrix_json.get("rows", {})
            num_rows = len(row_dict.get("Camera", []))
            print(f"[MainWindow] Number of rows in data: {len(row_dict)}")
            point_rows = []
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
                    row_dict["affine_transformed_ransac_B"][i],
                    row_dict["affine_transform_ransac_errors"][i],
                ]
                point_rows.append(row)

            if "Camera" in row_dict and "Hololens" in row_dict:
                try:
                    camera_points = [np.array(p) for p in row_dict["Camera"]]
                    hololens_points = [np.array(p) for p in row_dict["Hololens"]]
                    if len(camera_points) == len(hololens_points):
                        self.set_all_pairs(camera_points, hololens_points)
                        print(f"[MainWindow] Loaded {len(camera_points)} Camera/Hololens point pairs from matrix file.")
                    else:
                        print("[MainWindow] Warning: 'Camera' and 'Hololens' arrays are unequal in length.")
                except Exception as e:
                    print(f"[MainWindow] Failed to parse Camera/Hololens point arrays: {e}")

            self.matrix_info_window = MatrixInfoWindow(self.transform_matrices, point_rows, rmse_values)
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
        self.handle_matrix_mode()

    def _zoom(self, factor: float):
        camera = self.plotter.camera
        if hasattr(camera, "Zoom"):
            camera.Zoom(factor)
        self.plotter.render()

    def open_gaze_tracking(self):
        print("[MainWindow] Launching gaze tracking window...")
        self.gaze_window = GazeTrackingWindow(transform_matrices=self.transform_matrices)
        self.gaze_window.show()

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
            )

        if self.holo_points:
            self.plotter.add_points(
                np.vstack(self.holo_points),
                color="blue",
                point_size=12,
                opacity=1.0 if self.holo_checkbox.isChecked() else 0.0,
                render_points_as_spheres=True,
            )

        if self.transform_checkbox.isChecked():
            if self.matrix_transform_points:
                self.plotter.add_points(
                    np.vstack(self.matrix_transform_points),
                    color="green",  # transformed camera points
                    point_size=12,
                    render_points_as_spheres=True,
                )
            if self.manual_points:
                self.plotter.add_points(
                    np.vstack(self.manual_points),
                    color="#CBCB00",  # manual point
                    point_size=12,
                    render_points_as_spheres=True,
                )
            if self.manual_transformed_points:
                self.plotter.add_points(
                    np.vstack(self.manual_transformed_points),
                    color="#C48300",  # dark yellow for manual transform
                    point_size=12,
                    render_points_as_spheres=True,
                )

        if (
            self.camera_points
            and self.holo_points
            and self.cam_checkbox.isChecked()
            and self.holo_checkbox.isChecked()
        ):
            for cam, holo in zip(self.camera_points, self.holo_points):
                self.draw_dashed_line(cam, holo)

        self.plotter.render()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())