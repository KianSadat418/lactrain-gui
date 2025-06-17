import sys
import socket
import json
import random
import time
from typing import List

import numpy as np
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets
import pyvista as pv
from pyvistaqt import QtInteractor


class DataReceiver(QtCore.QThread):
    """Thread that receives point pairs from a socket or generates synthetic data."""

    new_pair = QtCore.pyqtSignal(object, object)

    def __init__(self, host: str = "0.0.0.0", port: int = 9991, parent=None):
        super().__init__(parent)
        self.host = host
        self.port = port
        self._running = True
        self.test_mode = False

    def stop(self):
        self._running = False

    def run(self):
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            #server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            #server_socket.listen(1)
            #print(f"[Receiver] Listening on {self.host}:{self.port}...")
            #conn, _ = server_socket.accept()
            #print("[Receiver] Connection established.")
            #fileobj = conn.makefile()
        except Exception as e:
            print(f"[Receiver] Failed to start server: {e}")
            self.test_mode = True

        if self.test_mode:
            self._run_test_mode()
            return

        added_idx = []
        while self._running:
            try:
                #line = fileobj.readline()
                data, addr = server_socket.recvfrom(650000)
                if not data:
                    continue
                line = data.decode('utf-8')
                data = json.loads(line)
                for key, value in data.items():
                    if key not in added_idx:
                        added_idx.append(key)
                        cam = np.array(value[0])
                        holo = np.array(value[1])
                        print(f"[Receiver] Received pair {key}: Cam {cam}, Holo {holo}")
                        self.new_pair.emit(cam, holo)
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"[Receiver] Error receiving data: {e}")
                break

        #conn.close()
        #server_socket.close()

    def _run_test_mode(self):
        rng = np.random.default_rng()
        while self._running:
            cam = rng.random(3)
            holo = cam + rng.normal(scale=0.1, size=3)
            self.new_pair.emit(cam, holo)
            time.sleep(random.uniform(0.5, 1.0))


class MainWindow(QtWidgets.QWidget):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Point Viewer")

        # Data storage
        self.camera_points: List[np.ndarray] = []
        self.holo_points: List[np.ndarray] = []
        self.transform_points: List[np.ndarray] = []

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
        self.transform_apply = QtWidgets.QPushButton("Apply")
        transform_layout.addWidget(self.transform_checkbox)
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

        self.receiver = DataReceiver()
        self.receiver.new_pair.connect(self.add_pair)
        self.receiver.start()

    def closeEvent(self, event):
        self.receiver.stop()
        self.receiver.wait()
        super().closeEvent(event)

    @QtCore.pyqtSlot(object, object)
    def add_pair(self, cam, holo):
        self.camera_points.append(cam)
        self.holo_points.append(holo)
        # Keep only the last 12 pairs
        self.camera_points = self.camera_points[-12:]
        self.holo_points = self.holo_points[-12:]
        self.update_rmse()
        self.update_scene()

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

    def apply_transform(self):
        """Read transform point from fields and store it."""
        try:
            x = float(self.transform_x.text())
            y = float(self.transform_y.text())
            z = float(self.transform_z.text())
        except ValueError:
            return
        self.transform_points.append(np.array([x, y, z]))
        self.transform_points = self.transform_points[-12:]
        self.update_scene()

    def _zoom(self, factor: float):
        camera = self.plotter.camera
        if hasattr(camera, "Zoom"):
            camera.Zoom(factor)
        self.plotter.render()

    def draw_dashed_line(self, p1, p2, segments=12):
        points = np.linspace(p1, p2, segments * 2).reshape(-1, 2, 3)
        for i, (start, end) in enumerate(points):
            if i % 2 == 0:
                self.plotter.add_lines(np.array([start, end]), color="gray", width=4)

    def update_scene(self):
        self.plotter.clear()
        self.plotter.show_axes()
        self.plotter.show_grid()

        if self.camera_points and self.cam_checkbox.isChecked():
            self.plotter.add_points(
                np.vstack(self.camera_points),
                color="red",
                point_size=14,
                render_points_as_spheres=True,
            )

        if self.holo_points and self.holo_checkbox.isChecked():
            self.plotter.add_points(
                np.vstack(self.holo_points),
                color="blue",
                point_size=14,
                render_points_as_spheres=True,
            )

        if self.transform_points and self.transform_checkbox.isChecked():
            self.plotter.add_points(
                np.vstack(self.transform_points),
                color="green",
                point_size=18,
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
