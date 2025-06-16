import sys
import socket
import random
import time
from typing import List

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyvista as pv
from pyvistaqt import QtInteractor


class DataReceiver(QtCore.QThread):
    """Thread that receives point pairs from a socket or generates synthetic data."""

    new_pair = QtCore.pyqtSignal(object, object)

    def __init__(self, host: str = "localhost", port: int = 50007, parent=None):
        super().__init__(parent)
        self.host = host
        self.port = port
        self._running = True
        self.test_mode = False

    def stop(self):
        self._running = False

    def run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.host, self.port))
            fileobj = sock.makefile()
        except OSError:
            # Socket unavailable; switch to synthetic data mode
            self.test_mode = True

        if self.test_mode:
            self._run_test_mode()
            return

        while self._running:
            line = fileobj.readline()
            if not line:
                time.sleep(0.1)
                continue
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            try:
                nums = list(map(float, parts))
            except ValueError:
                continue
            cam = np.array(nums[:3])
            holo = np.array(nums[3:])
            self.new_pair.emit(cam, holo)
        sock.close()

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

        # UI setup
        self.plotter = QtInteractor(self)
        self.cam_checkbox = QtWidgets.QCheckBox("Show Camera Points")
        self.cam_checkbox.setChecked(True)
        self.holo_checkbox = QtWidgets.QCheckBox("Show HoloLens Points")
        self.holo_checkbox.setChecked(True)
        self.rmse_label = QtWidgets.QLabel("RMSE: N/A")

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.cam_checkbox)
        controls.addWidget(self.holo_checkbox)
        controls.addStretch()
        controls.addWidget(self.rmse_label)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.plotter.interactor)

        self.cam_checkbox.stateChanged.connect(self.update_scene)
        self.holo_checkbox.stateChanged.connect(self.update_scene)

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

    def update_scene(self):
        self.plotter.clear()
        if self.camera_points and self.cam_checkbox.isChecked():
            self.plotter.add_points(np.vstack(self.camera_points), color="red", point_size=10,
                                    render_points_as_spheres=True)
        if self.holo_points and self.holo_checkbox.isChecked():
            self.plotter.add_points(np.vstack(self.holo_points), color="blue", point_size=10,
                                    render_points_as_spheres=True)
        # Draw lines between pairs
        if self.camera_points and self.holo_points:
            for cam, holo in zip(self.camera_points, self.holo_points):
                pts = np.array([cam, holo])
                self.plotter.add_lines(pts, color="green")
        self.plotter.reset_camera()
        self.plotter.render()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())