import sys
import socket
import time
from datetime import datetime
import json
from typing import List
from collections import deque
import numpy as np
import os

import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets
import pyvista as pv
from pyvista import Color
pv.global_theme.allow_empty_mesh = True
from pyvistaqt import QtInteractor

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
                    "gaze_position": gaze_data.get("gaze_position", []),
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

                if line.startswith("D"):
                    data = json.loads(line[1:])
                    self.full_gaze_received.emit(
                        np.array(data.get("gaze_position")),
                        float(data.get("roi", 0.0)),
                        int(data.get("intercept", 0)),
                        float(data.get("gaze_distance", 0.0)),
                        np.array(data.get("pegs", []))
                    )

                else:
                    print(f"[Receiver] Unknown message type: {line}")
                    pass

            except Exception as e:
                print(f"[Receiver] Failed to process message: {e}")

        print("[Receiver] Shutting down cleanly.")


class GazeTrackingWindow(QtWidgets.QWidget):
    def __init__(self, transform_matrices=None, enable_receiver=True, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gaze Tracking")
        self.setMinimumSize(800, 600)

        # === Core Attributes ===
        self.recorder = Recorder()
        self.transform_matrices = transform_matrices or []
        self.latest_gaze_position = None
        self.latest_roi = 0.0
        self.latest_intercept = 0
        self.latest_pegs = []
        self.latest_gaze_distance = 0.0
        self.fix_view_actors = []
        self.peg_circle_actor = None  # Actor for the yellow circle around peg of interest
        self.peg_circle_mesh = None   # Mesh for the circle
        self.peg_circle_radius = 0.1  # Default radius, will be updated

        self.peg_smoothing_enabled = True  # toggle if needed
        self.peg_history = [deque(maxlen=5) for _ in range(6)]

        self.receiver = None
        if enable_receiver:
            self.receiver = DataReceiver(parent=self)
            self.receiver.full_gaze_received.connect(self.update_gaze_data)
            self.receiver.start()

        # === Plotter ===
        self.plotter = QtInteractor(self)
        
        # Set up the plotter for 2D view
        self.plotter.set_background('white')
        self.plotter.view_xy()  # Set to top-down view
        
        # Create a 2D grid for reference (smaller grid with 100mm spacing)
        x = np.linspace(-1, 1, 21)  # 20x20 grid with 100mm spacing
        y = np.linspace(-1, 1, 21)
        grid = pv.RectilinearGrid(x, y, [0])
        self.plotter.add_mesh(grid, color='lightgray', style='wireframe', opacity=0.3, line_width=0.5)
        
        # Show simple axes with labels
        self.plotter.show_axes()
        # Add axis labels
        self.plotter.add_text('X (mm)', position='lower_right', font_size=10, color='black')
        self.plotter.add_text('Y (mm)', position='upper_left', font_size=10, color='black')
        # Remove any existing scalar bars
        try:
            if hasattr(self.plotter, 'scalar_bars') and self.plotter.scalar_bars:
                for name in list(self.plotter.scalar_bars.keys()):
                    self.plotter.remove_scalar_bar(name)
        except:
            pass
        
        # Initialize peg points (2D) with explicit float32 type to avoid warnings
        self.peg_points = pv.PolyData(np.zeros((6, 3), dtype=np.float32))
        self.peg_actor = self.plotter.add_points(
            self.peg_points, 
            color="blue", 
            point_size=8,  # Smaller point size
            render_points_as_spheres=True
        )
        
        # Initialize gaze point (2D) with explicit float32 type
        self.gaze_point = pv.PolyData(np.array([[0, 0, 0]], dtype=np.float32))
        self.gaze_actor = self.plotter.add_points(
            self.gaze_point,
            color="red",
            point_size=10,  # Slightly larger than peg points
            render_points_as_spheres=True
        )
        
        # Set initial camera position for 2D view
        self.plotter.camera_position = 'xy'
        self.plotter.camera.parallel_projection = True  # Use orthographic projection
        self.plotter.camera.parallel_scale = 1  # Larger view to see more of the grid

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
        right_panel.addWidget(self.peg_selector_group)
        right_panel.addWidget(view_group)
        right_panel.addWidget(self.gaze_distance_label)
        right_panel.addStretch()

        # === Timer Display ===
        timer_layout = QtWidgets.QHBoxLayout()
        
        # Current recording timer
        self.timer_label = QtWidgets.QLabel("00:00.000")
        self.timer_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        
        # Last recorded time
        last_time_layout = QtWidgets.QHBoxLayout()
        last_time_label = QtWidgets.QLabel("Last Time:")
        self.last_time_display = QtWidgets.QLabel("00:00.000")
        self.last_time_display.setStyleSheet("font-size: 14px;")
        last_time_layout.addWidget(last_time_label)
        last_time_layout.addWidget(self.last_time_display)
        last_time_layout.addStretch()
        
        # Add to timer layout
        timer_layout.addWidget(self.timer_label)
        timer_layout.addStretch()
        timer_layout.addLayout(last_time_layout)
        
        # === Recording Controls ===
        self.start_recording_button = QtWidgets.QPushButton("Start Recording")
        self.stop_recording_button = QtWidgets.QPushButton("Stop & Save Recording")
        self.stop_recording_button.setEnabled(False)

        # Recording indicator
        self.record_indicator = QtWidgets.QLabel("● REC")
        self.record_indicator.setStyleSheet("color: red; font-weight: bold;")
        self.record_indicator.hide()

        right_panel.addLayout(timer_layout)
        right_panel.addWidget(self.start_recording_button)
        right_panel.addWidget(self.stop_recording_button)
        right_panel.addWidget(self.record_indicator)
        right_panel.addStretch()
        
        # Connect button signals
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
        self.render_timer.setInterval(60)  # ~17 FPS
        self.render_timer.timeout.connect(self._throttled_render)
        self.render_timer.start()
        self._needs_render = False

        # Timer used to log frames while recording
        self.record_timer = QtCore.QTimer()
        self.record_timer.setInterval(60)
        self.record_timer.timeout.connect(self._log_current_frame)

        # Timer used to update recording time
        self.recording_timer = QtCore.QTimer()
        self.recording_timer.timeout.connect(self._update_recording_timer)
        self.recording_duration = 0.0
        self.last_recorded_time = 0.0
        self.record_start_time = None

    def set_peg_of_interest(self, index: int):
        # Store the 0-based index for Python use
        self.peg_of_interest_index = index
        print(f"[GazeTracking] Peg of interest set to: {index}")
        self._update_gaze_position()

        # Send peg selection to Unity (convert to 1-based for C#)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Convert to 1-based index for C#
            message = f'P{index + 1}'
            sock.sendto(message.encode('utf-8'), ('127.0.0.1', 9989))  # Match Unity's `listenPort`
            sock.close()
        except Exception as e:
            print(f"[GazeTracking] Failed to send peg selection: {e}")

    @QtCore.pyqtSlot(object, float, int, float, object)
    def update_gaze_data(self, gaze_position, roi, intercept, gaze_distance, pegs):
        try:
            # Initialize if needed
            if not hasattr(self, 'latest_gaze_position'):
                self.latest_gaze_position = np.array([0, 0, 0])
            if not hasattr(self, 'latest_pegs'):
                self.latest_pegs = np.zeros((6, 3))
            
            # Skip processing if window isn't visible
            if not self.isVisible():
                return
            
            try:
                # Process gaze position
                new_gaze_pos = None
                if gaze_position is not None and hasattr(gaze_position, '__len__') and len(gaze_position) >= 2:
                    new_gaze_pos = np.array([float(gaze_position[0]), float(gaze_position[1]), 0])
                
                # Process pegs
                new_pegs = None
                if pegs is not None and hasattr(pegs, '__len__') and len(pegs) > 0:
                    pegs_2d = []
                    for p in pegs:
                        if hasattr(p, '__len__') and len(p) >= 2:
                            pegs_2d.append([float(p[0]), float(p[1]), 0])
                        else:
                            pegs_2d.append([0, 0, 0])
                    
                    if len(pegs_2d) >= 6:
                        new_pegs = np.array(pegs_2d[:6])
                    else:
                        new_pegs = np.vstack([np.array(pegs_2d), np.zeros((6 - len(pegs_2d), 3))])
                
                # Only update if we have new data
                needs_update = False
                
                if new_gaze_pos is not None:
                    if not hasattr(self, 'latest_gaze_position') or not np.array_equal(self.latest_gaze_position, new_gaze_pos):
                        self.latest_gaze_position = new_gaze_pos
                        needs_update = True
                
                if new_pegs is not None:
                    if not hasattr(self, 'latest_pegs') or not np.array_equal(self.latest_pegs, new_pegs):
                        self.latest_pegs = new_pegs
                        needs_update = True
                
                # Update other state
                new_intercept = bool(intercept) if intercept is not None else self.latest_intercept if hasattr(self, 'latest_intercept') else False
                new_gaze_distance = float(gaze_distance) if gaze_distance is not None else self.latest_gaze_distance if hasattr(self, 'latest_gaze_distance') else 0.0
                
                if not hasattr(self, 'latest_intercept') or not hasattr(self, 'latest_gaze_distance') or \
                   self.latest_intercept != new_intercept or self.latest_gaze_distance != new_gaze_distance:
                    self.latest_intercept = new_intercept
                    self.latest_gaze_distance = new_gaze_distance
                    needs_update = True
                    
                    # Update UI elements
                    self.gaze_distance_label.setText(f"Gaze Distance: {self.latest_gaze_distance:.2f} mm")
                
                # Only trigger visualization update if something changed
                if needs_update:
                    self._update_gaze_position()
                
            except Exception as e:
                print(f"[GazeTracking] Error processing data: {e}")
                import traceback
                traceback.print_exc()
            
        except Exception as e:
            print(f"[GazeTracking] Failed to update 2D visualization: {e}")
            import traceback
            traceback.print_exc()

    def _format_time(self, seconds):
        """Format seconds into MM:SS.mmm format."""
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:06.3f}"
        
    def _update_recording_timer(self):
        """Update the recording timer display."""
        if self.record_start_time is not None:
            self.recording_duration = time.time() - self.record_start_time
            self.timer_label.setText(self._format_time(self.recording_duration))
    
    def _throttled_render(self):
        """Render only if needed and window is visible."""
        if self._needs_render and self.isVisible():
            self.plotter.render()
            self._needs_render = False

    def _update_gaze_position(self):
        if not hasattr(self, 'latest_pegs') or not hasattr(self, 'latest_gaze_position') or not self.isVisible():
            return
            
        try:
            needs_update = False
            
            # Handle peg updates
            if len(self.latest_pegs) > 0:
                latest_pegs = np.array(self.latest_pegs, dtype=np.float32)
                
                if not hasattr(self, 'peg_points'):
                    self.peg_points = pv.PolyData(latest_pegs)
                    self.peg_actor = self.plotter.add_points(
                        self.peg_points,
                        color="blue",
                        point_size=8,
                        render_points_as_spheres=True
                    )
                    needs_update = True
                elif not np.array_equal(self.peg_points.points, latest_pegs):
                    self.peg_points.points = latest_pegs
                    needs_update = True
            
            # Handle gaze point updates
            if len(self.latest_gaze_position) > 0:
                gaze_pos = np.array([self.latest_gaze_position], dtype=np.float32)
                current_color = "green" if self.latest_intercept else "red"
                
                if not hasattr(self, 'gaze_actor'):
                    self.gaze_point = pv.PolyData(gaze_pos)
                    self.gaze_actor = self.plotter.add_points(
                        self.gaze_point,
                        color=current_color,
                        point_size=10,
                        render_points_as_spheres=True
                    )
                    needs_update = True
                else:
                    # Only update if position or color has changed
                    if not np.array_equal(self.gaze_point.points, gaze_pos):
                        self.gaze_point.points = gaze_pos
                        needs_update = True
                    
                    if not np.array_equal(self.gaze_actor.prop.color, current_color):
                        self.gaze_actor.prop.color = current_color
                        needs_update = True
            
            # Update yellow circle around peg of interest
            if hasattr(self, 'latest_pegs') and len(self.latest_pegs) > self.peg_of_interest_index:
                peg_pos = self.latest_pegs[self.peg_of_interest_index]
                if len(peg_pos) >= 2:  # Ensure we have at least x,y coordinates
                    try:
                        radius = max(0.04, self.latest_roi)  # Ensure radius is positive
                        
                        # Create points for the circle
                        theta = np.linspace(0, 2 * np.pi, 50)
                        x = radius * np.cos(theta) + peg_pos[0]
                        y = radius * np.sin(theta) + peg_pos[1]
                        z = np.zeros_like(x)
                        points = np.column_stack((x, y, z))
                        
                        # Close the circle
                        points = np.vstack([points, points[0]])
                        
                        # Create line segments
                        lines = np.column_stack(
                            [np.ones(len(points)-1, dtype=np.int32) * 2,
                             np.arange(len(points)-1, dtype=np.int32),
                             np.arange(1, len(points), dtype=np.int32)]
                        )
                        
                        if self.peg_circle_mesh is None or radius != self.peg_circle_radius:
                            # Create new mesh if needed
                            self.peg_circle_radius = radius
                            self.peg_circle_mesh = pv.PolyData(points, lines=lines)
                            
                            # Remove old actor if it exists
                            if self.peg_circle_actor is not None:
                                self.plotter.remove_actor(self.peg_circle_actor)
                            
                            # Add new mesh to plotter
                            self.peg_circle_actor = self.plotter.add_mesh(
                                self.peg_circle_mesh,
                                color='yellow',
                                line_width=2,
                                name='peg_circle',
                                opacity=0.7
                            )
                        else:
                            # Update existing mesh
                            self.peg_circle_mesh.points = points
                            self.peg_circle_mesh.lines = lines
                            self.peg_circle_mesh.Modified()
                        
                        needs_update = True
                        
                    except Exception as e:
                        print(f"[GazeTracking] Error updating circle: {e}")
                        import traceback
                        traceback.print_exc()
            
            # Schedule render if needed
            if needs_update:
                self._needs_render = True
                
        except Exception as e:
            print(f"[GazeTracking] Error updating 2D visualization: {e}")
            import traceback
            traceback.print_exc()

    def reset_view(self):
        try:
            # Set to top-down 2D view
            self.plotter.view_xy()
            
            # Reset camera to default position
            self.plotter.camera_position = 'xy'
            
            # Use parallel projection for true 2D view
            self.plotter.camera.parallel_projection = True
            
            # Set the view to show the same area as initial view (-1000 to 1000 in both X and Y)
            self.plotter.camera.parallel_scale = 1  # Shows 2000 units total (1000 in each direction)
            
            # Position the camera to look at the center of the grid
            self.plotter.camera.position = (0, 0, 1)    # Slightly above the XY plane
            self.plotter.camera.focal_point = (0, 0, 0)  # Look at the origin
            self.plotter.camera.up = (0, 1, 0)          # Y is up
            
            # Force a render update
            self.plotter.render()
            
        except Exception as e:
            print(f"[Playback] Error resetting 2D view: {e}")
            import traceback
            traceback.print_exc()

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

    def _save_recording(self):
        self.recorder.stop()
        self.record_timer.stop()
        self.recording_timer.stop()
        self.record_indicator.hide()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Recording", "", "JSON Files (*.json)")
        if path:
            self.recorder.save(path)
            self.last_recorded_time = self.recording_duration
            self.last_time_display.setText(self._format_time(self.last_recorded_time))
        
        # Reset UI state
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
        self.recording_timer.start(50)  # Update every 50ms for smooth display
        self.record_timer.start()

    def _log_current_frame(self):
        if not self.recorder.recording:
            return
        if self.latest_gaze_position is None:
            return
        gaze_dict = {
            "gaze_position": self.latest_gaze_position.tolist(),
            "roi": self.latest_roi,
            "intercept": self.latest_intercept,
            "gaze_distance": self.latest_gaze_distance,
        }
        peg_list = (
            self.latest_pegs.tolist()
            if isinstance(self.latest_pegs, np.ndarray)
            else self.latest_pegs
        )
        # Use default transform_id of -1 since matrix_group is not available in 2D version
        self.recorder.log_frame(gaze_dict, peg_list, -1)

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
        self.intercept_history = []  # Track intercept history for consecutive frames
        self.trajectory_initialized = False  # Track if trajectories have been initialized
        self._needs_render = False  # Track if a render is needed

        # === PyVista Plotter ===
        self.plotter = QtInteractor(self)
        
        # Set up the plotter for 2D view
        self.plotter.set_background('white')
        self.plotter.view_xy()  # Set to top-down view
        
        # Create a 2D grid for reference (smaller grid with 100mm spacing)
        x = np.linspace(-1, 1, 21)  # 20x20 grid with 100mm spacing
        y = np.linspace(-1, 1, 21)
        grid = pv.RectilinearGrid(x, y, [0])
        self.plotter.add_mesh(grid, color='lightgray', style='wireframe', opacity=0.3, line_width=0.5)
        
        # Show simple axes with labels
        self.plotter.show_axes()
        self.plotter.add_text('X (mm)', position='lower_right', font_size=10, color='black')
        self.plotter.add_text('Y (mm)', position='upper_left', font_size=10, color='black')
        
        # Initialize peg points (2D) with explicit float32 type to avoid warnings
        self.peg_points = pv.PolyData(np.zeros((6, 3), dtype=np.float32))
        self.peg_actor = self.plotter.add_points(
            self.peg_points, 
            color="blue", 
            point_size=8,  # Smaller point size
            render_points_as_spheres=True
        )
        
        # Initialize gaze point (2D) with explicit float32 type
        self.gaze_point = pv.PolyData(np.array([[0, 0, 0]], dtype=np.float32))
        self.gaze_actor = self.plotter.add_points(
            self.gaze_point,
            color="red",
            point_size=10,  # Slightly larger than peg points
            render_points_as_spheres=True
        )
        
        # Initialize trajectory points
        self.trajectory_meshes = []
        self.trajectory_actors = []
        
        # Set initial camera position for 2D view
        self.plotter.camera_position = 'xy'
        self.plotter.camera.parallel_projection = True
        self.plotter.camera.parallel_scale = 1

        # === Timer for Playback ===
        self.timer = QtCore.QTimer()
        self.timer.setInterval(60)  # ~17 FPS
        self.timer.timeout.connect(self.advance_frame)
        
        # Render timer for throttled updates
        self.render_timer = QtCore.QTimer()
        self.render_timer.setInterval(60)  # ~17 FPS
        self.render_timer.timeout.connect(self._throttled_render)
        self.render_timer.start()

        # === Viewer Controls ===
        self.reset_button = QtWidgets.QPushButton("Reset View")
        self.zoom_in_button = QtWidgets.QPushButton("+")
        self.zoom_out_button = QtWidgets.QPushButton("-")
        self.fix_view_checkbox = QtWidgets.QCheckBox("Fix View")
        self.fix_view_checkbox.setChecked(False)

        # === Analysis Buttons ===
        self.heatmap_button = QtWidgets.QPushButton("Show Fixation Heatmap")
        self.heatmap_button.clicked.connect(self.generate_heatmap)
        
        self.gaze_heatmap_button = QtWidgets.QPushButton("Show Gaze Heatmap")
        self.gaze_heatmap_button.clicked.connect(self.generate_gaze_heatmap)

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
        controls_layout.addWidget(self.gaze_heatmap_button)
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
        self.trajectory_checkbox.stateChanged.connect(lambda _: self.update_frame(self.current_index if self.frames else None))
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
            
            # Reset playback state
            self.scrubber.setMaximum(len(self.frames) - 1)
            self.current_index = 0
            self.fixation_points = []
            self.intercept_history = []
            
            # Enable/disable controls
            has_frames = len(self.frames) > 0
            self.play_button.setEnabled(has_frames)
            self.pause_button.setEnabled(False)
            self.restart_button.setEnabled(has_frames)
            self.scrubber.setEnabled(has_frames)
            
            # Update to first frame
            if has_frames:
                self.update_frame(0)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load Error", f"Failed to load recording: {e}")

    def _throttled_render(self):
        """Render only if needed and window is visible."""
        if self._needs_render and self.isVisible():
            self.plotter.render()
            self._needs_render = False

    def update_frame(self, idx=None):
        if not self.frames:
            return

        # Handle index bounds
        if idx is None or not (0 <= idx < len(self.frames)):
            idx = 0
            
        # Only update if index has changed
        if idx == self.current_index and hasattr(self, '_actors_initialized'):
            return
            
        self.current_index = idx
        self.scrubber.setValue(idx)
        
        frame = self.frames[idx]
        gaze_data = frame.get("gaze_data", {})
        pegs = np.array(frame.get("peg_data", []))
        
        # Update intercept history for fixation detection
        current_intercept = gaze_data.get("intercept", 0) == 1
        self.intercept_history.append(current_intercept)
        
        # Keep only last 3 frames of intercept history
        if len(self.intercept_history) > 3:
            self.intercept_history.pop(0)
            
        # Track fixation points (3 consecutive frames with intercept)
        if len(self.intercept_history) >= 3 and all(self.intercept_history[-3:]):
            if "gaze_position" in gaze_data and len(gaze_data["gaze_position"]) >= 2:
                end_point = np.array([gaze_data["gaze_position"][0], gaze_data["gaze_position"][1], 0])
                # Only add if different from last point
                if not self.fixation_points or not np.array_equal(self.fixation_points[-1], end_point):
                    self.fixation_points.append(end_point)
        
        # Initialize actors on first frame
        if not hasattr(self, '_actors_initialized'):
            self.plotter.clear()
            self.plotter.show_axes()
            self.plotter.show_grid()
            
            # Create 2D grid for reference
            x = np.linspace(-1000, 1000, 21)
            y = np.linspace(-1000, 1000, 21)
            grid = pv.RectilinearGrid(x, y, [0])
            self.plotter.add_mesh(grid, color='lightgray', style='wireframe', opacity=0.3, line_width=0.5)
            
            # Initialize gaze point
            gaze_pos = gaze_data.get("gaze_position", [0, 0])
            self.gaze_point = pv.PolyData(np.array([[gaze_pos[0], gaze_pos[1], 0]]))
            self.gaze_actor = self.plotter.add_points(
                self.gaze_point,
                color="red",
                point_size=10,
                render_points_as_spheres=True
            )
            
            # Initialize pegs
            self.peg_points = pv.PolyData(pegs if pegs.size > 0 else np.zeros((6, 3)))
            self.peg_actor = self.plotter.add_points(
                self.peg_points,
                color="blue",
                point_size=8,
                render_points_as_spheres=True
            )
            
            # Initialize trajectory meshes
            self.trajectory_meshes = []
            self.trajectory_actors = []
            peg_count = len(pegs) if pegs.size > 0 else 6
            for _ in range(peg_count):
                mesh = pv.PolyData()
                self.trajectory_meshes.append(mesh)
                actor = self.plotter.add_mesh(mesh, color="green", line_width=2)
                self.trajectory_actors.append(actor)
            
            self._actors_initialized = True
        else:
            # Update existing actors
            if "gaze_position" in gaze_data and len(gaze_data["gaze_position"]) >= 2:
                gaze_pos = gaze_data["gaze_position"]
                self.gaze_point.points = np.array([[gaze_pos[0], gaze_pos[1], 0]])
            
            if pegs.size > 0:
                self.peg_points.points = pegs
        
        # Update trajectories if enabled
        if hasattr(self, "trajectory_checkbox") and self.trajectory_checkbox.isChecked():
            self._update_trajectories(idx)
        
        # Request render - handled by _throttled_render

    def generate_heatmap(self):
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

        print(f"[Heatmap] Fixation heatmap saved to {os.path.abspath(save_path)}")
        
    def generate_gaze_heatmap(self):
        if not self.frames:
            QtWidgets.QMessageBox.information(self, "No Data", "No gaze data available.")
            return
            
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde
        import os
        
        # Extract all gaze end points (tip of the gaze line)
        gaze_points = []
        for frame in self.frames:
            gaze_position = frame["gaze_data"]["gaze_position"]
            if len(gaze_position) >= 2:  # Ensure we have both start and end points
                # Get the end point of the gaze line (point B)
                end_point = np.array(gaze_position)
                gaze_points.append(end_point)
        
        if not gaze_points:
            QtWidgets.QMessageBox.information(self, "No Data", "No valid gaze points found.")
            return
            
        points = np.array(gaze_points)
        x = points[:, 0]
        y = points[:, 1]  # Using X-Y plane

        # Kernel density estimate for smooth heatmap
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy)
        
        # Create grid for the heatmap
        xi, yi = np.mgrid[x.min():x.max():500j, y.min():y.max():500j]
        zi = kde(np.vstack([xi.flatten(), yi.flatten()]))

        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.title("Gaze Position Heatmap (X-Y Plane)")
        plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading="auto", cmap="viridis")
        plt.colorbar(label="Gaze Density")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.tight_layout()

        save_path = "gaze_heatmap.png"
        plt.savefig(save_path)
        plt.show()

        print(f"[Heatmap] Gaze heatmap saved to {os.path.abspath(save_path)}")

    def reset_view(self):
        """Reset the view to match the initial 2D view."""
        try:
            # Set to top-down 2D view
            self.plotter.view_xy()
            
            # Reset camera to default position
            self.plotter.camera_position = 'xy'
            
            # Use parallel projection for true 2D view
            self.plotter.camera.parallel_projection = True
            
            # Set the view to show the same area as initial view (-1000 to 1000 in both X and Y)
            self.plotter.camera.parallel_scale = 1000  # Shows 2000 units total (1000 in each direction)
            
            # Position the camera to look at the center of the grid
            self.plotter.camera.position = (0, 0, 1)    # Slightly above the XY plane
            self.plotter.camera.focal_point = (0, 0, 0)  # Look at the origin
            self.plotter.camera.up = (0, 1, 0)          # Y is up
            
            # Force a render update
            self.plotter.render()
            
        except Exception as e:
            print(f"[Playback] Error resetting 2D view: {e}")
            import traceback
            traceback.print_exc()

    def zoom_in(self):
        """Zoom in by a factor."""
        try:
            self.plotter.camera.zoom(1.5)
        except Exception as e:
            print(f"[GazeTracking] Error zooming in: {e}")

    def zoom_out(self):
        """Zoom out by a factor."""
        try:
            self.plotter.camera.zoom(0.75)
        except Exception as e:
            print(f"[GazeTracking] Error zooming out: {e}")

    def _initialize_trajectories(self):
        """Initialize trajectory meshes for peg tracking."""
        if not hasattr(self, 'trajectory_meshes') or not self.trajectory_meshes:
            self.trajectory_meshes = []
            self.trajectory_actors = []
            
            # Create a trajectory mesh for each peg
            peg_count = len(self.frames[0]["peg_data"]) if self.frames else 6
            for i in range(peg_count):
                mesh = pv.PolyData()
                actor = self.plotter.add_mesh(
                    mesh, 
                    color="blue", 
                    line_width=2,
                    opacity=0.5
                )
                self.trajectory_meshes.append(mesh)
                self.trajectory_actors.append(actor)
    
    def _update_trajectories(self, current_frame_idx):
        if not self.trajectory_initialized or current_frame_idx < 1:
            return False  # Return whether trajectories were updated

        needs_update = False
        
        # Update each peg's trajectory
        for i in range(6):
            if i >= len(self.trajectory_meshes):
                # Initialize new trajectory
                self.trajectory_meshes.append(None)
                self.trajectory_actors.append(None)

            # Get all positions up to current frame for this peg
            positions = []
            for frame in self.frames[:current_frame_idx + 1]:
                pegs = frame.get("peg_data", [])
                if i < len(pegs) and len(pegs[i]) >= 2:
                    positions.append([pegs[i][0], pegs[i][1], 0])

            if len(positions) < 2:
                if self.trajectory_actors[i] is not None:
                    self.plotter.remove_actor(self.trajectory_actors[i])
                    self.trajectory_actors[i] = None
                    self.trajectory_meshes[i] = None
                    needs_update = True
                continue

            # Create line from positions
            trajectory = pv.lines_from_points(np.array(positions))
            
            # Only update if trajectory has changed
            if self.trajectory_meshes[i] is None or \
               not np.array_equal(self.trajectory_meshes[i].points, trajectory.points):
                
                # Remove old actor if it exists
                if self.trajectory_actors[i] is not None:
                    self.plotter.remove_actor(self.trajectory_actors[i])
                
                # Add new trajectory
                self.trajectory_meshes[i] = trajectory
                self.trajectory_actors[i] = self.plotter.add_mesh(
                    trajectory, color="green", line_width=2, name=f"trajectory_{i}",
                    render_lines_as_tubes=False  # Faster rendering for lines
                )
                needs_update = True

        return needs_update

    
    def fix_view_bounds(self):
        # Remove any previously added actors
        if hasattr(self, "fix_view_actors"):
            for actor in self.fix_view_actors:
                self.plotter.remove_actor(actor)
            self.fix_view_actors = []
        else:
            self.fix_view_actors = []

        if self.fix_view_checkbox.isChecked():
            # For 2D view, we'll use a fixed range that matches our grid
            bounds = np.array([[-1000, -1000, 0], [1000, 1000, 0]], dtype=np.float32)
            origin = np.array([[0, 0, 0]], dtype=np.float32)

            # Add bounds (invisible, just to fix the view)
            bounds_actor = self.plotter.add_points(
                bounds,
                color="white",
                opacity=0.0,
                point_size=1,
                render_points_as_spheres=True
            )
            
            # Add origin point (visible)
            origin_actor = self.plotter.add_points(
                origin,
                color="black",
                point_size=5,
                render_points_as_spheres=True
            )

            self.fix_view_actors.extend([bounds_actor, origin_actor])
            
            # Reset the view to show the full grid
            self.reset_view()

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


class MainMenuWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mixed Reality Surgical Training System 2D")
        self.setMinimumSize(400, 300)

        layout = QtWidgets.QVBoxLayout()

        # Info Label
        info_label = QtWidgets.QLabel(
            "<h2>Welcome to the Surgical Training GUI 2D</h2>"
            "<p>Select a mode to begin:</p>"
        )
        info_label.setAlignment(QtCore.Qt.AlignCenter)

        # Buttons
        gaze_tracking_btn = QtWidgets.QPushButton("Gaze Tracking Mode")
        playback_btn = QtWidgets.QPushButton("Playback Mode")

        gaze_tracking_btn.clicked.connect(self.launch_gaze_tracking)
        playback_btn.clicked.connect(self.launch_playback)

        layout.addWidget(info_label)
        layout.addStretch()
        layout.addWidget(gaze_tracking_btn)
        layout.addWidget(playback_btn)
        layout.addStretch()

        self.setLayout(layout)

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
