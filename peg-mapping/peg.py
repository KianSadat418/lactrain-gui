import numpy as np
import cv2

class Peg:
    def __init__(self, peg_id):
        self.id = peg_id
        self.kf = self._init_kalman_filter()
        self.initialized = False
        self.skipped_frames = 0
        self.last_position = None

    def _init_kalman_filter(self):
        kf = cv2.KalmanFilter(6, 3)  # state: [x, y, z, vx, vy, vz] ; measurement: [x, y, z]

        # Transition matrix F
        kf.transitionMatrix = np.eye(6, dtype=np.float32)
        for i in range(3):
            kf.transitionMatrix[i, i + 3] = 1  # position += velocity

        # Measurement matrix H
        kf.measurementMatrix = np.zeros((3, 6), dtype=np.float32)
        kf.measurementMatrix[0, 0] = 1
        kf.measurementMatrix[1, 1] = 1
        kf.measurementMatrix[2, 2] = 1

        # Noise covariances
        kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-3
        kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-2
        kf.errorCovPost = np.eye(6, dtype=np.float32)

        # Initial state
        kf.statePost = np.zeros((6, 1), dtype=np.float32)

        return kf

    def update(self, measured_pos):
        if measured_pos is not None and np.all(np.isfinite(measured_pos)):
            measured = np.array(measured_pos, dtype=np.float32).reshape(3, 1)

            if not self.initialized:
                # First update: set initial position
                self.kf.statePost[:3] = measured
                self.kf.statePost[3:] = 0  # initial velocity = 0
                self.initialized = True
                print(f"[INIT] Peg {self.id} initialized at {measured.flatten()}")
            else:
                self.kf.correct(measured)
                print(f"[UPDATE] Peg {self.id} corrected to {measured.flatten()}")

            self.skipped_frames = 0
        else:
            self.skipped_frames += 1
            print(f"[SKIP] Peg {self.id}: No valid measurement, using prediction")

        self.last_position = self.get_position()
        self.kf.predict()

    def get_position(self):
        return self.kf.statePost[:3].flatten()