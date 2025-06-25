import numpy as np
import cv2

class Peg:
    def __init__(self, peg_id):
        self.id = peg_id
        self.kf = self._init_kalman_filter()
        self.last_valid = None
        self.initialized = False
        self.skipped_frames = 0

    def _init_kalman_filter(self):
        kf = cv2.KalmanFilter(6, 3)  # [x, y, z, vx, vy, vz] -> [x, y, z]

        # Transition matrix (F)
        kf.transitionMatrix = np.eye(6, dtype=np.float32)
        kf.transitionMatrix[0, 3] = 1
        kf.transitionMatrix[1, 4] = 1
        kf.transitionMatrix[2, 5] = 1

        # Measurement matrix (H)
        kf.measurementMatrix = np.zeros((3, 6), dtype=np.float32)
        kf.measurementMatrix[0, 0] = 1
        kf.measurementMatrix[1, 1] = 1
        kf.measurementMatrix[2, 2] = 1

        kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-1
        kf.errorCovPost = np.eye(6, dtype=np.float32)

        return kf

    def update(self, measured_pos):
        if measured_pos is not None and np.all(np.isfinite(measured_pos)):
            measured = np.array(measured_pos, dtype=np.float32).reshape(3, 1)
            self.kf.correct(measured)
            self.last_valid = measured_pos
            self.skipped_frames = 0
            self.initialized = True
        else:
            self.skipped_frames += 1

        self.kf.predict()

    def get_position(self):
        predicted = self.kf.predict()
        return predicted[:3].flatten()