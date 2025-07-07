import numpy as np
from collections import deque
import cv2
from collections import deque
import numpy as np
from scipy.optimize import linear_sum_assignment

class Peg:
    def __init__(self, peg_id, initial_position_3d, max_history=500):
        self.id = peg_id
        self.kf = self._init_kalman_filter()
        self.kf.statePost[:3, 0] = initial_position_3d
        self.kf.statePost[3:, 0] = [0, 0, 0]

        self.position_3d = np.array(initial_position_3d, dtype=np.float32)
        self.velocity_3d = np.zeros(3, dtype=np.float32)
        self.missing_frames = 0
        self.movement_history = deque(maxlen=max_history)
        self.position_log = []
        self.velocity_log = []

    def _init_kalman_filter(self):
        kf = cv2.KalmanFilter(6, 3)
        dt = 1.0

        kf.transitionMatrix = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        kf.measurementMatrix = np.eye(3, 6, dtype=np.float32)
        kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-4
        kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-1
        kf.errorCovPost = np.eye(6, dtype=np.float32)

        return kf

    def predict(self):
        pred = self.kf.predict()
        return pred[:3].flatten()

    def correct(self, measurement_3d):
        measured = np.array(measurement_3d, dtype=np.float32).reshape(3, 1)
        corrected = self.kf.correct(measured)
        new_pos = corrected[:3].flatten()

        velocity_estimate = new_pos - self.position_3d
        alpha = 0.3
        self.velocity_3d = alpha * velocity_estimate + (1 - alpha) * self.velocity_3d
        self.position_3d = new_pos

        speed = np.linalg.norm(velocity_estimate)
        self.movement_history.append(speed)
        self.position_log.append(new_pos.tolist())
        self.velocity_log.append(speed)

        return new_pos

    def is_actively_moving(self, threshold=3.0):
        """Returns True if average movement exceeds threshold."""
        if len(self.movement_history) < 5:
            return False
        avg_movement = np.mean(list(self.movement_history)[-5:])
        return avg_movement > threshold


class PegTracker:
    def __init__(self, movement_threshold=3.0, max_missing_frames=10):
        self.pegs = {}  # peg_id: Peg
        self.initialized = False
        self.max_missing_frames = max_missing_frames
        self.movement_threshold = movement_threshold
        self.frame_count = 0

        # New movement mode vars
        self.movement_phase = False
        self.moving_peg_id = None

    def initialize(self, detections_3d):
        assert len(detections_3d) == 6, "Expected 6 pegs"
        self.pegs = {i: Peg(i, pos) for i, pos in enumerate(detections_3d)}
        self.initialized = True
        print("[INIT] PegTracker initialized with 6 pegs.")

    def update(self, detections_3d):
        self.frame_count += 1
        if not self.initialized:
            print("[WARN] Tracker not initialized.")
            return

        peg_ids = list(self.pegs.keys())
        predictions = [self.pegs[i].predict() for i in peg_ids]

        cost_matrix = np.zeros((6, len(detections_3d)))
        for i, pred in enumerate(predictions):
            for j, det in enumerate(detections_3d):
                cost_matrix[i, j] = np.linalg.norm(pred - det)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched = {}

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 50:
                matched[i] = detections_3d[j]
            else:
                continue  # Skip if match too far

        # If not in motion phase, detect moving peg
        if not self.movement_phase:
            motions = {
                pid: np.mean(self.pegs[pid].movement_history)
                for pid in matched
                if len(self.pegs[pid].movement_history) > 3
            }
            for pid, det in matched.items():
                pos = self.pegs[pid].correct(det)
            if motions:
                max_pid = max(motions, key=motions.get)
                if motions[max_pid] > self.movement_threshold:
                    self.movement_phase = True
                    self.moving_peg_id = max_pid
                    print(f"[MOVE] Peg {max_pid} is moving. Press Enter to finalize.")
        else:
            # Only update moving peg
            if self.moving_peg_id in matched:
                self.pegs[self.moving_peg_id].correct(matched[self.moving_peg_id])
            for pid in self.pegs:
                if pid != self.moving_peg_id:
                    self.pegs[pid].predict()

    def finalize_movement(self):
        print(f"[FINALIZE] Peg {self.moving_peg_id} finalized.")
        self.movement_phase = False
        self.moving_peg_id = None

    def get_peg_positions(self):
        result = {}
        for pid in range(6):
            if pid in self.pegs:
                peg = self.pegs[pid]
                result[pid] = peg.position_3d.tolist()
            else:
                result[pid] = [None, None, None]
        return result

    def get_moving_peg_id(self):
        return self.moving_peg_id

    
class Triangulator:
    def __init__(self, K1, D1, K2, D2, R, T, transformation_matrix, image_size):
        self.K1 = K1
        self.D1 = D1
        self.K2 = K2
        self.D2 = D2
        self.R = R
        self.T = T
        self.transformation_matrix = transformation_matrix
        self.image_size = image_size

        # Compute projection matrices
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            K1, D1, K2, D2, image_size, R, T, alpha=0, flags=cv2.CALIB_ZERO_DISPARITY
        )

        self.baseline_offset = T[0][0] / 2.0

    def triangulate_points(self, left_points, right_points):
        """Triangulate 3D points from corresponding 2D coordinates in left/right images."""
        assert len(left_points) == len(right_points), "Left/right lists must be same length"

        points_3d = []
        for left_pt, right_pt in zip(left_points, right_points):
            pl = np.array(left_pt).reshape(2, 1)
            pr = np.array(right_pt).reshape(2, 1)

            point_4d = cv2.triangulatePoints(self.P1, self.P2, pl, pr)
            point_3d = point_4d[:3] / point_4d[3]  # Convert homogeneous to 3D
            x, y, z = point_3d.flatten()
            x += self.baseline_offset  # Optional: re-center
            y *= -1  # Optional: flip y-axis for Unity
            points_3d.append((x, y, z))

        return points_3d

    def transform_to_world(self, points_3d):
        """Apply 4x4 affine transformation to bring into Unity/world coordinates."""
        transformed = []
        for x, y, z in points_3d:
            vec = np.array([x, y, z, 1.0])
            vec_world = self.transformation_matrix @ vec
            transformed.append(vec_world[:3])
        return transformed

    def stereo_remap(self, left_img, right_img):
        """Apply stereo rectification remapping to left/right images."""
        map1x, map1y = cv2.initUndistortRectifyMap(
            self.K1, self.D1, self.R1, self.P1, self.image_size, cv2.CV_32FC1
        )
        map2x, map2y = cv2.initUndistortRectifyMap(
            self.K2, self.D2, self.R2, self.P2, self.image_size, cv2.CV_32FC1
        )
        left_rect = cv2.remap(left_img, map1x, map1y, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_img, map2x, map2y, cv2.INTER_LINEAR)
        return left_rect, right_rect

    def is_valid_point(self, point):
        """Basic check for invalid triangulated point."""
        x, y, z = point
        return 0.0 < z < 200.0  # Valid depth range