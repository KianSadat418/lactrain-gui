import cv2
import numpy as np
import socket
import os
import json
import struct
import time
from ultralytics import YOLO
from collections import deque
from scipy.optimize import linear_sum_assignment

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Enable OpenEXR support in OpenCV

# =====================
# Constants / IO
# =====================
CHUNK_SIZE = 30000
UNITY_LEFT_PORT = 9998
UNITY_RIGHT_PORT = 9999
OUTPUT_PORT = 9989
OUTPUT_IP = "127.0.0.1"

# =====================
# Load model and calib
# =====================
model = YOLO(r"C:\\Users\\kiansadat\\Desktop\\Azimi Project\\3D Eye Gaze\\Assets\\Scripts\\Peg-Detection-Scripts\\Training-05-22\\result\\content\\runs\\detect\\yolo8_peg_detector\\weights\\best.pt")
calib = np.load(r"C:\\Users\\kiansadat\\Desktop\\Azimi Project\\3D Eye Gaze\\Assets\\Scripts\\Camera Calibration\\stereo_camera_calibration2.npz")
transformation_matrix = np.load(r"C:\\Users\\kiansadat\\Desktop\\Azimi Project\\3D Eye Gaze\\Assets\\Scripts\\Camera Calibration\\TransformationMatrix.npz")["affine_transform_ransac"]

# Camera parameters
K1, D1 = calib["cameraMatrixL"], calib["distL"]
K2, D2 = calib["cameraMatrixR"], calib["distR"]
R, T = calib["R"], calib["T"]
baseline_offset = T[0][0] / 2.0
image_size = (800, 600)
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T, alpha=0, flags=cv2.CALIB_ZERO_DISPARITY)

# =====================
# UDP sockets
# =====================
output_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
left_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
right_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# =====================
# Video capture
# =====================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

# =====================
# Utilities
# =====================

def send_frame(sock, frame, port, frame_id=0):
    _, buffer = cv2.imencode(".jpg", frame)
    data = buffer.tobytes()
    total_chunks = (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE
    header = struct.pack('!II', frame_id, total_chunks)
    sock.sendto(b'H' + header, (OUTPUT_IP, port))
    for i in range(total_chunks):
        chunk_data = data[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]
        chunk_header = struct.pack('!II', frame_id, i)
        sock.sendto(b'D' + chunk_header + chunk_data, (OUTPUT_IP, port))


def transform_pegs_world(points_cam3d):
    out = []
    for x, y, z in points_cam3d:
        p = np.array([x, y, z, 1.0])
        p_w = transformation_matrix @ p
        out.append(p_w[:3])
    return out

# =====================
# Stereo pairing (robust, epipolar gating)
# =====================

def yolo_centers(result, conf_th=0.35, max_dets=20):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    centers = []
    for i in range(min(len(boxes), max_dets)):
        c = float(boxes.conf[i].item()) if hasattr(boxes, 'conf') else 1.0
        if c < conf_th:
            continue
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
        centers.append(((x1+x2)/2.0, (y1+y2)/2.0))
    return centers


def triangulate_points(P1_, P2_, pairs):
    """pairs: list of (ptL(x,y), ptR(x,y))"""
    pts3d = []
    for (xl, yl), (xr, yr) in pairs:
        pl = np.array([xl, yl], dtype=np.float32).reshape(2, 1)
        pr = np.array([xr, yr], dtype=np.float32).reshape(2, 1)
        X_h = cv2.triangulatePoints(P1_, P2_, pl, pr)
        X = (X_h[:3] / X_h[3]).reshape(3)
        pts3d.append((float(X[0] + baseline_offset), float(-X[1]), float(X[2])))
    return pts3d


def pair_stereo(left_pts, right_pts, y_tol=6.0, z_min=0.0, z_max=220.0):
    """Return 3D points from robust bipartite y-matching with gating on y and depth."""
    if len(left_pts) == 0 or len(right_pts) == 0:
        return []
    # Build cost matrix on |dy|
    L, Rn = len(left_pts), len(right_pts)
    cost = np.full((L, Rn), 1e6, dtype=np.float32)
    for i, (xl, yl) in enumerate(left_pts):
        for j, (xr, yr) in enumerate(right_pts):
            dy = abs(yl - yr)
            if dy <= y_tol and (xr - xl) < 0:  # disparity must be positive if cameras are rectified L|R (assumes R image is to the right)
                cost[i, j] = dy
    row_ind, col_ind = linear_sum_assignment(cost)
    pairs = []
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] >= 1e6:
            continue
        pairs.append((left_pts[i], right_pts[j]))
    # Triangulate & depth gate
    pts3d = triangulate_points(P1, P2, pairs)
    pts_valid = []
    for p in pts3d:
        if z_min <= p[2] <= z_max:
            pts_valid.append(p)
    return pts_valid

# =====================
# 3D Kalman Track
# =====================

class KFTrack:
    def __init__(self, track_id, x0, init_vel=np.zeros(3), pos_var=4.0, vel_var=1.0, meas_var=9.0):
        self.id = track_id
        self.dim_x = 6
        self.dim_z = 3
        # state [x y z vx vy vz]
        self.x = np.zeros((6, 1), dtype=np.float32)
        self.x[:3, 0] = np.array(x0, dtype=np.float32)
        self.x[3:, 0] = np.array(init_vel, dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 50.0  # large init uncertainty
        self.F = np.eye(6, dtype=np.float32)
        self.H = np.zeros((3, 6), dtype=np.float32)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0
        self.Q_base = np.diag([pos_var, pos_var, pos_var, vel_var, vel_var, vel_var]).astype(np.float32)
        self.R = np.eye(3, dtype=np.float32) * meas_var
        self.history = deque(maxlen=20)
        self.missing = 0
        self.is_moving = False

    def predict(self, dt, q_scale=1.0):
        # Update F for dt
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt
        self.x = (self.F @ self.x).astype(np.float32)
        Q = (self.Q_base * float(q_scale)).astype(np.float32)
        self.P = (self.F @ self.P @ self.F.T + Q).astype(np.float32)
        return self.x[:3, 0].copy()

    def update(self, z):
        z = np.array(z, dtype=np.float32).reshape(3, 1)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        I = np.eye(6, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        self.missing = 0
        self.history.append(self.x[:3, 0].copy())

    def mahalanobis(self, z):
        z = np.array(z, dtype=np.float32).reshape(3, 1)
        S = self.H @ self.P @ self.H.T + self.R
        y = z - self.H @ self.x
        return float((y.T @ np.linalg.inv(S) @ y)[0, 0])

    @property
    def pos(self):
        return self.x[:3, 0].copy()

    @property
    def vel(self):
        return self.x[3:, 0].copy()

# =====================
# Peg tracker using 3D KF + GNN assignment + occlusion handling
# =====================

class Peg3DTrackerKF:
    def __init__(self, max_pegs=6):
        self.max_pegs = max_pegs
        self.tracks = {}
        self.initialized = False
        self.next_id = 0
        self.last_ts = None
        self.missed_threshold = 150  # frames allowed missing (occlusion tolerance)
        self.mahal_gate = 14.0       # ~chi2_3(0.999)
        self.stationary_q = 0.25      # process noise scale when not moving
        self.moving_q = 8.0          # higher process noise for moving peg
        self.speed_thresh = 4.0      # mm/frame, tune per FPS
        self.moving_id = None

    # -----------------
    # Stereo & 3D input
    # -----------------
    def yolo_stereo_to_3d(self, left_frame, right_frame):
        res_l = model(left_frame, verbose=False)[0]
        res_r = model(right_frame, verbose=False)[0]
        L = yolo_centers(res_l)
        R = yolo_centers(res_r)
        pts3d = pair_stereo(L, R)
        return pts3d

    # -----------------
    # Public update
    # -----------------
    def update(self, left_frame, right_frame):
        now = time.time()
        if self.last_ts is None:
            dt = 1/30.0
        else:
            dt = max(1/120.0, min(1/15.0, now - self.last_ts))  # clamp for stability
        self.last_ts = now

        detections = self.yolo_stereo_to_3d(left_frame, right_frame)

        # 1) If we have fewer than max tracks, try to bootstrap
        if not self.initialized and len(detections) >= self.max_pegs:
            # choose the best 6 (closest to median depth) for stability
            Z = np.array([d[2] for d in detections])
            med = np.median(Z)
            idx = np.argsort(np.abs(Z - med))[:self.max_pegs]
            init_pts = [detections[i] for i in idx]
            # deterministic ID assignment: left-to-right by X
            init_pts_sorted = sorted(init_pts, key=lambda p: p[0])
            for k, p in enumerate(init_pts_sorted):
                tr = KFTrack(k, p)
                self.tracks[k] = tr
            self.next_id = self.max_pegs
            self.initialized = True
            self.moving_id = None

        # Predict all tracks
        for tid, tr in self.tracks.items():
            q = self.moving_q if (self.moving_id == tid) else self.stationary_q
            tr.predict(dt, q_scale=q)

        # Early out: if no detections, just return predicted
        if len(detections) == 0:
            for tr in self.tracks.values():
                tr.missing += 1
            return {tid: tr.pos for tid, tr in self.tracks.items()}

        # 2) Build cost matrix (tracks x detections) with gating
        track_ids = list(self.tracks.keys())
        if len(track_ids) == 0 and self.initialized:
            # shouldn't happen, but guard
            return {}

        cost = np.full((len(track_ids), len(detections)), 1e6, dtype=np.float32)
        for i, tid in enumerate(track_ids):
            tr = self.tracks[tid]
            for j, det in enumerate(detections):
                m = tr.mahalanobis(det)
                if m <= self.mahal_gate:
                    cost[i, j] = m

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost)
        assigned_tr = set()
        assigned_det = set()

        # 3) Update matched tracks
        for ri, ci in zip(row_ind, col_ind):
            if cost[ri, ci] < 1e6 and cost[ri, ci] <= self.mahal_gate:
                tid = track_ids[ri]
                self.tracks[tid].update(detections[ci])
                assigned_tr.add(tid)
                assigned_det.add(ci)

        # 4) Unmatched tracks: keep predicting, increment missing
        for tid in track_ids:
            if tid not in assigned_tr:
                self.tracks[tid].missing += 1

        # 5) (Optional) If not fully initialized yet, create new tracks up to 6
        if not self.initialized:
            for j, det in enumerate(detections):
                if j in assigned_det:
                    continue
                tid = self.next_id
                tr = KFTrack(tid, det)
                self.tracks[tid] = tr
                self.next_id += 1
                if len(self.tracks) >= self.max_pegs:
                    # lock to exactly 6 by picking most reliable (lowest P trace)
                    if len(self.tracks) > self.max_pegs:
                        # drop worst
                        srt = sorted(self.tracks.items(), key=lambda kv: np.trace(kv[1].P))
                        keep = dict(srt[:self.max_pegs])
                        self.tracks = keep
                        self.next_id = max(self.tracks.keys()) + 1
                    self.initialized = True
                    break

        # 6) Never re-ID: we do NOT create new IDs after initialization; ignore extra detections

        # 7) Determine moving peg based on speed (one-at-a-time prior)
        if self.initialized and len(self.tracks) == self.max_pegs:
            speeds = {tid: float(np.linalg.norm(tr.vel)) for tid, tr in self.tracks.items()}
            if len(speeds) > 0:
                cand_id = max(speeds, key=lambda k: speeds[k])
                cand_speed = speeds[cand_id]
                others = [v for k, v in speeds.items() if k != cand_id]
                second = max(others) if others else 0.0
                if cand_speed > self.speed_thresh and cand_speed > 2.0 * second:
                    self.moving_id = cand_id
                elif second > 0 and cand_speed <= 1.5 * second:
                    # nobody clearly moving
                    self.moving_id = None

        # 8) Prune nothing: keep tracks alive for long occlusions
        # If a track has been missing way too long, we still DO NOT delete it; we keep predicting to preserve ID stability.
        # (If you truly need pruning, lower self.missed_threshold and re-init carefully.)

        return {tid: tr.pos for tid, tr in self.tracks.items()}

# =====================
# Main loop
# =====================

if __name__ == "__main__":
    try:
        # Init rectification maps
        map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

        tracker = Peg3DTrackerKF(max_pegs=6)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Split and rectify
            left_raw = frame[:, :800]
            right_raw = frame[:, 800:]
            left_frame = cv2.remap(left_raw, map1x, map1y, cv2.INTER_LINEAR)
            right_frame = cv2.remap(right_raw, map2x, map2y, cv2.INTER_LINEAR)

            # Send frames to Unity
            send_frame(left_sock, left_frame, UNITY_LEFT_PORT, frame_count)
            send_frame(right_sock, right_frame, UNITY_RIGHT_PORT, frame_count)

            # Update tracker
            peg_positions_cam = tracker.update(left_frame, right_frame)

            # Assemble message (always 6 IDs 1..6). If not yet initialized or missing, use predicted.
            message = {}
            for peg_id in range(6):
                if peg_id in peg_positions_cam:
                    x, y, z = peg_positions_cam[peg_id]
                else:
                    # If track not yet created, default to origin (rare after init)
                    x, y, z = 0.0, 0.0, 0.0
                # Transform to world
                try:
                    Xw, Yw, Zw = transform_pegs_world([(x, y, z)])[0]
                    message[str(peg_id + 1)] = {
                        "x": round(float(Xw), 3),
                        "y": round(float(Yw), 3),
                        "z": round(float(Zw), 3)
                    }
                except Exception:
                    message[str(peg_id + 1)] = {"x": 0.0, "y": 0.0, "z": 0.0}

            # Send to Unity
            try:
                output_socket.sendto(json.dumps(message).encode('utf-8'), (OUTPUT_IP, OUTPUT_PORT))
            except Exception as e:
                print(f"Error sending message: {e}")

            frame_count += 1
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cap.release()
        output_socket.close()
        left_sock.close()
        right_sock.close()
        cv2.destroyAllWindows()
