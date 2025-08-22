import cv2
import numpy as np
import socket
import os
import json
import struct
import time
from ultralytics import YOLO
from collections import deque
import itertools

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Enable OpenEXR support in OpenCV

# Constants
CHUNK_SIZE = 30000
UNITY_LEFT_PORT = 9998
UNITY_RIGHT_PORT = 9999
OUTPUT_PORT = 9989
OUTPUT_IP = "127.0.0.1"

# Initialize YOLO model and calibration data
model = YOLO("C:\\Users\\kiansadat\\Desktop\\Azimi Project\\3D Eye Gaze\\Assets\\Scripts\\Peg-Detection-Scripts\\Training-05-22\\result\\content\\runs\\detect\\yolo8_peg_detector\\weights\\best.pt")
calib = np.load("C:\\Users\\kiansadat\\Desktop\\Azimi Project\\3D Eye Gaze\\Assets\\Scripts\\Camera Calibration\\stereo_camera_calibration2.npz")
transformation_matrix = np.load("C:\\Users\\kiansadat\\Desktop\\Azimi Project\\3D Eye Gaze\\Assets\\Scripts\\Camera Calibration\\TransformationMatrix.npz")["affine_transform_ransac"]

# Camera parameters
K1, D1 = calib["cameraMatrixL"], calib["distL"]
K2, D2 = calib["cameraMatrixR"], calib["distR"]
R, T = calib["R"], calib["T"]
baseline_offset = T[0][0] / 2.0
image_size = (800, 600)
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T, alpha=0, flags=cv2.CALIB_ZERO_DISPARITY)

# Initialize sockets
output_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
left_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
right_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

# ---------- helpers: low-latency smoothing + 3D constant-velocity KF ----------
class OneEuro3D:
    def __init__(self, freq=120.0, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.freq=freq; self.min_cutoff=min_cutoff; self.beta=beta; self.d_cutoff=d_cutoff
        self.t_prev=None; self.x_prev=None; self.dx_prev=None
    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0/(2*np.pi*cutoff)
        return 1.0/(1.0 + tau/dt)
    def filter(self, x, t):
        x=np.asarray(x, dtype=float)
        if self.t_prev is None:
            self.t_prev=t; self.x_prev=x.copy(); self.dx_prev=np.zeros_like(x); return x
        dt=max(1e-6, t-self.t_prev); self.t_prev=t
        dx=(x-self.x_prev)/dt
        alpha_d=self._alpha(self.d_cutoff, dt)
        dx_hat=alpha_d*dx + (1-alpha_d)*self.dx_prev
        cutoff=self.min_cutoff + self.beta*np.linalg.norm(dx_hat)
        alpha=self._alpha(cutoff, dt)
        x_hat=alpha*x + (1-alpha)*self.x_prev
        self.x_prev=x_hat; self.dx_prev=dx_hat
        return x_hat

class Kalman3D:
    """6D state [x y z vx vy vz] with constant-velocity dynamics."""
    def __init__(self, q_pos=1e-4, q_vel=1e-3, r_meas=2e-4):
        self.x=np.zeros((6,1))
        self.P=np.eye(6)*1e-3
        self.Q=np.diag([q_pos,q_pos,q_pos,q_vel,q_vel,q_vel])
        self.R=np.eye(3)*r_meas
        self.F=np.eye(6)
        self.H=np.zeros((3,6)); self.H[0,0]=self.H[1,1]=self.H[2,2]=1.0
    def predict(self, dt):
        self.F=np.eye(6); self.F[0,3]=self.F[1,4]=self.F[2,5]=dt
        self.x=self.F@self.x
        self.P=self.F@self.P@self.F.T + self.Q
    def update(self, z):
        z=np.asarray(z, dtype=float).reshape(3,1)
        y=z - self.H@self.x
        S=self.H@self.P@self.H.T + self.R
        K=self.P@self.H.T@np.linalg.inv(S)
        self.x=self.x + K@y
        self.P=(np.eye(6) - K@self.H)@self.P
    def pos(self): return self.x[:3,0].copy()
    def vel(self): return self.x[3:,0].copy()

# --------------------------- DROP-IN REPLACEMENT ---------------------------
class PegStereoMatcher:
    """
    Fixes for your tracker:
    - Initializes deterministically (left->right) when 6 pegs are visible.
    - Per-peg 3D Kalman + OneEuro smoothing.
    - Gated association: all static pegs use a tight gate; ONE moving token uses a wide gate.
    - When a static peg is occluded, we keep publishing its last stable pose (no swapping).
    - 2-frame publish guard to kill one-frame pops.
    """
    def __init__(self):
        self.max_pegs = 6

        # existing knobs you had; I'll reuse them for scale
        self.movement_threshold = 2.5   # "mm" scale used elsewhere in your code
        self.position_history_length = 20
        self.max_missing_frames = 30

        # new, derived thresholds (tuned for your scale)
        self.gate_static = self.movement_threshold * 5.0    # ~15
        self.gate_moving = self.movement_threshold * 25.0   # ~50
        self.publish_guard_radius = 10.0                     # mm-ish
        self.speed_on  = 0.020   # m/s-ish in your units (we only compare relative)
        self.speed_off = 0.010
        self.token_on_frames  = 3
        self.token_off_frames = 6

        # state
        self.pegs = {}                 # id -> dict(track state)
        self.stable_positions = {}     # id -> last "settled" pos
        self.initialized = False
        self.next_peg_id = 0
        self.frame_count = 0
        self.moving_peg_id = None
        self.last_moving_peg_change = 0
        self._t_prev = time.perf_counter()

    # --------------------------- detection helpers -------------------------
    def get_center(self, results):
        if len(results.boxes) == 0:
            return []
        out=[]
        for box in results.boxes:
            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
            out.append([float((x1+x2)/2.0), float((y1+y2)/2.0)])
        return out

    def find_3d_points(self, left_points, right_points):
        pts=[]
        for pt_left, pt_right in zip(left_points, right_points):
            pl = np.array(pt_left).reshape(2,1)
            pr = np.array(pt_right).reshape(2,1)
            X4 = cv2.triangulatePoints(P1, P2, pl, pr)
            X3 = X4[:3]/(X4[3]+1e-12)
            x,y,z = X3[:,0]
            pts.append((x + baseline_offset, -y, z))
        return pts

    def find_pairs(self, left_points, right_points):
        if not left_points or not right_points:
            return [], []
        # Try exact match when we have 6 & 6 (like your code)
        if len(left_points)==6 and len(right_points)==6:
            all_pairings = [list(zip(left_points, perm)) for perm in itertools.permutations(right_points)]
            for pairings in all_pairings:
                valid=True; L=[]; R=[]
                for lp, rp in pairings:
                    if abs(lp[1]-rp[1]) > 20.0:   # relaxed a bit
                        valid=False; break
                    L.append(lp); R.append(rp)
                if valid:
                    pts3d=self.find_3d_points(L,R)
                    if all(0 <= p[2] <= 200 for p in pts3d):
                        return L,R
        # fallback greedy
        return self.find_pairs_simple(left_points, right_points)

    def find_pairs_simple(self, left_points, right_points):
        usedL=set(); usedR=set(); L=[]; R=[]
        for i in range(len(left_points)):
            for j in range(len(right_points)):
                if i in usedL or j in usedR: continue
                lp = left_points[i]; rp = right_points[j]
                if abs(lp[1]-rp[1]) > 20.0:   # relaxed
                    continue
                p3 = self.find_3d_points([lp],[rp])[0]
                if 0 <= p3[2] <= 200:
                    usedL.add(i); usedR.add(j); L.append(lp); R.append(rp)
        return L,R

    # --------------------------- core tracking ------------------------------
    def _init_tracks(self, points_3d):
        """Initialize 6 IDs deterministically left->right when we first see 6 pegs."""
        if len(points_3d) < self.max_pegs:
            return False
        points_3d = sorted(points_3d, key=lambda p: p[0])[:self.max_pegs]
        self.pegs = {}
        tnow = time.perf_counter()
        for pid, pos in enumerate(points_3d):
            kf = Kalman3D(); kf.x[:3,0] = np.array(pos).reshape(3)
            eu = OneEuro3D(freq=120.0, min_cutoff=1.0, beta=0.02, d_cutoff=1.0)
            self.pegs[pid] = {
                'position': tuple(pos),     # last published (after guard)
                'kf': kf,
                'euro': eu,
                'pub_buffer': deque(maxlen=3),
                'missing_frames': 0,
                'moving_frames': 0,
                'static_frames': 0,
                'last_update_t': tnow
            }
        self.stable_positions = {pid: self.pegs[pid]['position'] for pid in self.pegs}
        self.next_peg_id = self.max_pegs
        self.initialized = True
        self.moving_peg_id = None
        self.last_moving_peg_change = self.frame_count
        return True

    def _predict_all(self, dt):
        for pid, st in self.pegs.items():
            st['kf'].predict(dt)

    def _pick_token(self):
        # speed-based hysteresis
        moving = []
        for pid, st in self.pegs.items():
            speed = float(np.linalg.norm(st['kf'].vel()))
            if speed > self.speed_on:
                st['moving_frames'] += 1; st['static_frames'] = 0
            else:
                st['static_frames'] += 1; st['moving_frames'] = 0
            if st['moving_frames'] >= self.token_on_frames:
                moving.append(pid)

        if self.moving_peg_id is None and len(moving) == 1 and (self.frame_count - self.last_moving_peg_change) > self.token_on_frames:
            self.moving_peg_id = moving[0]
            self.last_moving_peg_change = self.frame_count

        if self.moving_peg_id is not None:
            st = self.pegs[self.moving_peg_id]
            if st['static_frames'] >= self.token_off_frames and np.linalg.norm(st['kf'].vel()) < self.speed_off:
                # peg has settled; update stable pos and drop token
                self.stable_positions[self.moving_peg_id] = tuple(st['kf'].pos())
                self.moving_peg_id = None
                self.last_moving_peg_change = self.frame_count

    def _associate(self, meas):
        """
        Associate measurements to tracks.
        - Static pegs: tight gate to their predicted pos.
        - Moving token (if any): wide gate.
        Unique assignment enforced.
        """
        assigned = {}
        used_meas = set()

        # 1) Static pegs first (prevents swaps)
        cand_list = []
        for pid, st in self.pegs.items():
            if pid == self.moving_peg_id: 
                continue
            p_pred = st['kf'].pos()
            for j, m in enumerate(meas):
                d = float(np.linalg.norm(p_pred - np.array(m)))
                if d <= self.gate_static:
                    cand_list.append((d, pid, j))
        cand_list.sort(key=lambda t: t[0])
        taken_pid = set()
        for d, pid, j in cand_list:
            if pid in taken_pid or j in used_meas: 
                continue
            assigned[pid] = j
            taken_pid.add(pid)
            used_meas.add(j)

        # 2) Moving token (wider gate)
        if self.moving_peg_id is not None:
            pid = self.moving_peg_id
            st = self.pegs[pid]
            p_pred = st['kf'].pos()
            best = None; bestd = 1e9; bestj = None
            for j, m in enumerate(meas):
                if j in used_meas: 
                    continue
                d = float(np.linalg.norm(p_pred - np.array(m)))
                if d < bestd and d <= self.gate_moving:
                    bestd = d; best = m; bestj = j
            if bestj is not None:
                assigned[pid] = bestj
                used_meas.add(bestj)

        return assigned

    def _update_tracks(self, assigned, meas, tnow):
        for pid, st in self.pegs.items():
            if pid in assigned:
                z = np.array(meas[assigned[pid]])
                st['kf'].update(z)
                st['missing_frames'] = 0
            else:
                st['missing_frames'] += 1
                # If static and occluded, freeze at last published (prevents drift/glitch)
                if pid != self.moving_peg_id and st['missing_frames'] <= self.max_missing_frames:
                    z = np.array(st['position'])
                    st['kf'].update(z.reshape(3))
                # if moving and occluded, KF prediction carries it

            # publish with smoothing + two-frame guard
            sm = st['euro'].filter(st['kf'].pos(), tnow)
            st['pub_buffer'].append(sm)
            if len(st['pub_buffer']) >= 2:
                a, b = st['pub_buffer'][-2], st['pub_buffer'][-1]
                if np.linalg.norm(a - b) <= self.publish_guard_radius:
                    st['position'] = tuple(b)
            else:
                st['position'] = tuple(sm)

    # ------------------------------- public --------------------------------
    def update(self, left_frame, right_frame):
        # 1) detections (lowered conf a bit; small objects)
        res_l = model(left_frame,  verbose=False, conf=0.10)[0]
        res_r = model(right_frame, verbose=False, conf=0.10)[0]
        left_pegs  = self.get_center(res_l)
        right_pegs = self.get_center(res_r)

        # 2) stereo pairing & triangulation
        matched_left, matched_right = self.find_pairs(left_pegs, right_pegs)
        points_3d = self.find_3d_points(matched_left, matched_right) if (matched_left and matched_right) else []

        # 3) init if needed (deterministic)
        if not self.initialized:
            if self._init_tracks(points_3d):
                return self._get_current_positions()  # immediately publish initialized
            else:
                return self._get_current_positions()  # may be empty until 6 seen

        # 4) predict → associate → update
        tnow = time.perf_counter()
        dt = max(1e-3, tnow - self._t_prev)
        self._t_prev = tnow

        self._predict_all(dt)
        self._pick_token()
        assigned = self._associate(points_3d)
        self._update_tracks(assigned, points_3d, tnow)

        self.frame_count += 1
        return self._get_current_positions()

    def _get_current_positions(self):
        return {pid: tuple(map(float, st['position'])) for pid, st in self.pegs.items()}

def transform_pegs(pegs):
    """Transform 3D points from camera coordinates to world coordinates."""
    transformed_pegs = []
    for x, y, z in pegs:
        p = np.array([x, y, z, 1.0])
        p_transformed = transformation_matrix @ p
        transformed_pegs.append(p_transformed[:3])
    return transformed_pegs

def send_frame_left(frame, frame_id=0):
    _, buffer = cv2.imencode(".jpg", frame)
    data = buffer.tobytes()
    total_chunks = (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    header = struct.pack('!II', frame_id, total_chunks)
    left_sock.sendto(b'H' + header, (OUTPUT_IP, UNITY_LEFT_PORT))
    
    for i in range(total_chunks):
        chunk_data = data[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]
        chunk_header = struct.pack('!II', frame_id, i)
        left_sock.sendto(b'D' + chunk_header + chunk_data, (OUTPUT_IP, UNITY_LEFT_PORT))

def send_frame_right(frame, frame_id=0):
    _, buffer = cv2.imencode(".jpg", frame)
    data = buffer.tobytes()
    total_chunks = (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    header = struct.pack('!II', frame_id, total_chunks)
    right_sock.sendto(b'H' + header, (OUTPUT_IP, UNITY_RIGHT_PORT))
    
    for i in range(total_chunks):
        chunk_data = data[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]
        chunk_header = struct.pack('!II', frame_id, i)
        right_sock.sendto(b'D' + chunk_header + chunk_data, (OUTPUT_IP, UNITY_RIGHT_PORT))

if __name__ == "__main__":
    try:
        # Initialize rectification maps
        map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)
        
        # Initialize peg tracker
        peg_tracker = PegStereoMatcher()
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Split and rectify frames
            left_frame = frame[:, :800]
            right_frame = frame[:, 800:]
            
            left_frame = cv2.remap(left_frame, map1x, map1y, cv2.INTER_LINEAR)
            right_frame = cv2.remap(right_frame, map2x, map2y, cv2.INTER_LINEAR)
            
            # Send frames to Unity
            send_frame_left(left_frame, frame_count)
            send_frame_right(right_frame, frame_count)
            
            # Update peg tracking
            peg_positions = peg_tracker.update(left_frame, right_frame)
            
            # Prepare message for Unity - always send 6 pegs
            message = {}
            
            # First, get all detected pegs with their current positions
            current_pegs = {}
            for peg_id, pos in peg_positions.items():
                if 0 <= peg_id < 6:  # Only include pegs with valid IDs (0-5)
                    current_pegs[peg_id] = pos
            
            # Fill in any missing pegs with their last known positions
            for peg_id in range(6):
                if peg_id in peg_positions:
                    x, y, z = peg_positions[peg_id]
                else:
                    # If peg not detected, use last known position or default to origin
                    if peg_id in peg_tracker.pegs:
                        x, y, z = peg_tracker.pegs[peg_id]['position']
                    else:
                        x, y, z = 0, 0, 0
                
                # Transform to world coordinates
                try:
                    transformed_pos = transform_pegs([(x, y, z)])[0]
                    message[str(peg_id + 1)] = {
                        "x": round(float(transformed_pos[0]), 3),
                        "y": round(float(transformed_pos[1]), 3),
                        "z": round(float(transformed_pos[2]), 3)
                    }
                except Exception as e:
                    print(f"Error transforming peg {peg_id}: {e}")
                    # Send zero position if transformation fails
                    message[str(peg_id + 1)] = {"x": 0, "y": 0, "z": 0}
            
            # Send message to Unity
            if message:
                try:
                    message_json = json.dumps(message)
                    output_socket.sendto(message_json.encode('utf-8'), (OUTPUT_IP, OUTPUT_PORT))
                except Exception as e:
                    print(f"Error sending message: {e}")
            
            frame_count += 1
            
            # Add small delay to prevent overwhelming the system
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cap.release()
        output_socket.close()
        left_sock.close()
        right_sock.close()
        cv2.destroyAllWindows()
