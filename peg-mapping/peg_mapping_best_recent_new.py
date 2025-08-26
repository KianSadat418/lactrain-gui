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
# No scipy imports needed

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ============================== CONSTANTS ==============================
CHUNK_SIZE = 30000
UNITY_LEFT_PORT = 9998
UNITY_RIGHT_PORT = 9999
OUTPUT_PORT = 9989
OUTPUT_IP = "127.0.0.1"
SHOW_DETECTIONS = False

# ============================== IO / CALIBRATION ==============================
model = YOLO(r"C:\Users\kiansadat\Desktop\Azimi Project\3D Eye Gaze\Assets\Scripts\Peg-Detection-Scripts\Training-05-22\result\content\runs\detect\yolo8_peg_detector\weights\best.pt")
calib = np.load(r"C:\Users\kiansadat\Desktop\Azimi Project\3D Eye Gaze\Assets\Scripts\Camera Calibration\stereo_camera_calibration2.npz")
transformation_matrix = np.load(r"C:\Users\kiansadat\Desktop\Azimi Project\3D Eye Gaze\Assets\Scripts\Camera Calibration\TransformationMatrix.npz")["affine_transform_ransac"]

K1, D1 = calib["cameraMatrixL"], calib["distL"]
K2, D2 = calib["cameraMatrixR"], calib["distR"]
R, T = calib["R"], calib["T"]
baseline_offset = T[0][0] / 2.0
image_size = (800, 600)
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T, alpha=0, flags=cv2.CALIB_ZERO_DISPARITY)

output_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
left_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
right_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

# ============================== ADAPTIVE FILTERING ==============================

class AdaptiveOneEuro3D:
    """OneEuro filter that adapts to motion state"""
    def __init__(self, freq=120.0):
        self.freq = freq
        self.t_prev = None
        self.x_prev = None
        self.dx_prev = None
        self.speed_history = deque(maxlen=5)
        
    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0/(2*np.pi*cutoff)
        return 1.0/(1.0 + tau/dt)
    
    def filter(self, x, t, is_moving=False):
        x = np.asarray(x, dtype=float)
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x
            
        dt = max(1e-6, t - self.t_prev)
        self.t_prev = t
        
        # Calculate velocity
        dx = (x - self.x_prev)/dt
        alpha_d = self._alpha(1.0, dt)
        dx_hat = alpha_d*dx + (1-alpha_d)*self.dx_prev
        
        # Track speed
        speed = np.linalg.norm(dx_hat)
        self.speed_history.append(speed)
        avg_speed = np.mean(self.speed_history) if self.speed_history else speed
        
        # ADAPTIVE PARAMETERS based on explicit movement state
        if is_moving:
            # When peg is actively moving - minimal filtering
            min_cutoff = 10.0
            beta = 0.5
        else:
            # When static - moderate filtering for stability
            min_cutoff = 1.0
            beta = 0.02
        
        cutoff = min_cutoff + beta * speed
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha*x + (1-alpha)*self.x_prev
        
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


class AdaptiveKalman3D:
    """Kalman filter with adaptive noise parameters"""
    def __init__(self):
        self.x = np.zeros((6,1))  # [x y z vx vy vz]
        self.P = np.eye(6) * 1e-3
        self.F = np.eye(6)
        self.H = np.zeros((3,6))
        self.H[0,0] = self.H[1,1] = self.H[2,2] = 1.0
        
        # Base noise parameters
        self.Q_base = np.diag([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3])
        self.R_base = np.eye(3) * 2e-4
        
        self.last_measurement = None
        
    def predict(self, dt, is_moving=False):
        self.F = np.eye(6)
        self.F[0,3] = self.F[1,4] = self.F[2,5] = dt
        
        # Scale process noise based on movement
        if is_moving:
            Q = self.Q_base * 25.0  # Higher uncertainty when moving
        else:
            Q = self.Q_base * 0.5   # Lower uncertainty when static
            
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q
        
    def update(self, z):
        z = np.asarray(z, dtype=float).reshape(3,1)
        
        # Adaptive measurement noise based on consistency
        if self.last_measurement is not None:
            meas_change = np.linalg.norm(z - self.last_measurement)
            if meas_change < 2.0:
                R = self.R_base * 0.5  # Trust consistent measurements more
            elif meas_change < 10.0:
                R = self.R_base
            else:
                R = self.R_base * 2.0  # Trust less during jumps
        else:
            R = self.R_base
            
        self.last_measurement = z.copy()
        
        # Standard Kalman update
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        
    def pos(self): 
        return self.x[:3, 0].copy()
    
    def vel(self): 
        return self.x[3:, 0].copy()


class HybridPegTracker:
    """
    Hybrid tracker combining:
    - Fast response from original code
    - Occlusion handling via Hungarian algorithm
    - Single moving peg constraint
    - Adaptive filtering based on movement state
    """
    def __init__(self):
        self.max_pegs = 6
        self.pegs = {}
        self.stable_positions = {}
        self.initialized = False
        self.next_peg_id = 0
        self.frame_count = 0
        self._t_prev = time.perf_counter()
        
        # Movement detection (from original)
        self.speed_on = 0.030   # Speed to start moving
        self.speed_off = 0.010  # Speed to stop moving
        self.token_on_frames = 1
        self.token_off_frames = 6
        
        # Association gates
        self.base_gate_radius = 15.0
        self.moving_gate_multiplier = 3.0
        self.max_missing_frames = 30
        
        # Tracking state
        self.moving_peg_id = None
        self.last_moving_peg_change = 0
        
    # --------------------------- Detection helpers -------------------------
    def get_center(self, results):
        if results is None or len(results.boxes) == 0:
            return []
        out = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            out.append([float((x1+x2)/2.0), float((y1+y2)/2.0)])
        return out
    
    def find_3d_points(self, left_points, right_points):
        pts = []
        for pt_left, pt_right in zip(left_points, right_points):
            pl = np.array(pt_left).reshape(2,1)
            pr = np.array(pt_right).reshape(2,1)
            X4 = cv2.triangulatePoints(P1, P2, pl, pr)
            X3 = X4[:3]/(X4[3]+1e-12)
            x, y, z = X3[:, 0]
            pts.append((x + baseline_offset, -y, z))
        return pts
    
    def find_pairs(self, left_points, right_points):
        if not left_points or not right_points:
            return [], []
            
        # Try exact match when we have 6 & 6
        if len(left_points) == 6 and len(right_points) == 6:
            all_pairings = [list(zip(left_points, perm)) 
                          for perm in itertools.permutations(right_points)]
            for pairings in all_pairings:
                valid = True
                L, R = [], []
                for lp, rp in pairings:
                    if abs(lp[1] - rp[1]) > 25.0:
                        valid = False
                        break
                    L.append(lp)
                    R.append(rp)
                if valid:
                    pts3d = self.find_3d_points(L, R)
                    if all(0 <= p[2] <= 200 for p in pts3d):
                        return L, R
                        
        # Fallback to greedy matching
        return self.find_pairs_simple(left_points, right_points)
    
    def find_pairs_simple(self, left_points, right_points):
        usedL, usedR = set(), set()
        L, R = [], []
        for i in range(len(left_points)):
            for j in range(len(right_points)):
                if i in usedL or j in usedR:
                    continue
                lp = left_points[i]
                rp = right_points[j]
                if abs(lp[1] - rp[1]) > 25.0:
                    continue
                p3 = self.find_3d_points([lp], [rp])[0]
                if 0 <= p3[2] <= 200:
                    usedL.add(i)
                    usedR.add(j)
                    L.append(lp)
                    R.append(rp)
        return L, R
    
    # --------------------------- Core tracking ------------------------------
    def _init_tracks(self, points_3d):
        """Initialize tracks when we first see 6 pegs"""
        if len(points_3d) < self.max_pegs:
            return False
            
        # Sort by x-coordinate for consistent ID assignment
        points_3d = sorted(points_3d, key=lambda p: p[0])[:self.max_pegs]
        self.pegs = {}
        tnow = time.perf_counter()
        
        for pid, pos in enumerate(points_3d):
            kf = AdaptiveKalman3D()
            kf.x[:3, 0] = np.array(pos).reshape(3)
            euro = AdaptiveOneEuro3D(freq=120.0)
            
            self.pegs[pid] = {
                'position': tuple(pos),
                'raw_position': tuple(pos),
                'kf': kf,
                'euro': euro,
                'missing_frames': 0,
                'moving_frames': 0,
                'static_frames': 0,
                'last_update_t': tnow,
                'is_moving': False,
                'speed_history': deque(maxlen=10),
            }
            
        self.stable_positions = {pid: self.pegs[pid]['position'] for pid in self.pegs}
        self.next_peg_id = self.max_pegs
        self.initialized = True
        return True
    
    def _detect_moving_peg(self):
        """Detect which single peg is moving using speed-based hysteresis"""
        moving_candidates = []
        
        for pid, st in self.pegs.items():
            speed = float(np.linalg.norm(st['kf'].vel()))
            st['speed_history'].append(speed)
            
            # Hysteresis for movement detection
            if speed > self.speed_on:
                st['moving_frames'] += 1
                st['static_frames'] = 0
            else:
                st['static_frames'] += 1
                st['moving_frames'] = 0
            
            # Check if qualifies as moving
            if st['moving_frames'] >= self.token_on_frames:
                moving_candidates.append(pid)
        
        # Assign moving token to single peg
        if self.moving_peg_id is None:
            if len(moving_candidates) == 1:
                cooldown = self.frame_count - self.last_moving_peg_change
                if cooldown > self.token_on_frames:
                    self.moving_peg_id = moving_candidates[0]
                    self.last_moving_peg_change = self.frame_count
        
        # Check if moving peg has stopped
        if self.moving_peg_id is not None:
            st = self.pegs[self.moving_peg_id]
            if st['static_frames'] >= self.token_off_frames:
                speed = np.linalg.norm(st['kf'].vel())
                if speed < self.speed_off:
                    # Update stable position
                    self.stable_positions[self.moving_peg_id] = st['position']
                    self.moving_peg_id = None
                    self.last_moving_peg_change = self.frame_count
        
        # Update moving flags
        for pid, st in self.pegs.items():
            st['is_moving'] = (pid == self.moving_peg_id)
    
    def _associate_robust(self, measurements):
        """
        Robust association without scipy - prioritizes static pegs first to prevent swaps
        """
        if len(measurements) == 0:
            return {}
        
        assignments = {}
        used_measurements = set()
        
        # Step 1: Associate static pegs first (tighter gate)
        static_associations = []
        for pid, st in self.pegs.items():
            if st['is_moving']:
                continue  # Skip moving peg for now
                
            pred_pos = st['kf'].pos()
            for j, meas in enumerate(measurements):
                if j in used_measurements:
                    continue
                    
                dist = np.linalg.norm(pred_pos - np.array(meas))
                if dist <= self.base_gate_radius:
                    static_associations.append((dist, pid, j))
        
        # Sort by distance and assign greedily
        static_associations.sort(key=lambda x: x[0])
        for dist, pid, j in static_associations:
            if pid not in assignments and j not in used_measurements:
                assignments[pid] = j
                used_measurements.add(j)
        
        # Step 2: Associate moving peg with wider gate
        if self.moving_peg_id is not None and self.moving_peg_id not in assignments:
            st = self.pegs[self.moving_peg_id]
            pred_pos = st['kf'].pos()
            
            best_dist = float('inf')
            best_j = None
            
            for j, meas in enumerate(measurements):
                if j in used_measurements:
                    continue
                    
                dist = np.linalg.norm(pred_pos - np.array(meas))
                max_dist = self.base_gate_radius * self.moving_gate_multiplier
                
                if dist < best_dist and dist <= max_dist:
                    best_dist = dist
                    best_j = j
            
            if best_j is not None:
                assignments[self.moving_peg_id] = best_j
                used_measurements.add(best_j)
        
        # Step 3: Try to associate any remaining unassigned pegs
        for pid, st in self.pegs.items():
            if pid in assignments:
                continue
                
            pred_pos = st['kf'].pos()
            best_dist = float('inf')
            best_j = None
            
            # Use appropriate gate based on state
            if st['is_moving']:
                max_dist = self.base_gate_radius * self.moving_gate_multiplier
            else:
                max_dist = self.base_gate_radius * 1.5  # Slightly wider for occluded static pegs
            
            for j, meas in enumerate(measurements):
                if j in used_measurements:
                    continue
                    
                dist = np.linalg.norm(pred_pos - np.array(meas))
                if dist < best_dist and dist <= max_dist:
                    best_dist = dist
                    best_j = j
            
            if best_j is not None:
                assignments[pid] = best_j
                used_measurements.add(best_j)
        
        return assignments
    
    def _update_tracks(self, assignments, measurements, tnow, dt):
        """Update tracks with adaptive filtering"""
        for pid, st in self.pegs.items():
            # Predict
            st['kf'].predict(dt, is_moving=st['is_moving'])
            
            # Update with measurement if available
            if pid in assignments:
                z = np.array(measurements[assignments[pid]])
                st['kf'].update(z)
                st['raw_position'] = tuple(z)
                st['missing_frames'] = 0
            else:
                st['missing_frames'] += 1
                # For static pegs with brief occlusion, maintain position
                if not st['is_moving'] and st['missing_frames'] <= 5:
                    st['kf'].update(np.array(st['position']))
            
            # Get filtered position
            kf_pos = st['kf'].pos()
            
            # Apply adaptive smoothing
            smoothed = st['euro'].filter(kf_pos, tnow, is_moving=st['is_moving'])
            
            # Update position - no restrictive publish guard
            st['position'] = tuple(smoothed)
            
            # Update stable position when peg settles
            if not st['is_moving'] and st['missing_frames'] == 0:
                speed = np.linalg.norm(st['kf'].vel())
                if speed < self.speed_off:
                    self.stable_positions[pid] = st['position']
    
    # --------------------------- Public interface ---------------------------
    def update(self, left_frame, right_frame, res_l=None, res_r=None):
        """Main update function"""
        # Get detections
        if res_l is None or res_r is None:
            res_l = model(left_frame, verbose=False, conf=0.10)[0]
            res_r = model(right_frame, verbose=False, conf=0.10)[0]
        
        left_pegs = self.get_center(res_l)
        right_pegs = self.get_center(res_r)
        
        # Stereo matching and triangulation
        matched_left, matched_right = self.find_pairs(left_pegs, right_pegs)
        points_3d = self.find_3d_points(matched_left, matched_right) if (matched_left and matched_right) else []
        
        # Initialize if needed
        if not self.initialized:
            if self._init_tracks(points_3d):
                return self._get_current_positions()
            else:
                return self._get_current_positions()
        
        # Time update
        tnow = time.perf_counter()
        dt = max(1e-3, tnow - self._t_prev)
        self._t_prev = tnow
        
        # Track update pipeline
        self._detect_moving_peg()
        assignments = self._associate_robust(points_3d)
        self._update_tracks(assignments, points_3d, tnow, dt)
        
        self.frame_count += 1
        return self._get_current_positions()
    
    def _get_current_positions(self):
        """Get current peg positions"""
        positions = {}
        for pid, st in self.pegs.items():
            positions[pid] = tuple(map(float, st['position']))
        return positions
    
    def get_debug_info(self):
        """Get debug information about tracking state"""
        info = {}
        for pid, st in self.pegs.items():
            info[pid] = {
                'is_moving': st['is_moving'],
                'speed': float(np.linalg.norm(st['kf'].vel())),
                'missing_frames': st['missing_frames'],
                'position': st['position']
            }
        info['moving_peg'] = self.moving_peg_id
        return info

# ============================== Helper Functions ==============================
def transform_pegs(pegs):
    """Transform 3D points from camera to world coordinates"""
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

# =================================== MAIN ====================================
if __name__ == "__main__":
    try:
        # Initialize rectification maps
        map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)
        
        # Initialize hybrid tracker
        peg_tracker = HybridPegTracker()
        frame_count = 0
        
        # Debug window
        if SHOW_DETECTIONS:
            cv2.namedWindow("YOLO detections (L|R)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("YOLO detections (L|R)", 1280, 480)
        
        # FPS tracking
        fps_counter = deque(maxlen=30)
        last_time = time.perf_counter()
        
        print("Hybrid Peg Tracker initialized. Waiting for 6 pegs...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Split and rectify
            left_frame = frame[:, :800]
            right_frame = frame[:, 800:]
            
            left_frame = cv2.remap(left_frame, map1x, map1y, cv2.INTER_LINEAR)
            right_frame = cv2.remap(right_frame, map2x, map2y, cv2.INTER_LINEAR)
            
            # Run YOLO once
            res_l = model(left_frame, verbose=False, conf=0.10)[0]
            res_r = model(right_frame, verbose=False, conf=0.10)[0]
            
            # Visualization
            if SHOW_DETECTIONS:
                vis_l = res_l.plot()
                vis_r = res_r.plot()
                vis = np.hstack([vis_l, vis_r])
                
                # Add debug info overlay
                debug_info = peg_tracker.get_debug_info()
                y_pos = 30
                
                # Show moving peg
                if 'moving_peg' in debug_info and debug_info['moving_peg'] is not None:
                    cv2.putText(vis, f"Moving: Peg {debug_info['moving_peg']+1}", 
                              (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_pos += 30
                
                # Show each peg's state
                for pid in range(6):
                    if pid in debug_info:
                        info = debug_info[pid]
                        color = (0, 255, 0) if not info['is_moving'] else (0, 165, 255)
                        if info['missing_frames'] > 0:
                            color = (0, 0, 255)
                        
                        text = f"P{pid+1}: "
                        if info['is_moving']:
                            text += "MOVING "
                        text += f"spd:{info['speed']:.1f} miss:{info['missing_frames']}"
                        
                        cv2.putText(vis, text, (10, y_pos), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        y_pos += 25
                
                # FPS counter
                current_time = time.perf_counter()
                fps = 1.0 / (current_time - last_time)
                fps_counter.append(fps)
                avg_fps = np.mean(fps_counter)
                cv2.putText(vis, f"FPS: {avg_fps:.1f}", 
                          (vis.shape[1]-150, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                last_time = current_time
                
                if vis.shape[1] > 1920:
                    scale = 1920.0 / vis.shape[1]
                    vis = cv2.resize(vis, (int(vis.shape[1]*scale), int(vis.shape[0]*scale)))
                cv2.imshow("YOLO detections (L|R)", vis)
            
            # Send frames to Unity
            send_frame_left(left_frame, frame_count)
            send_frame_right(right_frame, frame_count)
            
            # Update tracking
            peg_positions = peg_tracker.update(left_frame, right_frame, res_l=res_l, res_r=res_r)
            
            # Prepare Unity message
            message = {}
            for peg_id in range(6):
                if peg_id in peg_positions:
                    x, y, z = peg_positions[peg_id]
                else:
                    if peg_id in peg_tracker.pegs:
                        x, y, z = peg_tracker.pegs[peg_id]['position']
                    else:
                        x, y, z = 0.0, 0.0, 0.0
                
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
                    message[str(peg_id + 1)] = {"x": 0.0, "y": 0.0, "z": 0.0}
            
            # Send to Unity
            if message:
                try:
                    message_json = json.dumps(message)
                    output_socket.sendto(message_json.encode('utf-8'), (OUTPUT_IP, OUTPUT_PORT))
                except Exception as e:
                    print(f"Error sending message: {e}")
            
            frame_count += 1
            
            # Check for exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):  # Reset tracker
                peg_tracker = HybridPegTracker()
                print("Tracker reset - waiting for 6 pegs...")
    
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cap.release()
        output_socket.close()
        left_sock.close()
        right_sock.close()
        cv2.destroyAllWindows()
