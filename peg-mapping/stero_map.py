import cv2
import numpy as np
import socket
import os
from ultralytics import YOLO
import json
from scipy.optimize import linear_sum_assignment
import itertools
import struct
from collections import deque
from scipy.spatial.distance import cdist
import time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Enable OpenEXR support in OpenCV

model = YOLO("Assets/Scripts/Peg-Detection-Scripts/Training-05-22/result/content/runs/detect/yolo8_peg_detector/weights/best.pt")
calib = np.load("Assets/Scripts/Camera Calibration/stereo_camera_calibration2.npz")
transformation_matrix = np.load("Assets/Scripts/Camera Calibration/TransformationMatrix.npz")["affine_transform_ransac"]
K1, D1 = calib["cameraMatrixL"], calib["distL"]
K2, D2 = calib["cameraMatrixR"], calib["distR"]
R, T = calib["R"], calib["T"]

baseline_offset = T[0][0] / 2.0  # Half the translation between cameras (baseline)

image_size = (800, 600)

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T, alpha=0, flags=cv2.CALIB_ZERO_DISPARITY)

UNITY_LEFT_PORT = 9998
UNITY_RIGHT_PORT = 9999
OUTPUT_PORT = 9989
OUTPUT_IP = "127.0.0.1"

output_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
left_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
right_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

CHUNK_SIZE = 30000
count = 0

# pegs = {str(i): ([0, 0], [0, 0], [0, 0, 0]) for i in range(1, 7)}
# isInitialized = False

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


def find_pairs2(left_points, right_points):
    result_pairs_right = []
    result_pairs_left = []
    for i in range(len(left_points)):
        for j in range(len(right_points)):
            if right_points[j] in result_pairs_right or left_points[i] in result_pairs_left:
                continue

            left_x = left_points[i]
            right_x = right_points[j]
            if abs(left_x[1] - right_x[1]) > 10.0:
                continue

            pointIn3D = find_3D_points([left_x], [right_x])[0]
            if pointIn3D[2] > 200.0 or pointIn3D[2] < 0.0:
                continue

            result_pairs_left.append(left_x)
            result_pairs_right.append(right_x)

    return result_pairs_left, result_pairs_right

def find_pairs(left_points, right_points):
    all_pairings = [list(zip(left_points, perm)) for perm in itertools.permutations(right_points)]

    best_pairs_left = []
    best_pairs_right = []
    for j, pairings in enumerate(all_pairings):
        best_pair_found = True

        left_x = [pt[0] for pt in pairings]
        right_x = [pt[1] for pt in pairings]

        for i in range(len(left_x)):
            if abs(left_x[i][1] - right_x[i][1]) > 15.0:
                best_pair_found = False
                break

        if not best_pair_found:
            continue

        pointsIn3D = find_3D_points(left_x, right_x)
        for i, (x, y, z) in enumerate(pointsIn3D):
            if z > 200.0 or z < 0.0:
                best_pair_found = False
                break
            
        if not best_pair_found:
            continue
        
        best_pairs_left.append(left_x)
        best_pairs_right.append(right_x)

    return best_pairs_left[0], best_pairs_right[0]

def transform_pegs(pegs):
    """Transform 3D points from camera coordinates to world coordinates using the transformation matrix."""
    transformedPegs = []
    for i, (x, y, z) in enumerate(pegs):
        p = np.array([round(x, 3), round(y, 3), round(z, 3), 1.0])
        p_transformed = transformation_matrix @ p
        transformedPegs.append(p_transformed[:3])

    return transformedPegs

def find_3D_points(left_points, right_points):
    points_3D = []
    for pt_left, pt_right in zip(left_points, right_points):
        pts_left = np.array(pt_left).reshape(2, 1)
        pts_right = np.array(pt_right).reshape(2, 1)

        # Triangulate 3D point
        point_4d = cv2.triangulatePoints(P1, P2, pts_left, pts_right)
        point_3d = point_4d[:3] / point_4d[3]  # Convert from homogeneous to 3D
        x, y, z = point_3d[:, 0]
        points_3D.append((x + baseline_offset, -y, z))

    return points_3D

def get_center(results):
    if len(results.boxes) == 0:
        return []  # Return empty list instead of None
    
    pegs = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Add .cpu().numpy()
        x = float((x1 + x2) / 2)
        y = float((y1 + y2) / 2)
        pegs.append([x, y])

    return pegs

def res_without_rectify(right_frame, left_frame):
    res_l = model(left_frame, verbose=False)[0]
    res_r = model(right_frame, verbose=False)[0]

    left_frame_peg = get_center(res_l)
    right_frame_peg = get_center(res_r)

    return left_frame_peg, right_frame_peg

class PegKalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 3)  # 6 state vars: pos (x,y,z) + vel (vx,vy,vz), 3 measurements

        # Transition matrix: x_k = A * x_(k-1)
        dt = 1.0  # Assume 1 frame per time step
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        # Measurement matrix: only position is observed
        self.kf.measurementMatrix = np.eye(3, 6, dtype=np.float32)

        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-4  # Lower → smoother, Higher → adapts quicker
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-1  # Lower = more trust in detections
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 1  # Initial uncertainty

        self.kf.statePost = np.zeros((6, 1), dtype=np.float32)

    def initialize(self, xyz):
        """Set initial position and zero velocity"""
        x, y, z = xyz
        self.kf.statePost[:3, 0] = [x, y, z]
        self.kf.statePost[3:, 0] = [0, 0, 0]

    def predict(self):
        pred = self.kf.predict()
        return pred[:3].flatten()

    def correct(self, xyz):
        measurement = np.array(xyz, dtype=np.float32).reshape(3, 1)
        corrected = self.kf.correct(measurement)
        return corrected[:3].flatten()


class OptimizedPegTracker:
    def __init__(self):
        # Adjusted parameters based on your specs
        self.max_distance_2d = 30  # pixels (considering 60fps, movement should be small between frames)
        self.max_distance_3d = 40  # mm (5cm/s at 60fps = ~0.8mm per frame, with safety margin)
        self.max_missing_frames = 15  # 0.5 seconds at 60fps
        self.max_prediction_frames = self.max_missing_frames + 15
        self.initialization_frames = 5  # Require stable detection for initialization
        
        self.pegs = {}
        self.next_peg_id = 0
        self.frame_count = 0
        self.is_initialized = False
        self.initialization_buffer = []
        
        # Movement tracking for identifying active peg
        self.movement_threshold = 5.0  # mm - movement above this suggests user interaction
        self.movement_history_size = 15  # 0.5 seconds of history
        
    def add_initialization_frame(self, left_detections, right_detections, triangulated_3d):
        """Buffer frames for stable initialization"""
        if len(left_detections) == 6 and len(right_detections) == 6 and len(triangulated_3d) == 6:
            self.initialization_buffer.append({
                'left': left_detections,
                'right': right_detections,
                'triangulated': triangulated_3d,
                'frame': self.frame_count
            })
            
            # Keep only recent frames
            if len(self.initialization_buffer) > self.initialization_frames:
                self.initialization_buffer.pop(0)
                
            # Check if we can initialize
            if len(self.initialization_buffer) == self.initialization_frames:
                self._attempt_initialization()
        else:
            # Clear buffer if we don't have 6 pegs
            self.initialization_buffer.clear()
    
    def _attempt_initialization(self):
        """Initialize pegs using stable detections from buffer"""
        if len(self.initialization_buffer) < self.initialization_frames:
            return
            
        # Use the most recent frame for initialization
        latest_frame = self.initialization_buffer[-1]
        
        # Verify stability by checking if pegs haven't moved too much
        first_frame = self.initialization_buffer[0]
        
        stable = True
        for i in range(6):
            movement_3d = np.linalg.norm(
                np.array(latest_frame['triangulated'][i]) - 
                np.array(first_frame['triangulated'][i])
            )
            if movement_3d > 20:  # 2cm movement over initialization period suggests instability
                stable = False
                break
        
        if stable:
            # Initialize pegs
            for i in range(6):
                peg_id = self.next_peg_id
                peg_filter = PegKalmanFilter()
                peg_filter.initialize(latest_frame['triangulated'][i])
                self.pegs[peg_id] = {
                    'left_2d': latest_frame['left'][i],
                    'right_2d': latest_frame['right'][i],
                    'position_3d': latest_frame['triangulated'][i],
                    'velocity_3d': np.zeros(3),
                    'last_seen': self.frame_count,
                    'missing_frames': 0,
                    'movement_history': deque(maxlen=self.movement_history_size),
                    'confidence': 1.0,
                    'total_movement': 0.0,
                    'kf': peg_filter
                }
                self.next_peg_id += 1
            
            self.is_initialized = True
            self.initialization_buffer.clear()
            print(f"Pegs initialized successfully with {len(self.pegs)} pegs")
        else:
            # Clear buffer and start over
            self.initialization_buffer.clear()
    
    def update_tracks(self, left_detections, right_detections, triangulated_3d):
        """Main update function"""
        self.frame_count += 1
        
        if not self.is_initialized:
            self.add_initialization_frame(left_detections, right_detections, triangulated_3d)
            return self.get_current_state()
        
        # Handle different detection scenarios
        if len(triangulated_3d) == 6:
            return self._update_with_full_detections(left_detections, right_detections, triangulated_3d)
        else:
            return self._update_with_partial_detections(left_detections, right_detections, triangulated_3d)
    
    def _update_with_full_detections(self, left_detections, right_detections, triangulated_3d):
        """Update when we have exactly 6 detections"""
        active_peg_ids = [pid for pid, peg in self.pegs.items() if peg['missing_frames'] < self.max_missing_frames]
 
        # Create cost matrix based on 3D distance with velocity prediction
        if len(active_peg_ids) != len(triangulated_3d):
            print(f"[Warning] Mismatch: {len(active_peg_ids)} active pegs vs {len(triangulated_3d)} detections.")
            return self.get_current_state()

        cost_matrix = np.zeros((len(active_peg_ids), len(triangulated_3d)))
        predictions = self._predict_positions()

        if cost_matrix.size == 0:
            print("[Warning] Empty cost matrix — skipping frame.")
            return self.get_current_state()
        
        for i, peg_id in enumerate(active_peg_ids):
            predicted_pos = predictions.get(peg_id, self.pegs[peg_id]['position_3d'])
            for j, detection_3d in enumerate(triangulated_3d):
                distance_3d = np.linalg.norm(predicted_pos - np.array(detection_3d))
                
                # Add 2D distance penalty for additional validation
                left_dist = np.linalg.norm(np.array(self.pegs[peg_id]['left_2d']) - np.array(left_detections[j]))
                right_dist = np.linalg.norm(np.array(self.pegs[peg_id]['right_2d']) - np.array(right_detections[j]))
                
                # Combined cost with weights
                cost_matrix[i, j] = distance_3d + 0.1 * (left_dist + right_dist)
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Update pegs with assignments
        for i, j in zip(row_indices, col_indices):
            peg_id = active_peg_ids[i]
            if cost_matrix[i, j] < self.max_distance_3d:
                self._update_peg(peg_id, left_detections[j], right_detections[j], triangulated_3d[j])
            else:
                # Assignment too costly, mark as missing
                self.pegs[peg_id]['missing_frames'] += 1
        
        return self.get_current_state()
    
    def _update_with_partial_detections(self, left_detections, right_detections, triangulated_3d):
        """Update when we have partial detections"""
        active_peg_ids = [pid for pid, peg in self.pegs.items() if peg['missing_frames'] < self.max_missing_frames]
        predictions = self._predict_positions()
        
        # Greedy matching for partial detections
        matched_pegs = set()
        used_detections = set()
        
        # Sort detections by confidence (distance to predictions)
        detection_scores = []
        for i, detection_3d in enumerate(triangulated_3d):
            best_distance = float('inf')
            for peg_id in active_peg_ids:
                if peg_id in matched_pegs:
                    continue
                predicted_pos = self.pegs[peg_id]['kf'].predict()
                distance = np.linalg.norm(predicted_pos - np.array(detection_3d))
                best_distance = min(best_distance, distance)
            detection_scores.append((i, best_distance))
  
        # Sort by best distance (most confident matches first)
        detection_scores.sort(key=lambda x: x[1])
        
        # Match in order of confidence
        for det_idx, _ in detection_scores:
            best_match = None
            best_distance = float('inf')
            detection = np.array(triangulated_3d[det_idx])

            for peg_id in active_peg_ids:
                if peg_id in matched_pegs:
                    continue

                predicted_pos = self.pegs[peg_id]['kf'].predict()
                distance = np.linalg.norm(predicted_pos - detection)

                if distance < self.max_distance_3d and distance < best_distance:
                    best_match = peg_id
                    best_distance = distance

            if best_match is not None:
                self._update_peg(best_match, left_detections[det_idx], right_detections[det_idx], detection.tolist())
                matched_pegs.add(best_match)
                used_detections.add(det_idx)
        
        # Increment missing frames for unmatched pegs
        for peg_id in active_peg_ids:
            if peg_id not in matched_pegs:
                self.pegs[peg_id]['missing_frames'] += 1

        # Try to recover lost pegs
        lost_pegs = [
            pid for pid, peg in self.pegs.items()
            if peg['missing_frames'] >= self.max_missing_frames and peg['missing_frames'] < self.max_missing_frames + 10
        ]

        for det_idx in range(len(triangulated_3d)):
            if det_idx in used_detections:
                continue

            detection = np.array(triangulated_3d[det_idx])

            for peg_id in lost_pegs:
                predicted_pos = self.pegs[peg_id]['kf'].predict()
                distance = np.linalg.norm(predicted_pos - detection)

                if distance < self.max_distance_3d:
                    # Consider it recovered
                    self.pegs[peg_id]['missing_frames'] = 0
                    self.pegs[peg_id]['position_3d'] = detection.tolist()
                    self.pegs[peg_id]['kf'].correct(detection.tolist())
                    self.pegs[peg_id]['left_2d'] = left_detections[det_idx]
                    self.pegs[peg_id]['right_2d'] = right_detections[det_idx]
                    self.pegs[peg_id]['last_seen'] = self.frame_count

                    self.pegs[peg_id]['movement_history'].append(0.0)
                    used_detections.add(det_idx)
                    print(f"[Recovery] Peg {peg_id} reappeared.")
                    break
        
        return self.get_current_state()
    
    def _predict_positions(self):
        """Predict next positions based on velocity"""
        predictions = {}
        for peg_id, peg in self.pegs.items():
            if peg['missing_frames'] < self.max_missing_frames:
                # Simple linear prediction
                predictions[peg_id] = np.array(peg['position_3d']) + peg['velocity_3d']
        return predictions
    
    def _update_peg(self, peg_id, left_2d, right_2d, position_3d):
        """Update individual peg"""
        peg = self.pegs[peg_id]
        
        # Kalman correction
        alpha = 0.3  # Smoothing factor
        filtered_pos = peg['kf'].correct(position_3d)

        # Estimate velocity and position
        movement = filtered_pos - np.array(peg['position_3d'])
        movement_magnitude = np.linalg.norm(movement)

        peg['velocity_3d'] = alpha * movement + (1 - alpha) * peg['velocity_3d']
        peg['position_3d'] = filtered_pos.tolist()
        
        # Update position
        peg['left_2d'] = left_2d
        peg['right_2d'] = right_2d
        peg['last_seen'] = self.frame_count
        peg['missing_frames'] = 0
        
        # Track movement history
        peg['movement_history'].append(movement_magnitude)
        peg['total_movement'] += movement_magnitude
        
        # Update confidence based on movement consistency
        if movement_magnitude < 5:  # Small movement = high confidence
            peg['confidence'] = min(1.0, peg['confidence'] + 0.02)
        elif movement_magnitude > 30:  # Large movement = lower confidence
            peg['confidence'] = max(0.3, peg['confidence'] - 0.05)
    
    def get_current_state(self):
        state = {}
        for peg_id in range(6):
            peg = self.pegs.get(peg_id)
            if peg is None:
                continue  # Should not happen, but safe check

            if peg['missing_frames'] < self.max_prediction_frames:
                if peg['missing_frames'] < self.max_missing_frames:
                    pos = peg['position_3d']
                    status = "tracked"
                else:
                    pos = peg['kf'].predict().tolist()
                    status = "predicted"

                # Apply exponential confidence decay
                decay_factor = 0.9 ** (peg['missing_frames'] - self.max_missing_frames)
                confidence = max(0.0, peg['confidence'] * decay_factor)

                state[str(peg_id)] = {
                    "position_3d": pos,
                    "moving": peg.get('is_moving', False),
                    "confidence": round(confidence, 3),
                    "status": status
                }
        return state
    
    def get_moving_peg_id(self):
        """Identify which peg is currently being moved"""
        if not self.is_initialized:
            return None
        
        movement_scores = {}
        for peg_id, peg in self.pegs.items():
            if peg['missing_frames'] < self.max_missing_frames and len(peg['movement_history']) > 5:
                # Average movement over recent frames
                recent_movements = list(peg['movement_history'])[-10:]
                avg_movement = np.mean(recent_movements)
                movement_scores[peg_id] = avg_movement
        
        if movement_scores:
            moving_peg_id = max(movement_scores, key=movement_scores.get) # type: ignore
            if movement_scores[moving_peg_id] > self.movement_threshold:
                return moving_peg_id
        
        return None
    
    def get_predicted_position(self, peg_id):
        """Return the predicted position of a peg if not currently detected."""
        if peg_id in self.pegs:
            peg = self.pegs[peg_id]
            if peg['missing_frames'] < self.max_prediction_frames:
                return peg['position_3d']  # Use last known
            else:
                return peg['kf'].predict().tolist()  # Kalman-based prediction
        return [None, None, None]

# Initialize tracker
tracker = OptimizedPegTracker()

if __name__ == "__main__":
    try:
        map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            left_frame = frame[:, :800]
            right_frame = frame[:, 800:]

            left_frame = cv2.remap(left_frame, map1x, map1y, cv2.INTER_LINEAR)
            right_frame = cv2.remap(right_frame, map2x, map2y, cv2.INTER_LINEAR)

            send_frame_left(left_frame, count)
            send_frame_right(right_frame, count)

            count += 1

            left_frame_peg, right_frame_peg = res_without_rectify(right_frame, left_frame)

            # Handle different detection counts
            pegsInCamera = []
            if left_frame_peg and right_frame_peg:
                if len(left_frame_peg) == 6 and len(right_frame_peg) == 6:
                    # Perfect case - use your existing pairing logic
                    try:
                        left_frame_peg, right_frame_peg = find_pairs(left_frame_peg, right_frame_peg)
                        pegsInCamera = find_3D_points(left_frame_peg, right_frame_peg)
                    except (IndexError, ValueError) as e:
                        print(f"Error in find_pairs: {e}")
                        pegsInCamera = []
                elif len(left_frame_peg) > 0 and len(right_frame_peg) > 0:
                    # Partial detections - use simpler pairing
                    try:
                        left_frame_peg, right_frame_peg = find_pairs2(left_frame_peg, right_frame_peg)
                        if left_frame_peg and right_frame_peg:
                            pegsInCamera = find_3D_points(left_frame_peg, right_frame_peg)
                    except (IndexError, ValueError) as e:
                        print(f"Error in find_pairs2: {e}")
                        pegsInCamera = []

            # Update tracker
            current_pegs = tracker.update_tracks(left_frame_peg or [], right_frame_peg or [], pegsInCamera)
            moving_peg_id = tracker.get_moving_peg_id()

            # Send data based on tracker state
            message = {}
            if tracker.is_initialized and current_pegs:
                # Use tracked positions
                peg_counter = 1
                for peg_id in range(6):  # Always send 6 pegs
                    if peg_id in current_pegs:
                        x, y, z = current_pegs[peg_id]['position_3d']
                    else:
                        x, y, z = tracker.get_predicted_position(peg_id)

                    try:
                        if None not in (x, y, z):
                            transformed_pos = transform_pegs([(x, y, z)])[0]
                            message[str(peg_id + 1)] = {
                                "x": round(transformed_pos[0], 3),
                                "y": round(transformed_pos[1], 3),
                                "z": round(transformed_pos[2], 3)
                            }
                    except Exception as e:
                        print(f"Error transforming peg {peg_id}: {e}")
                
                # Add moving peg info if needed
                if moving_peg_id is not None:
                    message["moving_peg"] = str(moving_peg_id)
                    
                #print(f"Tracking {len(current_pegs)} pegs, moving peg: {moving_peg_id}")
            elif len(pegsInCamera) == 6:
                # Fallback to direct detection during initialization
                try:
                    print(len(pegsInCamera), len(current_pegs))
                    pegsInHololens = transform_pegs(pegsInCamera)
                    for i, (x, y, z) in enumerate(pegsInHololens):
                        message[str(i+1)] = {
                            "x": round(x, 3),
                            "y": round(y, 3),
                            "z": round(z, 3)
                        }
                    print(f"Initialization: detected {len(pegsInCamera)} pegs")
                except Exception as e:
                    print(f"Error during initialization: {e}")
                    continue
            else:
                print(f"Insufficient detections: L={len(left_frame_peg or [])}, R={len(right_frame_peg or [])}")
                
                if tracker.is_initialized:
                    # Fallback to predictions only
                    for peg_id in range(6):
                        x, y, z = tracker.get_predicted_position(peg_id)
                        try:
                            if None not in (x, y, z):
                                transformed_pos = transform_pegs([(x, y, z)])[0]
                                message[str(peg_id + 1)] = {
                                    "x": round(transformed_pos[0], 3),
                                    "y": round(transformed_pos[1], 3),
                                    "z": round(transformed_pos[2], 3)
                                }
                        except Exception as e:
                            print(f"Error transforming predicted peg {peg_id}: {e}")
                else:
                    # Still initializing — skip this frame
                    continue

            if message:
                try:
                    message_json = json.dumps(message)
                    output_socket.sendto(message_json.encode('utf-8'), (OUTPUT_IP, OUTPUT_PORT))
                except Exception as e:
                    print(f"Error sending message: {e}")

            # After the tracker.update_tracks call, add:
            if tracker.is_initialized:
                print(f"Frame {count}: Tracking {len(current_pegs)}/6 pegs, Moving: {moving_peg_id}")

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cap.release()
        output_socket.close()
        left_sock.close()
        right_sock.close()
        cv2.destroyAllWindows()
