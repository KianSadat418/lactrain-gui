import cv2
import numpy as np
import socket
import os
import json
import struct
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

class PegStereoMatcher:
    def __init__(self):
        self.max_pegs = 6
        self.max_missing_frames = 180
        self.position_history_length = 20
        self.movement_threshold = 2.5
        self.new_peg_confirm_frames = 5
        self.moving_peg_cooldown = 10 # Increased cooldown for more stable tracking
        self.min_peg_distance = 1.0  # Increased minimum distance between pegs (mm)

        
        self.potential_pegs = {}
        self.next_peg_id = 0
        self.last_moving_peg_change = 0
        self.frame_count = 0
        self.pegs = {}
        self.stable_positions = {}  # Last known stable positions
        self.moving_peg_id = None  # ID of the currently moving peg

    def get_center(self, results):
        if len(results.boxes) == 0:
            return []
        
        pegs = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x = float((x1 + x2) / 2)
            y = float((y1 + y2) / 2)
            pegs.append([x, y])
        return pegs
    
    def find_3d_points(self, left_points, right_points):
        points_3d = []
        for pt_left, pt_right in zip(left_points, right_points):
            pts_left = np.array(pt_left).reshape(2, 1)
            pts_right = np.array(pt_right).reshape(2, 1)
            
            point_4d = cv2.triangulatePoints(P1, P2, pts_left, pts_right)
            point_3d = point_4d[:3] / point_4d[3]
            x, y, z = point_3d[:, 0]
            points_3d.append((x + baseline_offset, -y, z))
            
        return points_3d
    
    def find_pairs(self, left_points, right_points):
        if not left_points or not right_points:
            return [], []
            
        if len(left_points) == 6 and len(right_points) == 6:
            all_pairings = [list(zip(left_points, perm)) for perm in itertools.permutations(right_points)]
            
            for pairings in all_pairings:
                valid = True
                left_matched = []
                right_matched = []
                
                for left_pt, right_pt in pairings:
                    if abs(left_pt[1] - right_pt[1]) > 15.0:
                        valid = False
                        break
                    left_matched.append(left_pt)
                    right_matched.append(right_pt)
                
                if valid:
                    points_3d = self.find_3d_points(left_matched, right_matched)
                    if all(0 <= p[2] <= 200 for p in points_3d):
                        return left_matched, right_matched
        
        return self.find_pairs_simple(left_points, right_points)
    
    def find_pairs_simple(self, left_points, right_points):
        result_pairs_right = []
        result_pairs_left = []
        
        for i in range(len(left_points)):
            for j in range(len(right_points)):
                if (right_points[j] in result_pairs_right or 
                    left_points[i] in result_pairs_left):
                    continue
                
                left_x = left_points[i]
                right_x = right_points[j]
                
                if abs(left_x[1] - right_x[1]) > 10.0:
                    continue
                
                point_in_3d = self.find_3d_points([left_x], [right_x])[0]
                if 0 <= point_in_3d[2] <= 200:
                    result_pairs_left.append(left_x)
                    result_pairs_right.append(right_x)
        
        return result_pairs_left, result_pairs_right
    
    def update(self, left_frame, right_frame):
        # Get detections
        res_l = model(left_frame, verbose=False)[0]
        res_r = model(right_frame, verbose=False)[0]
        
        left_pegs = self.get_center(res_l)
        right_pegs = self.get_center(res_r)
        
        # Find matching pairs
        matched_left, matched_right = self.find_pairs(left_pegs, right_pegs)
        
        if not matched_left or not matched_right:
            return self._get_current_positions()
        
        # Get 3D positions
        points_3d = self.find_3d_points(matched_left, matched_right)
        
        # Update tracking
        self._update_peg_positions(points_3d)
        
        return self._get_current_positions()
    
    def _find_moving_peg(self, current_positions):
        """Identify which peg is moving based on the 'only one moves' constraint"""
        if not self.stable_positions or not current_positions:
            return None
            
        max_movement = 0
        moving_id = None
        movements = {}
        
        # Calculate movement for all pegs
        for peg_id, pos in current_positions.items():
            if peg_id in self.stable_positions:
                movement = np.linalg.norm(np.array(pos) - np.array(self.stable_positions[peg_id]))
                movements[peg_id] = movement
                if movement > max_movement:
                    max_movement = movement
                    moving_id = peg_id
        
        # Only return a moving peg if its movement is significantly more than others
        if max_movement > self.movement_threshold * 1.5:  # 1.5x threshold for more confidence
            # Check if this peg's movement is at least 2x the next highest movement
            other_movements = [m for pid, m in movements.items() if pid != moving_id]
            if not other_movements or max_movement > 2 * max(other_movements, default=0):
                return moving_id
        
        return None

    def _is_valid_peg_position(self, position, existing_positions, min_distance=15.0):
        """Check if a new peg position is valid (not too close to existing ones)"""
        for existing_pos in existing_positions:
            if np.linalg.norm(np.array(position) - np.array(existing_pos)) < min_distance:
                return False
        return True

    def _update_peg_positions(self, points_3d):
        current_positions = {}
        
        # Get current stable positions if we don't have them yet
        if not self.stable_positions and self.pegs:
            self.stable_positions = {pid: data['position'] for pid, data in self.pegs.items()}
        
        # First, try to match existing pegs to new detections
        unused_detections = list(range(len(points_3d)))
        matched_pegs = set()
        
        # Sort peg IDs to maintain consistent ordering
        peg_ids = sorted(self.pegs.keys())
        
        # Check if we should be tracking a moving peg
        if self.moving_peg_id is not None and self.moving_peg_id in self.pegs:
            moving_peg = self.moving_peg_id
            best_match_idx = None
            min_dist = float('inf')
            
            # Find the best match for the moving peg
            for idx in unused_detections:
                pos = points_3d[idx]
                last_pos = self.pegs[moving_peg]['position']
                dist = np.linalg.norm(np.array(pos) - np.array(last_pos))
                
                if dist < self.movement_threshold * 6 and dist < min_dist:
                    min_dist = dist
                    best_match_idx = idx
            
            if best_match_idx is not None:
                # Update the moving peg's position
                pos = points_3d[best_match_idx]
                self.pegs[moving_peg]['position'] = pos
                self.pegs[moving_peg]['history'].append(pos)
                if len(self.pegs[moving_peg]['history']) > self.position_history_length:
                    self.pegs[moving_peg]['history'].popleft()
                self.pegs[moving_peg]['missing_frames'] = 0
                current_positions[moving_peg] = pos
                unused_detections.remove(best_match_idx)
                matched_pegs.add(moving_peg)
        
        # Match all pegs to detections (only update moving peg if one is set)
        for peg_id in peg_ids:
            if peg_id in matched_pegs:
                continue
                
            best_match_idx = None
            min_dist = float('inf')
            
            # Find the closest detection to this peg's last known position
            for idx in unused_detections:
                pos = points_3d[idx]
                last_pos = self.pegs[peg_id]['position']
                dist = np.linalg.norm(np.array(pos) - np.array(last_pos))
                
                if dist < self.movement_threshold * 6 and dist < min_dist:
                    min_dist = dist
                    best_match_idx = idx
            
            if best_match_idx is not None:
                # Found a match for this peg
                pos = points_3d[best_match_idx]
                
                # Only update position if it's a valid position
                existing_positions = [p for pid, p in current_positions.items() if pid != peg_id]
                if self._is_valid_peg_position(pos, existing_positions):
                    # Check if this peg has moved significantly from its stable position
                    if self.moving_peg_id is None and len(self.pegs[peg_id]['history']) > 0 and peg_id in self.stable_positions:
                        movement = np.linalg.norm(np.array(pos) - np.array(self.stable_positions[peg_id]))
                        if movement > self.movement_threshold * 1.5:
                            # Only set as moving peg if no other peg is currently moving
                            self.moving_peg_id = peg_id
                            self.last_moving_peg_change = self.frame_count
                    
                    self.pegs[peg_id]['position'] = pos
                    self.pegs[peg_id]['history'].append(pos)
                    if len(self.pegs[peg_id]['history']) > self.position_history_length:
                        self.pegs[peg_id]['history'].popleft()
                    self.pegs[peg_id]['missing_frames'] = 0
                    current_positions[peg_id] = pos
                    unused_detections.remove(best_match_idx)
                    matched_pegs.add(peg_id)
        
        # Handle any remaining detections (new pegs or lost pegs) - only if no moving peg
        if self.moving_peg_id is None:
            for idx in unused_detections:
                pos = points_3d[idx]
                
                # Only add new pegs if we're below max_pegs and position is valid
                if len(current_positions) < self.max_pegs and self._is_valid_peg_position(pos, current_positions.values()):
                    peg_id = self.next_peg_id
                    self.pegs[peg_id] = {
                        'position': pos,
                        'history': deque([pos], maxlen=self.position_history_length),
                        'missing_frames': 0
                    }
                    current_positions[peg_id] = pos
                    self.next_peg_id += 1
        
        # Update stable positions if we have all pegs
        if len(current_positions) == self.max_pegs:
            self.stable_positions = {pid: data['position'] for pid, data in self.pegs.items()}
        
        # Check if we should be tracking a moving peg
        if self.moving_peg_id is None:
            # No moving peg - check if any peg is moving significantly
            moving_peg = self._find_moving_peg(current_positions)
            if moving_peg is not None and self.frame_count - self.last_moving_peg_change > self.moving_peg_cooldown:
                self.moving_peg_id = moving_peg
                self.last_moving_peg_change = self.frame_count
        else:
            # We have a moving peg - check if it has stopped moving
            if self.moving_peg_id in current_positions and self.moving_peg_id in self.stable_positions:
                movement = np.linalg.norm(
                    np.array(current_positions[self.moving_peg_id]) - 
                    np.array(self.stable_positions[self.moving_peg_id])
                )
                # If movement is very small for cooldown period, clear the moving peg
                if movement < self.movement_threshold * 0.5:  
                    if self.frame_count - self.last_moving_peg_change > self.moving_peg_cooldown:
                        # Update stable position for this peg and clear moving_peg_id
                        self.stable_positions[self.moving_peg_id] = current_positions[self.moving_peg_id]
                        self.moving_peg_id = None
                        # Reset the cooldown to allow immediate detection of new moving pegs
                        self.last_moving_peg_change = self.frame_count
        
        self.frame_count += 1
        
        # Clean up pegs that have been missing for too long
        missing_pegs = []
        for peg_id in list(self.pegs.keys()):
            if peg_id not in current_positions:
                if self.pegs[peg_id]['missing_frames'] >= self.max_missing_frames:
                    missing_pegs.append(peg_id)
                else:
                    self.pegs[peg_id]['missing_frames'] += 1
                    # Always use current position when no peg is moving
                    # When a peg is moving, only use current position for that peg
                    if self.moving_peg_id is None or peg_id == self.moving_peg_id:
                        current_positions[peg_id] = self.pegs[peg_id]['position']
                    elif peg_id in self.stable_positions:
                        current_positions[peg_id] = self.stable_positions[peg_id]
        
        # Remove missing pegs
        for peg_id in missing_pegs:
            del self.pegs[peg_id]
            if peg_id in self.stable_positions:
                del self.stable_positions[peg_id]
            if peg_id == self.moving_peg_id:
                self.moving_peg_id = None
        
        # If we have a moving peg but it's not in current positions, clear it
        if self.moving_peg_id is not None and self.moving_peg_id not in current_positions:
            self.moving_peg_id = None
        
        return current_positions
    
    def _get_current_positions(self):
        return {pid: data['position'] for pid, data in self.pegs.items()}

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
