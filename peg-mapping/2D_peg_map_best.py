import cv2
import numpy as np
import socket
import math
from ultralytics import YOLO

import os
# Set environment variable to handle OpenMP runtime warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

UNITY_LEFT_PORT = 9998
UNITY_RIGHT_PORT = 9999
OUTPUT_PORT = 9989
OUTPUT_IP = "127.0.0.1"

left_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
right_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
output_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

model = YOLO("C:\\Users\\kiansadat\\Desktop\\Azimi Project\\3D Eye Gaze\\Assets\\Scripts\\Peg-Detection-Scripts\\Training-05-22\\result\\content\\runs\\detect\\yolo8_peg_detector\\weights\\best.pt")
transformation_matrix = np.identity(3)

def setup_camera(camera_index=1, width=800, height=600):
    cap = cv2.VideoCapture(camera_index)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    if not cap.isOpened():
        raise Exception("Error: Could not open camera.")
    
    return cap

class PegTracker:
    def __init__(self, max_pegs=6, movement_threshold=5, still_frames=10, max_speed=150):
        self.max_pegs = max_pegs
        self.movement_threshold = movement_threshold
        self.still_frames = still_frames
        self.max_speed = max_speed  # Maximum expected pixels/frame for a moving peg
        
        # Track pegs and their states
        self.pegs = [None] * max_pegs  # Current positions
        self.last_positions = [None] * max_pegs  # Last known positions
        self.frames_since_seen = [0] * max_pegs  # Frames since each peg was seen
        self.frames_since_move = [0] * max_pegs  # Frames since each peg moved
        self.velocities = [(0, 0)] * max_pegs  # Current velocity (dx, dy) per peg
        self.is_moving = False
        self.moving_peg_idx = -1
        self.frame_count = 0
        self.peg_history = [[] for _ in range(max_pegs)]  # History of positions for each peg

    def distance(self, pos1, pos2):
        return math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)

    def get_center(self, results):
        if len(results.boxes) == 0:
            return []
        
        pegs = []
        confidences = []
        
        # Get all detections with their confidence scores
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            x = float((x1 + x2) / 2)
            y = float((y1 + y2) / 2)
            pegs.append((x, y, conf))
            confidences.append(conf)
        
        # Sort by confidence (highest first)
        if pegs:
            pegs = [p for _, p in sorted(zip(confidences, pegs), reverse=True)]
        
        return pegs

    def find_closest_peg(self, point, pegs):
        """Find the closest peg to the given point."""
        if not pegs:
            return None, float('inf')
        
        closest_peg = None
        min_dist = float('inf')
        
        for peg in pegs:
            dist = self.distance(point, peg)
            if dist < min_dist:
                min_dist = dist
                closest_peg = peg
                
        return closest_peg, min_dist

    def update(self, frame):
        self.frame_count += 1
        
        # Get detections with confidence scores
        results = model(frame, verbose=False)[0]
        current_detections = self.get_center(results)
        
        # Convert to list of (x,y) tuples, sorted by confidence
        current_pegs = [(x, y) for x, y, _ in current_detections[:self.max_pegs * 2]]  # Allow some extra detections
        
        # Initialize pegs if first frame with enough detections
        if not any(self.pegs) and len(current_pegs) >= self.max_pegs:
            for i in range(self.max_pegs):
                if i < len(current_pegs):
                    self.pegs[i] = current_pegs[i]
                    self.last_positions[i] = current_pegs[i]
            return self.pegs.copy()
        
        # If we don't have pegs yet, return empty list
        if not any(self.pegs):
            return []
            
        # Every 30 frames, try to reinitialize any missing pegs
        if self.frame_count % 30 == 0 and len(current_pegs) >= self.max_pegs:
            missing_pegs = [i for i, p in enumerate(self.pegs) if p is None]
            if missing_pegs and len(current_pegs) >= len(missing_pegs):
                # Sort missing pegs by their last known position (top to bottom, left to right)
                # This helps maintain consistent IDs
                missing_pegs.sort(key=lambda i: (self.peg_history[i][-1][1], self.peg_history[i][-1][0]) 
                                if self.peg_history[i] else (float('inf'), float('inf')))
                
                # Sort current detections by position (top to bottom, left to right)
                current_sorted = sorted(current_pegs, key=lambda p: (p[1], p[0]))
                
                for i, idx in enumerate(missing_pegs):
                    if i < len(current_sorted):
                        self.pegs[idx] = current_sorted[i]
                        self.last_positions[idx] = current_sorted[i]
                        self.frames_since_seen[idx] = 0
                        self.velocities[idx] = (0, 0)
                        if idx < len(self.peg_history):
                            self.peg_history[idx].append(current_sorted[i])
                        else:
                            self.peg_history.append([current_sorted[i]])
        
        # Find the best match for each peg with prediction
        matched = [False] * len(current_pegs)
        new_positions = [None] * self.max_pegs
        
        # First, try to match each peg to the closest detection with prediction
        for i in range(self.max_pegs):
            if self.pegs[i] is None:
                continue
                
            min_dist = float('inf')
            best_match = -1
            
            # Predict next position based on velocity
            predicted_x = self.pegs[i][0] + self.velocities[i][0]
            predicted_y = self.pegs[i][1] + self.velocities[i][1]
            
            for j, (x, y) in enumerate(current_pegs):
                if matched[j]:
                    continue
                
                # Calculate distance to predicted position
                dist = self.distance((predicted_x, predicted_y), (x, y))
                
                # Also consider direct distance for slow movements
                direct_dist = self.distance(self.pegs[i], (x, y))
                dist = min(dist, direct_dist)
                
                if dist < min_dist and dist < self.max_speed * 1.5:  # Allow some margin over max speed
                    min_dist = dist
                    best_match = j
            
            if best_match != -1:
                new_positions[i] = current_pegs[best_match]
                matched[best_match] = True
                self.frames_since_seen[i] = 0  # Reset not seen counter
                
                # Update velocity (simple low-pass filter)
                dx = (new_positions[i][0] - self.pegs[i][0]) * 0.3
                dy = (new_positions[i][1] - self.pegs[i][1]) * 0.3
                self.velocities[i] = (dx, dy)
            else:
                # If no match found, increment not seen counter
                self.frames_since_seen[i] += 1
                
                # If not seen for too long, mark for reinitialization
                if self.frames_since_seen[i] > 30:  # ~1 second at 30fps
                    # Keep the last known position in history
                    if self.pegs[i] is not None and i < len(self.peg_history):
                        self.peg_history[i].append(self.pegs[i])
                    self.pegs[i] = None
                    self.velocities[i] = (0, 0)
        
        # Check for movement using velocity
        movement_detected = False
        moving_peg = -1
        max_speed = 0
        
        for i in range(self.max_pegs):
            if self.pegs[i] is None:
                continue
                
            if new_positions[i] is None:
                continue
                
            # Calculate speed (pixels/frame)
            speed = self.distance((0, 0), self.velocities[i])
            
            if speed > self.movement_threshold and speed > max_speed:
                movement_detected = True
                max_speed = speed
                moving_peg = i
        
        # If we detect movement and no peg is currently moving, start tracking movement
        if movement_detected and not self.is_moving:
            self.is_moving = True
            self.moving_peg_idx = moving_peg
        
        # If we're tracking a moving peg
        if self.is_moving and self.moving_peg_idx != -1:
            # Only update the moving peg's position
            if new_positions[self.moving_peg_idx] is not None:
                self.pegs[self.moving_peg_idx] = new_positions[self.moving_peg_idx]
            
            # Check if movement has stopped (peg has been still for N frames)
            if max_speed < self.movement_threshold:
                self.frames_since_move[self.moving_peg_idx] += 1
                if self.frames_since_move[self.moving_peg_idx] > self.still_frames:
                    self.is_moving = False
                    self.moving_peg_idx = -1
            else:
                self.frames_since_move[self.moving_peg_idx] = 0
        else:
            # No movement detected, update all pegs
            for i in range(self.max_pegs):
                if new_positions[i] is not None:
                    self.pegs[i] = new_positions[i]
                    
        # Try to match any unmatched detections to missing pegs
        unmatched_dets = [j for j, m in enumerate(matched) if not m and j < len(current_pegs)]
        missing_pegs = [i for i, p in enumerate(self.pegs) if p is None]
        
        if unmatched_dets and missing_pegs:
            # Sort missing pegs by their last known position
            missing_pegs.sort(key=lambda i: (self.peg_history[i][-1][1], self.peg_history[i][-1][0]) 
                            if self.peg_history[i] else (float('inf'), float('inf')))
            
            # Sort unmatched detections by position
            unmatched_positions = sorted([current_pegs[j] for j in unmatched_dets], 
                                       key=lambda p: (p[1], p[0]))
            
            # Assign closest matches based on position
            for i, peg_idx in enumerate(missing_pegs):
                if i < len(unmatched_positions):
                    self.pegs[peg_idx] = unmatched_positions[i]
                    self.velocities[peg_idx] = (0, 0)
                    self.frames_since_seen[peg_idx] = 0
                    if peg_idx < len(self.peg_history):
                        self.peg_history[peg_idx].append(unmatched_positions[i])
                    else:
                        self.peg_history.append([unmatched_positions[i]])
        
        return [p for p in self.pegs if p is not None]

    def transform_pegs(pegs):
        """Transform 3D points from camera coordinates to world coordinates."""
        transformed_pegs = []
        for x, y in pegs:
            p = np.array([x, y, 1.0])
            p_transformed = transformation_matrix @ p
            transformed_pegs.append(p_transformed[:2])
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

def main():
    try:
        cap = setup_camera(camera_index=1)
        # Adjusted parameters for better fast movement tracking
        detector = PegTracker(
            movement_threshold=5,  # Lower threshold to detect movement earlier
            still_frames=10,       # Shorter still time to resume tracking faster
            max_speed=150          # Higher max speed for fast movements
        )
        
        # For FPS calculation
        prev_time = 0
        fps = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break   
            
            # Process frame
            pegs = detector.update(frame)
            
            # Draw pegs
            for i, (x, y) in enumerate(pegs):
                # Draw circle
                cv2.circle(frame, (int(x), int(y)), 8, (0, 255, 0), 2)
                # Draw peg number
                cv2.putText(frame, str(i+1), (int(x) - 5, int(y) + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Calculate and display FPS
            current_time = time.time()
            fps = 0.9 * fps + 0.1 * (1 / (current_time - prev_time)) if prev_time > 0 else 0
            prev_time = current_time
            
            # Display FPS and peg count
            cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'Pegs: {len(pegs)}/6', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Peg Detection', frame)

            message = {}

             # Transform to world coordinates
            try:
                transformed_pos = transform_pegs([(x, y, z)])[0]
                message[str(peg_id + 1)] = {
                    "x": round(float(transformed_pos[0]), 3),
                    "y": round(float(transformed_pos[1]), 3)
                }
            except Exception as e:
                print(f"Error transforming peg {peg_id}: {e}")
                # Send zero position if transformation fails
                message[str(peg_id + 1)] = {"x": 0, "y": 0}
        
            # Send message to Unity
            if message:
                try:
                    message_json = json.dumps(message)
                    output_sock.sendto(message_json.encode('utf-8'), (OUTPUT_IP, OUTPUT_PORT))
                except Exception as e:
                    print(f"Error sending message: {e}")
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

