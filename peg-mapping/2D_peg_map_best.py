import cv2
import numpy as np
import socket
import math
from ultralytics import YOLO
import time
import json

import os
# Set environment variable to handle OpenMP runtime warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Ports for receiving data from Unity
UNITY_LEFT_PORT = 9991
UNITY_RIGHT_PORT = 9992

# Port for sending data to Unity
UNITY_RECEIVE_PORT = 9989  # This should match the sendPort in Unity
OUTPUT_IP = "127.0.0.1"

# Socket for sending data to Unity
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
    def __init__(self, max_pegs=6, movement_threshold=5, still_frames=10, max_speed=150, max_occlusion_frames=30):
        self.max_pegs = max_pegs
        self.movement_threshold = movement_threshold
        self.still_frames = still_frames
        self.max_speed = max_speed
        self.max_occlusion_frames = max_occlusion_frames
        
        # Track pegs and their states
        self.peg_positions = [None] * max_pegs  # Current positions of pegs (None if not detected)
        self.peg_confidences = [0.0] * max_pegs  # Confidence scores for each peg
        self.frames_since_seen = [self.max_occlusion_frames] * max_pegs  # Frames since each peg was seen
        self.velocities = [(0, 0)] * max_pegs  # Current velocity (dx, dy) per peg
        self.moving_peg_idx = -1  # Index of the currently moving peg (-1 if none)

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
        # Get detections from YOLO
        results = model(frame, verbose=False)[0]
        
        # Get current detections with confidence scores
        current_detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            x = float((x1 + x2) / 2)
            y = float((y1 + y2) / 2)
            current_detections.append((x, y, conf))
        
        # Sort detections by confidence (highest first)
        current_detections.sort(key=lambda d: d[2], reverse=True)
        
        # If this is the first frame with detections, initialize all pegs
        if not any(self.peg_positions) and len(current_detections) >= self.max_pegs:
            for i in range(min(self.max_pegs, len(current_detections))):
                x, y, conf = current_detections[i]
                self.peg_positions[i] = (x, y)
                self.peg_confidences[i] = conf
                self.frames_since_seen[i] = 0
            return [p for p in self.peg_positions if p is not None]
        
        # Track which detections have been matched to existing pegs
        matched_detections = [False] * len(current_detections)
        
        # First pass: Update positions of pegs that are currently detected
        for i in range(self.max_pegs):
            if self.peg_positions[i] is None:
                continue
                
            best_match = None
            best_dist = float('inf')
            
            # Find the closest detection to this peg's last known position
            for j, (x, y, conf) in enumerate(current_detections):
                if matched_detections[j]:
                    continue
                    
                dist = self.distance(self.peg_positions[i], (x, y))
                
                # Only consider detections within max_speed distance
                if dist < best_dist and dist < self.max_speed * 2:
                    best_dist = dist
                    best_match = j
            
            if best_match is not None:
                # Update peg position and confidence
                x, y, conf = current_detections[best_match]
                self.peg_positions[i] = (x, y)
                self.peg_confidences[i] = conf
                self.frames_since_seen[i] = 0
                matched_detections[best_match] = True
            else:
                # Peg not detected in this frame
                self.frames_since_seen[i] += 1
        
        # Second pass: Handle unmatched detections (new pegs or false positives)
        for j, matched in enumerate(matched_detections):
            if not matched and current_detections[j][2] > 0.5:  # Only consider high confidence detections
                # Find the first empty peg slot
                for i in range(self.max_pegs):
                    if self.peg_positions[i] is None:
                        x, y, conf = current_detections[j]
                        self.peg_positions[i] = (x, y)
                        self.peg_confidences[i] = conf
                        self.frames_since_seen[i] = 0
                        break
        
        # Handle occlusions and missing pegs
        for i in range(self.max_pegs):
            if self.peg_positions[i] is not None and self.frames_since_seen[i] > self.max_occlusion_frames:
                # Peg has been occluded for too long, mark as missing
                self.peg_positions[i] = None
                self.peg_confidences[i] = 0.0
        
        # Return only the positions of currently visible pegs
        return [p for p in self.peg_positions if p is not None]

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

def transform_pegs(pegs):
        """Transform 3D points from camera coordinates to world coordinates."""
        transformed_pegs = []
        for x, y in pegs:
            p = np.array([x, y, 1.0])
            p_transformed = transformation_matrix @ p
            transformed_pegs.append(p_transformed[:2])
        return transformed_pegs

def main():
    try:
        # Initialize camera with 640x480 resolution (as detected by res_check.py)
        cap = setup_camera(camera_index=1, width=640, height=480)
        
        # Initialize the PegTracker with optimized parameters
        tracker = PegTracker(
            max_pegs=6,               # We're tracking exactly 6 pegs
            movement_threshold=3,      # Lower threshold for better movement detection
            still_frames=5,            # Fewer frames to confirm peg is stationary
            max_speed=200,             # Higher max speed for fast movements
            max_occlusion_frames=45    # Keep track of occluded pegs for 1.5 seconds at 30fps
        )
        
        # For FPS calculation
        prev_time = time.time()
        fps = 0
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Process frame
            pegs = tracker.update(frame)
            
            # # Draw pegs with visual feedback
            # for i, pos in enumerate(tracker.peg_positions):
            #     if pos is not None:
            #         x, y = pos
            #         # Draw circle (green for visible, blue for recently occluded)
            #         color = (0, 255, 0) if tracker.frames_since_seen[i] == 0 else (255, 165, 0)  # Green or orange
            #         cv2.circle(frame, (int(x), int(y)), 10, color, 2)
            #         # Draw peg number and confidence
            #         cv2.putText(frame, f"{i+1}", (int(x) - 5, int(y) + 5), 
            #                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            #         cv2.putText(frame, f"{tracker.peg_confidences[i]:.1f}", 
            #                    (int(x) + 10, int(y) + 10), 
            #                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # # Calculate and display FPS
            # current_time = time.time()
            # frame_count += 1
            # if frame_count % 10 == 0:  # Update FPS every 10 frames
            #     fps = 10 / (current_time - prev_time) if prev_time > 0 else 0
            #     prev_time = current_time
            
            # # Display FPS and peg count
            # visible_pegs = sum(1 for p in tracker.peg_positions if p is not None)
            # cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.putText(frame, f'Visible: {visible_pegs}/6', (10, 60), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # # Show the frame with detections
            # cv2.imshow('Peg Detection', frame)

            # Prepare message for Unity
            message = {}
            try:
                # Only send positions for pegs that are currently visible
                visible_pegs = [p for p in tracker.peg_positions if p is not None]
                if visible_pegs:
                    transformed = transform_pegs(visible_pegs)
                    for peg_id, pos in enumerate(transformed):
                        message[str(peg_id + 1)] = {
                            "x": round(float(pos[0]), 3),
                            "y": round(float(pos[1]), 3),
                        }
                
                # Also include occluded pegs with their last known positions
                for i, pos in enumerate(tracker.peg_positions):
                    if pos is None and tracker.frames_since_seen[i] < tracker.max_occlusion_frames:
                        message[str(i + 1)] = {
                            "x": round(float(tracker.peg_positions[i][0]), 3) if tracker.peg_positions[i] else 0,
                            "y": round(float(tracker.peg_positions[i][1]), 3) if tracker.peg_positions[i] else 0,
                        }
            
            except Exception as e:
                print(f"Error preparing message: {e}")
        
            # Send message to Unity
            if message:
                try:
                    message_json = json.dumps(message)
                    output_sock.sendto(message_json.encode('utf-8'), (OUTPUT_IP, UNITY_RECEIVE_PORT))
                    if frame_count % 10 == 0:  # Only print every 10 frames to reduce console spam
                        print(f"[Python] Sent data for {len([m for m in message.values()])} visible pegs")
                except Exception as e:
                    print(f"Error sending message: {e}")
            
            # Check for quit command
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC to quit
                print("Exiting...")
                break

    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

