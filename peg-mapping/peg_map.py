import cv2
import numpy as np
import socket
import json
import matplotlib.pyplot as plt
import os
from datetime import datetime
from ultralytics import YOLO
from peg import PegTracker, Triangulator

# ---- CONFIG ----
MODEL_PATH = "Assets/Scripts/Peg-Detection-Scripts/Training-05-22/result/content/runs/detect/yolo8_peg_detector/weights/best.pt"
CALIB_PATH = "Assets/Scripts/Camera Calibration/stereo_camera_calibration2.npz"
TRANSFORM_PATH = "Assets/Scripts/Camera Calibration/TransformationMatrix.npz"

UNITY_PORT = 9989
UNITY_IP = "127.0.0.1"

WIDTH = 1600
HEIGHT = 600
CHUNK_SIZE = 30000
image_size = (800, 600)

# ---- INIT ----
model = YOLO(MODEL_PATH)
calib = np.load(CALIB_PATH)
transform = np.load(TRANSFORM_PATH)["affine_transform_ransac"]

K1, D1 = calib["cameraMatrixL"], calib["distL"]
K2, D2 = calib["cameraMatrixR"], calib["distR"]
R, T = calib["R"], calib["T"]

triangulator = Triangulator(K1, D1, K2, D2, R, T, transform, image_size)
tracker = PegTracker()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

frame_id = 0
init_ready = False

def get_yolo_centers(results):
    if len(results.boxes) == 0:
        return []
    centers = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        centers.append([(x1 + x2) / 2, (y1 + y2) / 2])
    return centers

def send_to_unity(message_dict):
    try:
        message_json = json.dumps(message_dict)
        sock.sendto(message_json.encode('utf-8'), (UNITY_IP, UNITY_PORT))
    except Exception as e:
        print(f"[ERROR] Sending to Unity failed: {e}")

def greedy_match(left_pts, right_pts):
    result_left, result_right = [], []
    for i in range(len(left_pts)):
        for j in range(len(right_pts)):
            if right_pts[j] in result_right or left_pts[i] in result_left:
                continue
            if abs(left_pts[i][1] - right_pts[j][1]) > 15:
                continue
            x, y, z = triangulator.triangulate_points([left_pts[i]], [right_pts[j]])[0]
            if 0.0 < z < 200.0:
                result_left.append(left_pts[i])
                result_right.append(right_pts[j])
                break
    return result_left, result_right

def save_peg_summary(peg: Peg, folder="peg_logs"):
    os.makedirs(folder, exist_ok=True)
    peg_id = peg.id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{folder}/peg{peg_id}_{timestamp}"

    # --- Velocity plot ---
    plt.figure()
    plt.plot(peg.velocity_log, label="Speed (mm/frame)")
    plt.xlabel("Frame")
    plt.ylabel("Velocity")
    plt.title(f"Peg {peg_id} Velocity Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{prefix}_velocity.png")
    plt.close()

    # --- 3D trajectory plot ---
    positions = np.array(peg.position_log)
    if len(positions) > 0:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker='o', linewidth=2)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.set_title(f"Peg {peg_id} 3D Trajectory")
        plt.savefig(f"{prefix}_trajectory3D.png")
        plt.close()

    # --- Stats summary ---
    total_distance = np.sum(peg.velocity_log)
    peak_speed = np.max(peg.velocity_log) if peg.velocity_log else 0
    avg_speed = np.mean(peg.velocity_log) if peg.velocity_log else 0
    duration = len(peg.velocity_log)

    stats = {
        "peg_id": peg_id,
        "frames_moved": duration,
        "avg_speed": round(avg_speed, 3),
        "peak_speed": round(peak_speed, 3),
        "total_distance": round(total_distance, 3)
    }

    with open(f"{prefix}_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[LOG] Saved peg {peg_id} stats and visualizations to {prefix}_*.png/.json")

# ---- MAIN LOOP ----
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        left_raw = frame[:, :800]
        right_raw = frame[:, 800:]
        left, right = triangulator.stereo_remap(left_raw, right_raw)

        res_l = model(left, verbose=False)[0]
        res_r = model(right, verbose=False)[0]
        det_l = get_yolo_centers(res_l)
        det_r = get_yolo_centers(res_r)

        if len(det_l) >= 6 and len(det_r) >= 6:
            l_matched, r_matched = greedy_match(det_l, det_r)

            if len(l_matched) == len(r_matched) and len(l_matched) >= 6:
                pts3d = triangulator.triangulate_points(l_matched[:6], r_matched[:6])

                if not tracker.initialized:
                    tracker.initialize(pts3d)
                else:
                    tracker.update(pts3d)

                    # Handle movement finalization
                    if tracker.movement_phase:
                        input("[INPUT] Press Enter to confirm peg movement is done...")
                        # Save visuals/stats for the peg that just finished moving
                        peg_id = tracker.get_moving_peg_id()
                        if peg_id is not None and peg_id in tracker.pegs:
                            save_peg_summary(tracker.pegs[peg_id])

                        tracker.finalize_movement()

                    # Send to Unity
                    peg_positions = tracker.get_peg_positions()
                    transformed = triangulator.transform_to_world([peg_positions[i] for i in range(6)])

                    msg = {}
                    for i, (x, y, z) in enumerate(transformed):
                        msg[str(i+1)] = {
                            "x": round(x, 3),
                            "y": round(y, 3),
                            "z": round(z, 3)
                        }

                    moving_id = tracker.get_moving_peg_id()
                    if moving_id is not None:
                        msg["moving_peg"] = str(moving_id)

                    send_to_unity(msg)
                    print(f"[SEND] Frame {frame_id} | Moving peg: {moving_id}")
            else:
                print(f"[WARN] Could not match 6 pegs: L={len(det_l)}, R={len(det_r)}")
        else:
            print(f"[WARN] Incomplete detections: L={len(det_l)}, R={len(det_r)}")

        frame_id += 1

except KeyboardInterrupt:
    print("Exiting cleanly.")
finally:
    cap.release()
    sock.close()
    cv2.destroyAllWindows()