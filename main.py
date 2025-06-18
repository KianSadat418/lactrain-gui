import cv2
import numpy as np
import os
import socket
import json
import struct

OUTPUT_PORT = 9990
UNITY_LEFT_PORT = 9998
UNITY_RIGHT_PORT = 9999
OUTPUT_IP = "127.0.0.1"

left_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
right_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
output_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

image_size = (800, 600)

# === Load stereo calibration ===
calib = np.load("Assets/Scripts/Camera Calibration/stereo_camera_calibration2.npz")
K1, D1 = calib["cameraMatrixL"], calib["distL"]
K2, D2 = calib["cameraMatrixR"], calib["distR"]
R, T = calib["R"], calib["T"]

CHUNK_SIZE = 30000
count = 0

SELECTED_MARKERS = [1, 3, 5, 7, 9, 10, 12, 14, 16, 18, 19, 21, 23, 25, 27, 31, 33, 35, 37, 39]

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

    
def triangulate_3D(P1, P2, ptL, ptR):
    pt_4d = cv2.triangulatePoints(P1, P2, ptL, ptR)
    pt_3d = pt_4d[:3] / pt_4d[3]
    return pt_3d.ravel()

def find_checkered_pattern(right_frame, left_frame):
    markersIn3D = {}
    all_markers_count = 0
    Selected_markers_count = 0

    gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    retL3, cornersL3 = cv2.findChessboardCorners(gray_left, (3, 6), None)
    retR3, cornersR3 = cv2.findChessboardCorners(gray_right, (3, 6), None)
    retL4, cornersL4 = cv2.findChessboardCorners(gray_left, (3, 7), None)
    retR4, cornersR4 = cv2.findChessboardCorners(gray_right, (3, 7), None)
    
    if not retR4 or not retL4 or not retR3 or not retL3:
        return left_frame, right_frame, markersIn3D
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cornersL3 = cv2.cornerSubPix(gray_left, cornersL3, (11, 11), (-1, -1), criteria)
    cornersR3 = cv2.cornerSubPix(gray_right, cornersR3, (11, 11), (-1, -1), criteria)
    cornersL4 = cv2.cornerSubPix(gray_left, cornersL4, (11, 11), (-1, -1), criteria)
    cornersR4 = cv2.cornerSubPix(gray_right, cornersR4, (11, 11), (-1, -1), criteria)

    marker_size = 10

    # === Stereo rectify intrinsics (no rectification needed)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

    for _, (ptL, ptR) in enumerate(zip(cornersL3, cornersR3)):
        all_markers_count += 1
        if all_markers_count not in SELECTED_MARKERS:
            continue
        Selected_markers_count += 1

        markersIn3D[all_markers_count] = np.array(triangulate_3D(P1, P2, ptL.reshape(2, 1), ptR.reshape(2, 1)))

        ptL = tuple(ptL.ravel().astype(int))
        ptR = tuple(ptR.ravel().astype(int))

        center = (int(ptL[0]), int(ptL[1]))
        cv2.line(left_frame, (center[0] - marker_size, center[1]), (center[0] + marker_size, center[1]), (0, 255, 0), 2)
        cv2.line(left_frame, (center[0], center[1] - marker_size), (center[0], center[1] + marker_size), (0, 255, 0), 2)
        cv2.putText(left_frame, str(Selected_markers_count), (center[0] + 5, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        center = (int(ptR[0]), int(ptR[1]))
        cv2.line(right_frame, (center[0] - marker_size, center[1]), (center[0] + marker_size, center[1]), (0, 255, 0), 2)
        cv2.line(right_frame, (center[0], center[1] - marker_size), (center[0], center[1] + marker_size), (0, 255, 0), 2)
        cv2.putText(right_frame, str(Selected_markers_count), (center[0] + 5, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


    for _, (ptL, ptR) in enumerate(zip(cornersL4, cornersR4)):
        all_markers_count += 1
        if all_markers_count not in SELECTED_MARKERS:  
            continue
        Selected_markers_count += 1

        markersIn3D[all_markers_count] = np.array(triangulate_3D(P1, P2, ptL.reshape(2, 1), ptR.reshape(2, 1)))

        ptL = tuple(ptL.ravel().astype(int))
        ptR = tuple(ptR.ravel().astype(int))

        center = (int(ptL[0]), int(ptL[1]))
        cv2.line(left_frame, (center[0] - marker_size, center[1]), (center[0] + marker_size, center[1]), (0, 255, 0), 2)
        cv2.line(left_frame, (center[0], center[1] - marker_size), (center[0], center[1] + marker_size), (0, 255, 0), 2)
        cv2.putText(left_frame, str(Selected_markers_count), (center[0] + 5, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        center = (int(ptR[0]), int(ptR[1]))
        cv2.line(right_frame, (center[0] - marker_size, center[1]), (center[0] + marker_size, center[1]), (0, 255, 0), 2)
        cv2.line(right_frame, (center[0], center[1] - marker_size), (center[0], center[1] + marker_size), (0, 255, 0), 2)
        cv2.putText(right_frame, str(Selected_markers_count), (center[0] + 5, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return left_frame, right_frame, markersIn3D

while True:
    ret, frame = cap.read()
    if not ret:
        break

    left_frame = frame[:, :800]
    right_frame = frame[:, 800:]

    left_frame, right_frame, markersIn3D = find_checkered_pattern(right_frame, left_frame)

    send_frame_left(left_frame, count)
    send_frame_right(right_frame, count)

    count += 1
    print(f"📤 Frames {count} Sent, length: {len(left_frame)} bytes")    
    os.system('cls' if os.name == 'nt' else 'clear')


    if len(markersIn3D) > 0:
        cleaned = {k: v.tolist() for k, v in markersIn3D.items()}
        allPoints = json.dumps(cleaned)
        output_socket.sendto(allPoints.encode('utf-8'), (OUTPUT_IP, OUTPUT_PORT))
        print(f"{len(markersIn3D)} 3D Points Sent")


    # === Show annotated views ===
    combined = np.hstack((left_frame, right_frame))
    cv2.imshow("Annotated Corners", combined)
    cv2.waitKey(1)

    # if len(markersIn3D) > 0:
    #     # === Compute distance
    #     index1, index2 = 10, 11
    #     P3D_1 = markersIn3D.get(index1)
    #     P3D_2 = markersIn3D.get(index2)
    #     distance_mm = np.linalg.norm(P3D_1 - P3D_2)
    #     os.system('cls' if os.name == 'nt' else 'clear')
    #     print(f"Distance (mm): {distance_mm:.2f}")

 
