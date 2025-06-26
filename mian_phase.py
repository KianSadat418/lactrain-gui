import cv2
import numpy as np
import socket
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Enable OpenEXR support in OpenCV
from ultralytics import YOLO
import json
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Union
import itertools
import struct


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

pegs = {str(i): ([0, 0], [0, 0], [0, 0, 0]) for i in range(1, 7)}
isInitialized = False

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

def plot_3d_pegs(points_3d, title="Triangulated Pegs"):
    """
    Plots a list of 3D points using matplotlib.

    Args:
        points_3d: List of 3D (x, y, z) coordinates.
        title: Title of the plot window.
    """
    points_3d = np.array(points_3d)

    # Filter only finite points
    valid_mask = np.all(np.isfinite(points_3d), axis=1)
    valid_points = points_3d[valid_mask]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2], c='b', s=60)

    for i, (x, y, z) in enumerate(valid_points):
        ax.text(x, y, z, f'{i}', fontsize=10, color='red')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.grid(True)
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.show()

def find_pairs2(left_points, right_points):
    result_pairs_right = []
    result_pairs_left = []
    for i in range(len(left_points)):
        for j in range(len(right_points)):
            if right_points[j] in result_pairs_right or left_points[i] in result_pairs_left:
                continue

            left_x = left_points[i]
            right_x = right_points[j]
            #print(left_x, right_x)
            if abs(left_x[1] - right_x[1]) > 10.0:
                continue

            pointIn3D = find_3D_points([left_x], [right_x])[0]
            if pointIn3D[2] > 200.0 or pointIn3D[2] < 0.0:
                continue

            result_pairs_left.append(left_x)
            result_pairs_right.append(right_x)
            print(left_x, right_x, pointIn3D)


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

def trasform_pegs(pegs):
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
        return None 
    
    pegs = []
    for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
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

def initialize_pegs(left_frame_peg, right_frame_peg, pegsIn3D):
    for i in range(6):
        pegs[str(i+1)] = (left_frame_peg[i], right_frame_peg[i], pegsIn3D[i])

def pegTracking(left_frame_peg, right_frame_peg):
    isFound = [False for _ in range(6)]
    isUsedLeft = [False for _ in range(len(left_frame_peg))]
    isUsedRight = [False for _ in range(len(right_frame_peg))]

    threshold2D = 15.0
    threshold3D = 30.0
    for i in range(6):
        prev_left, prev_right, prev_3d = pegs[str(i+1)]
        min_dist = float('inf')
        best_left = None
        best_right = None
        for lp in left_frame_peg:
            if isUsedLeft[left_frame_peg.index(lp)]:
                continue
            dist = np.linalg.norm(np.array(lp) - np.array(prev_left))
            if dist < threshold2D:
                best_left = lp

        for rp in right_frame_peg:
            if isUsedRight[right_frame_peg.index(rp)]:
                continue
            dist = np.linalg.norm(np.array(rp) - np.array(prev_right))
            if dist < threshold2D:
                best_right = rp

        if best_left is not None and best_right is not None:
            pegin3D = find_3D_points([best_left], [best_right])[0]
            if np.linalg.norm(np.array(pegin3D) - np.array(prev_3d)) < threshold3D:
                isUsedLeft[left_frame_peg.index(best_left)] = True
                isUsedRight[right_frame_peg.index(best_right)] = True
                isFound[i] = True
                pegs[str(i+1)] = (best_left, best_right, pegin3D)

if __name__ == "__main__":

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        left_frame = frame[:, :800]
        right_frame = frame[:, 800:]

        map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

        left_frame = cv2.remap(left_frame, map1x, map1y, cv2.INTER_LINEAR)
        right_frame = cv2.remap(right_frame, map2x, map2y, cv2.INTER_LINEAR)

        send_frame_left(left_frame, count)
        send_frame_right(right_frame, count)

        count += 1

        # def draw_horizontal_line(img, interval=40):
        #     for y in range(0, img.shape[0], interval):
        #         cv2.line(img, (0, y), (img.shape[1], y), (0, 255, 0), 1)
            
        #     return img
        
        # debug_left = draw_horizontal_line(left_frame.copy())
        # debug_right = draw_horizontal_line(right_frame.copy())

        # cv2.imshow("Left Frame", debug_left)
        # cv2.imshow("Right Frame", debug_right)
        # cv2.waitKey(1)


        left_frame_peg, right_frame_peg = res_without_rectify(right_frame, left_frame)

        if isInitialized:
            pegTracking(left_frame_peg, right_frame_peg)
            pegsInCamera = [pegs[str(i)][2] for i in range(1, 7)]



        if not isInitialized:
            if len(left_frame_peg) != 6 or len(right_frame_peg) != 6:
                continue
            
            left_frame_peg, right_frame_peg = find_pairs(left_frame_peg, right_frame_peg)
            pegsInCamera = find_3D_points(left_frame_peg, right_frame_peg)
            initialize_pegs(left_frame_peg, right_frame_peg, pegsInCamera)
            isInitialized = True

        pegsInHololens = trasform_pegs(pegsInCamera)
        message = {}
        for i, (x, y, z) in enumerate(pegsInHololens):
            message[str(i+1)] = {
                "x": round(x, 3),
                "y": round(y, 3),
                "z": round(z, 3)
            }

        message = json.dumps(message)
        output_socket.sendto(message.encode('utf-8'), (OUTPUT_IP, OUTPUT_PORT))
