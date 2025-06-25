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


model = YOLO("Assets/Scripts/Peg-Detection-Scripts/Training-05-22/result/content/runs/detect/yolo8_peg_detector/weights/best.pt")
calib = np.load("Assets/Scripts/Camera Calibration/stereo_camera_calibration2.npz")
#transformation_matrix = np.load("Assets/Scripts/Camera Calibration/TransformationMatrix.npz")
K1, D1 = calib["cameraMatrixL"], calib["distL"]
K2, D2 = calib["cameraMatrixR"], calib["distR"]
R, T = calib["R"], calib["T"]

baseline_offset = T[0][0] / 2.0  # Half the translation between cameras (baseline)

image_size = (800, 600)

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T, alpha=0, flags=cv2.CALIB_ZERO_DISPARITY)

OUTPUT_PORT = 9991
OUTPUT_IP = "127.0.0.1"

output_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

def triangulate_best_peg_matches(
    left_points, right_points, 
    K1, D1, R1, P1,
    K2, D2, R2, P2,
    z_plane: float = 0.0
):
    def get_camera_origin(P):
        _, _, _, _, _, _, origin = cv2.decomposeProjectionMatrix(P)
        origin = origin.flatten()
        return origin[:3] / origin[3] if origin.shape[0] == 4 else origin[:3]

    def rectify_points(pts, K, D, R, P):
        pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        rectified = cv2.undistortPoints(pts, K, D, R=R, P=P)
        return rectified.reshape(-1, 2)

    def image_point_to_ray(image_pt):
        x, y = image_pt
        ray = np.array([x, y, 1.0])
        return ray / np.linalg.norm(ray)

    def triangulate_point(ray1, origin1, ray2, origin2):
        v1 = ray1
        v2 = ray2
        p1 = origin1
        p2 = origin2
        A = np.stack([v1, -v2], axis=1)
        b = p2 - p1
        t = np.linalg.lstsq(A, b, rcond=None)[0]
        pt1 = p1 + t[0] * v1
        pt2 = p2 + t[1] * v2
        midpoint = (pt1 + pt2) / 2
        return midpoint

    def project_to_z_plane(ray, origin, z_plane):
        if not np.isfinite(ray[2]) or abs(ray[2]) < 1e-6:
            return np.full(3, np.nan)  # Return [nan, nan, nan] if vertical component is zero
        t = (z_plane - origin[2]) / ray[2]
        return origin + t * ray

    def triangulation_error(ray1, origin1, ray2, origin2):
        cross = np.cross(ray1, ray2)
        denom = np.linalg.norm(cross)
        separation = abs(np.dot(origin2 - origin1, cross)) / denom if denom > 1e-6 else 1e6
        parallel_penalty = 1 / (denom + 1e-6)
        return separation + 10 * parallel_penalty

    # === Rectify points ===
    left_rect = rectify_points(left_points, K1, D1, R1, P1)
    right_rect = rectify_points(right_points, K2, D2, R2, P2)

    origin1 = get_camera_origin(P1)
    origin2 = get_camera_origin(P2)

    cost_matrix = np.zeros((6, 6))
    triangulated_candidates: list[list[Union[np.ndarray, None]]] = [[None]*6 for _ in range(6)]

    for i in range(6):
        ray1 = image_point_to_ray(left_rect[i])
        for j in range(6):
            ray2 = image_point_to_ray(right_rect[j])

            point_3d = triangulate_point(ray1, origin1, ray2, origin2)
            # Force Z-plane constraint
            point_3d_z0 = project_to_z_plane((point_3d - origin1), origin1, z_plane)

            point_3d_z0 = project_to_z_plane((point_3d - origin1), origin1, z_plane)
            if not np.all(np.isfinite(point_3d_z0)):
                cost_matrix[i][j] = 1e6  # Penalize invalid solutions
            else:
                triangulated_candidates[i][j] = point_3d_z0
                cost_matrix[i][j] = triangulation_error(ray1, origin1, ray2, origin2)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_points = [triangulated_candidates[i][j] for i, j in zip(row_ind, col_ind)]

    return matched_points

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

def find_pairs(left_points, right_points):

    all_pairings = [list(zip(left_points, perm)) for perm in itertools.permutations(right_points)]

    best_pair = []
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

        best_pair.append(pairings)
    

    return best_pair



def trasformed_pegs(pegs):
    """Transform 3D points from camera coordinates to world coordinates using the transformation matrix."""
    transformedPegs = []

    for i, (x, y, z) in enumerate(pegs):
        p = np.array([x, y, z, 1.0])
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

        # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

        # for i in range(6):
        #     pt_left = (int(left_frame_peg[i][0]), int(left_frame_peg[i][1]))
        #     pt_right = (int(right_frame_peg[i][0]), int(right_frame_peg[i][1]))
        #     cv2.circle(left_frame, pt_left, 5, colors[i], -1)
        #     cv2.circle(right_frame, pt_right, 5, colors[i], -1)

        # def draw_horizontal_line(img, interval=40):
        #     for y in range(0, img.shape[0], interval):
        #         cv2.line(img, (0, y), (img.shape[1], y), (0, 255, 0), 1)
            
        #     return img
        
        # left_frame = draw_horizontal_line(left_frame)
        # right_frame = draw_horizontal_line(right_frame)
        # cv2.imshow("Left Frame Pegs", left_frame)
        # cv2.imshow("Right Frame Pegs", right_frame)
        # cv2.waitKey(1)

        #print(f"Left Frame Pegs: {left_frame_peg}")
        #print(f"Right Frame Pegs: {right_frame_peg}")

        if len(left_frame_peg) != 6 or len(right_frame_peg) != 6:
            continue

        #print(cv2.decomposeProjectionMatrix(P1).shape)

        if left_frame_peg is None or right_frame_peg is None:
            continue

        # pegsInCamera = find_3D_points(left_frame_peg, right_frame_peg)
        # if not pegsInCamera:
        #     continue
        
        # print(f"Pegs in Camera Coordinates: {pegsInCamera}")

        result = []
        pairs = find_pairs(left_frame_peg, right_frame_peg)
        print(f"Pairs: {len(pairs)}")
        for i in pairs:
            l_p = []
            r_p = []
            for j in i:
                l_p.append(j[0])
                r_p.append(j[1])

            result.append(find_3D_points(l_p, r_p))
        
        break

        # result = triangulate_best_peg_matches(left_frame_peg, right_frame_peg, K1, D1, R1, P1, K2, D2, R2, P2)
        # break

        # # pegsInHololens = trasformed_pegs(pegsInCamera)

        # # for i, (x, y, z) in enumerate(pegsInHololens):
        # #     message = {"X": round(x, 2), "Y": round(y, 2), "Z": round(z, 2)}

        # # message = json.dumps(message)
        # # output_socket.sendto(message.encode('utf-8'), (OUTPUT_IP, OUTPUT_PORT))
        # # os.system('cls' if os.name == 'nt' else 'clear')
        # # print(f"📤 Sent: {x:.2f}, {y:.2f}, {z:.2f}")
    
    for i in result:
        for j in i:
            print(j)
        plot_3d_pegs(i, title="Triangulated Pegs from Best Matches")
        print("========================================")
