import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

def triangulate_best_peg_matches(
    left_points, right_points, 
    K1, D1, R1, P1,
    K2, D2, R2, P2
):
    """
    Robust triangulation and peg matching with rectified YOLO center points.
    """

    def get_camera_origin(P):
        _, _, _, _, _, _, origin = cv2.decomposeProjectionMatrix(P)
        origin = origin.flatten()
        return origin[:3] / origin[3] if origin.shape[0] == 4 else origin[:3]

    def ray_crossing_error(ray1, origin1, ray2, origin2):
        cross = np.cross(ray1, ray2)
        denom = np.linalg.norm(cross)
        if denom < 1e-6:
            return np.linalg.norm(np.cross(origin2 - origin1, ray1)) / np.linalg.norm(ray1)
        return abs(np.dot(origin2 - origin1, cross)) / denom

    def triangulate_point(pt1, pt2):
        pt1 = np.array(pt1, dtype=np.float32).reshape(2, 1)
        pt2 = np.array(pt2, dtype=np.float32).reshape(2, 1)
        point_4d = cv2.triangulatePoints(P1, P2, pt1, pt2)
        return (point_4d[:3] / point_4d[3]).flatten()

    def rectify_points(pts, K, D, R, P):
        pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        rectified = cv2.undistortPoints(pts, K, D, R=R, P=P)
        return rectified.reshape(-1, 2)

    # === Step 1: rectify 2D input points ===
    left_rect = rectify_points(left_points, K1, D1, R1, P1)
    right_rect = rectify_points(right_points, K2, D2, R2, P2)

    # === Step 2: Get camera origins ===
    origin1 = get_camera_origin(P1)
    origin2 = get_camera_origin(P2)

    # === Step 3: Build cost matrix ===
    cost_matrix = np.zeros((6, 6))
    triangulated_candidates = [[None]*6 for _ in range(6)]

    for i in range(6):
        lx, ly = left_rect[i]
        for j in range(6):
            rx, ry = right_rect[j]

            # === Force y-alignment for rectified geometry ===
            avg_y = (ly + ry) / 2.0
            left_aligned = [lx, avg_y]
            right_aligned = [rx, avg_y]

            # === Optional: skip bad disparities ===
            disparity = lx - rx
            if disparity <= 0:
                cost_matrix[i][j] = 1e6
                continue

            # === Triangulate 3D point ===
            pt3d = triangulate_point(left_aligned, right_aligned)
            triangulated_candidates[i][j] = pt3d

            # === Cost: hybrid between ray crossing and z magnitude penalty ===
            ray1 = np.array([lx, avg_y, 1.0])
            ray2 = np.array([rx, avg_y, 1.0])
            ray1 /= np.linalg.norm(ray1)
            ray2 /= np.linalg.norm(ray2)

            ray_error = ray_crossing_error(ray1, origin1, ray2, origin2)
            z_penalty = abs(pt3d[2]) if pt3d[2] < 0 else 0  # discourage negative Z

            cost_matrix[i][j] = ray_error + z_penalty * 0.1  # you can tune weight here

    # === Step 4: Match best pairs ===
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    best_3D_points = [triangulated_candidates[i][j] for i, j in zip(row_ind, col_ind)]

    return best_3D_points