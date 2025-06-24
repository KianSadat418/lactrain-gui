import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

def triangulate_best_peg_matches(
    left_points, right_points, 
    K1, D1, R1, P1,
    K2, D2, R2, P2
):
    """
    Triangulates best 6 peg matches from unrectified stereo 2D coordinates,
    using virtual rectification and ray intersection confidence.
    
    Inputs:
        - left_points, right_points: list of 6 (x, y) 2D image coordinates
        - K1, D1, R1, P1: Left camera intrinsics, distortion, rectification, projection
        - K2, D2, R2, P2: Right camera intrinsics, distortion, rectification, projection
        
    Output:
        - List of 6 matched (x, y, z) 3D peg positions
    """

    def get_camera_origin(P):
        _, _, _, _, _, _, origin = cv2.decomposeProjectionMatrix(P)
        origin = origin.flatten()
        return origin[:3] / origin[3] if origin.shape[0] == 4 else origin[:3]

    def ray_crossing_error(ray1, origin1, ray2, origin2):
        v1, v2 = ray1, ray2
        p1, p2 = origin1, origin2
        cross = np.cross(v1, v2)
        denom = np.linalg.norm(cross)
        if denom < 1e-6:
            return np.linalg.norm(np.cross(p2 - p1, v1)) / np.linalg.norm(v1)
        return abs(np.dot(p2 - p1, cross)) / denom

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

    # === Step 2: Get camera origins in 3D space ===
    origin1 = get_camera_origin(P1)
    origin2 = get_camera_origin(P2)

    # === Step 3: Build cost matrix ===
    cost_matrix = np.zeros((6, 6))
    triangulated_candidates = [[None]*6 for _ in range(6)]

    for i in range(6):
        ray1 = np.array([left_rect[i][0], left_rect[i][1], 1.0])
        ray1 /= np.linalg.norm(ray1)
        for j in range(6):
            ray2 = np.array([right_rect[j][0], right_rect[j][1], 1.0])
            ray2 /= np.linalg.norm(ray2)
            point_3d = triangulate_point(left_rect[i], right_rect[j])
            triangulated_candidates[i][j] = point_3d
            cost_matrix[i][j] = ray_crossing_error(ray1, origin1, ray2, origin2)

    # === Step 4: Solve best matching ===
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    best_3D_points = [triangulated_candidates[i][j] for i, j in zip(row_ind, col_ind)]

    return best_3D_points