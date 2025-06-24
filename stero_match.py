import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

def triangulate_best_peg_matches(left_points, right_points, P1, P2):

    def image_point_to_ray(image_pt, P):
        """Back-project 2D point into normalized camera space as a ray."""
        x, y = image_pt
        pt_h = np.array([x, y, 1.0])
        K, _ = cv2.decomposeProjectionMatrix(P)[:2]
        ray = np.linalg.inv(K) @ pt_h
        return ray / np.linalg.norm(ray)
    
    def get_camera_origin(P):
        _, _, _, _, _, _, origin = cv2.decomposeProjectionMatrix(P)
        origin = origin.flatten()
        if origin.shape[0] == 4:
            return origin[:3] / origin[3]
        else:
            return origin[:3]

    def triangulate_point(P1, P2, pt1, pt2):
        """Use OpenCV's triangulatePoints to get 3D point from 2D matches."""
        pt1 = np.array(pt1, dtype=np.float32).reshape(2, 1)
        pt2 = np.array(pt2, dtype=np.float32).reshape(2, 1)
        point_4d = cv2.triangulatePoints(P1, P2, pt1, pt2)
        point_3d = (point_4d[:3] / point_4d[3]).flatten()
        return point_3d

    def ray_crossing_error(ray1, origin1, ray2, origin2):
        """Compute shortest distance between two skew rays."""
        v1, v2 = ray1, ray2
        p1, p2 = origin1, origin2
        cross = np.cross(v1, v2)
        denom = np.linalg.norm(cross)
        if denom < 1e-6:  # nearly parallel rays
            return np.linalg.norm(np.cross(p2 - p1, v1)) / np.linalg.norm(v1)
        return abs(np.dot((p2 - p1), cross)) / denom

    # Get camera origins from projection matrices
    origin1 = get_camera_origin(P1)
    origin2 = get_camera_origin(P2)

    # Build cost matrix and triangulated candidate array
    cost_matrix = np.zeros((6, 6))
    triangulated_candidates = [[None]*6 for _ in range(6)]

    for i in range(6):
        ray_left = image_point_to_ray(left_points[i], P1)
        for j in range(6):
            ray_right = image_point_to_ray(right_points[j], P2)
            point_3d = triangulate_point(P1, P2, left_points[i], right_points[j])
            triangulated_candidates[i][j] = point_3d
            cost_matrix[i][j] = ray_crossing_error(ray_left, origin1, ray_right, origin2)

    # Solve for optimal left-right peg match
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    best_3D_points = [triangulated_candidates[i][j] for i, j in zip(row_ind, col_ind)]

    return best_3D_points
