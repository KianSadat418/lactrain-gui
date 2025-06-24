import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

def triangulate_best_peg_matches(left_points, right_points, P1, P2):
    """
    Given 6 2D points from left and right stereo images, returns the 6 triangulated 3D peg positions
    matched using ray crossing error.
    
    Args:
        left_points (list of (x, y)): List of 6 points from the left image.
        right_points (list of (x, y)): List of 6 points from the right image.
        P1, P2: Projection matrices for the left and right cameras.

    Returns:
        List of 6 (x, y, z) triangulated 3D points.
    """

    def image_point_to_ray(image_pt, P):
        """Back-project 2D point into normalized camera space as a ray."""
        x, y = image_pt
        pt_h = np.array([x, y, 1.0])
        K, _ = cv2.decomposeProjectionMatrix(P)[:2]
        ray = np.linalg.inv(K) @ pt_h
        return ray / np.linalg.norm(ray)

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
    _, _, _, _, _, _, origin1 = cv2.decomposeProjectionMatrix(P1)
    _, _, _, _, _, _, origin2 = cv2.decomposeProjectionMatrix(P2)
    origin1 = origin1[:3].flatten() / origin1[3]
    origin2 = origin2[:3].flatten() / origin2[3]

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
