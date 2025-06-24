import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

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
    triangulated_candidates = [[None]*6 for _ in range(6)]

    for i in range(6):
        ray1 = image_point_to_ray(left_rect[i])
        for j in range(6):
            ray2 = image_point_to_ray(right_rect[j])

            point_3d = triangulate_point(ray1, origin1, ray2, origin2)
            # Force Z-plane constraint
            point_3d_z0 = project_to_z_plane((point_3d - origin1), origin1, z_plane)
            triangulated_candidates[i][j] = point_3d_z0
            cost_matrix[i][j] = triangulation_error(ray1, origin1, ray2, origin2)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_points = [triangulated_candidates[i][j] for i, j in zip(row_ind, col_ind)]

    return matched_points