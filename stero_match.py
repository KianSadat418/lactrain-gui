import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def triangulate_best_peg_matches(
    left_points, right_points, 
    K1, D1, R1, P1,
    K2, D2, R2, P2,
    z_weight: float = 10.0
):
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

    # Rectify left and right peg coordinates
    left_rect = rectify_points(left_points, K1, D1, R1, P1)
    right_rect = rectify_points(right_points, K2, D2, R2, P2)

    origin1 = get_camera_origin(P1)
    origin2 = get_camera_origin(P2)

    triangulated_candidates = [[None for _ in range(6)] for _ in range(6)]
    cost_matrix = np.zeros((6, 6))

    for i in range(6):
        lx, ly = left_rect[i]
        for j in range(6):
            rx, ry = right_rect[j]

            # Align Y (pseudo-rectification)
            avg_y = (ly + ry) / 2
            pt_left = [lx, avg_y]
            pt_right = [rx, avg_y]

            # Reject negative disparities (right should be to the left of left)
            if lx - rx <= 0:
                cost_matrix[i][j] = 1e6
                continue

            # Triangulate
            point_3d = triangulate_point(pt_left, pt_right)
            triangulated_candidates[i][j] = point_3d

            # Ray direction (in normalized rectified space)
            ray1 = np.array([lx, avg_y, 1.0])
            ray2 = np.array([rx, avg_y, 1.0])
            ray1 /= np.linalg.norm(ray1)
            ray2 /= np.linalg.norm(ray2)

            ray_error = ray_crossing_error(ray1, origin1, ray2, origin2)
            z_penalty = abs(point_3d[2]) * z_weight  # Push Z values toward 0 plane
            cost_matrix[i][j] = ray_error + z_penalty

    # Normalize cost matrix
    cost_matrix = (cost_matrix - np.min(cost_matrix)) / (np.max(cost_matrix) - np.min(cost_matrix) + 1e-8)

    # Hungarian assignment for best match
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_3D_points = [triangulated_candidates[i][j] for i, j in zip(row_ind, col_ind)]

    return matched_3D_points

def plot_3d_pegs(points_3d, title="Triangulated Pegs"):
    """
    Plots a list of 3D points using matplotlib.

    Args:
        points_3d: List of 3D (x, y, z) coordinates.
        title: Title of the plot window.
    """
    points_3d = np.array(points_3d)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='b', s=60)

    for i, (x, y, z) in enumerate(points_3d):
        ax.text(x, y, z, f'{i}', fontsize=10, color='red')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.grid(True)
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.show()