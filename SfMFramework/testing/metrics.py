"""Evaluation metrics for feature matching."""

from typing import Dict, Tuple, Optional
import numpy as np
import cv2


def evaluate_matches(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    matches: np.ndarray,
    K0: np.ndarray,
    K1: np.ndarray,
    T_0to1: np.ndarray,
    depth0: Optional[np.ndarray] = None,
    depth1: Optional[np.ndarray] = None,
    epipolar_threshold: float = 1.0,
    depth_threshold: float = 0.1,
) -> Dict[str, float]:
    """Evaluate match quality against ground truth.

    Args:
        kpts0: Keypoints in image 0 [N, 2]
        kpts1: Keypoints in image 1 [M, 2]
        matches: Match indices [K, 2] (indices into kpts0 and kpts1)
        K0: Camera intrinsics for image 0 [3, 3]
        K1: Camera intrinsics for image 1 [3, 3]
        T_0to1: Ground truth pose from image 0 to 1 [4, 4]
        depth0: Depth map for image 0 (optional)
        depth1: Depth map for image 1 (optional)
        epipolar_threshold: Epipolar error threshold in pixels
        depth_threshold: Depth reprojection error threshold in meters

    Returns:
        Dictionary of metrics:
            - num_matches: Number of matches
            - epipolar_error: Mean epipolar error (pixels)
            - inlier_ratio: Ratio of matches within epipolar threshold
            - depth_error: Mean depth reprojection error (if depth available)
    """
    metrics = {}

    if len(matches) == 0:
        return {
            "num_matches": 0,
            "epipolar_error": float("inf"),
            "inlier_ratio": 0.0,
        }

    # Get matched keypoints
    mkpts0 = kpts0[matches[:, 0]]
    mkpts1 = kpts1[matches[:, 1]]

    metrics["num_matches"] = len(matches)

    # Compute essential matrix from ground truth
    R = T_0to1[:3, :3]
    t = T_0to1[:3, 3]
    t_skew = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])
    E = t_skew @ R

    # Fundamental matrix
    F = np.linalg.inv(K1.T) @ E @ np.linalg.inv(K0)

    # Compute epipolar errors
    epipolar_errors = compute_epipolar_errors(mkpts0, mkpts1, F)
    metrics["epipolar_error"] = float(np.mean(epipolar_errors))
    metrics["median_epipolar_error"] = float(np.median(epipolar_errors))
    metrics["inlier_ratio"] = float(np.mean(epipolar_errors < epipolar_threshold))

    # Depth-based evaluation (if available)
    if depth0 is not None and depth1 is not None:
        depth_errors = compute_depth_reprojection_errors(
            mkpts0, mkpts1, matches, depth0, depth1, K0, K1, T_0to1
        )
        if depth_errors is not None and len(depth_errors) > 0:
            metrics["depth_error"] = float(np.mean(depth_errors))
            metrics["depth_inlier_ratio"] = float(np.mean(depth_errors < depth_threshold))

    return metrics


def compute_epipolar_errors(
    pts0: np.ndarray,
    pts1: np.ndarray,
    F: np.ndarray
) -> np.ndarray:
    """Compute epipolar errors for point correspondences.

    Args:
        pts0: Points in image 0 [N, 2]
        pts1: Points in image 1 [N, 2]
        F: Fundamental matrix [3, 3]

    Returns:
        Epipolar errors [N] in pixels
    """
    # Convert to homogeneous coordinates
    pts0_h = np.concatenate([pts0, np.ones((len(pts0), 1))], axis=1)
    pts1_h = np.concatenate([pts1, np.ones((len(pts1), 1))], axis=1)

    # Compute epipolar lines in image 1
    lines1 = (F @ pts0_h.T).T  # [N, 3]

    # Point-to-line distance
    numerator = np.abs(np.sum(pts1_h * lines1, axis=1))
    denominator = np.sqrt(lines1[:, 0]**2 + lines1[:, 1]**2)
    errors = numerator / (denominator + 1e-8)

    return errors


def compute_depth_reprojection_errors(
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    matches: np.ndarray,
    depth0: np.ndarray,
    depth1: np.ndarray,
    K0: np.ndarray,
    K1: np.ndarray,
    T_0to1: np.ndarray
) -> Optional[np.ndarray]:
    """Compute 3D reprojection errors using depth maps.

    Args:
        mkpts0: Matched keypoints in image 0 [N, 2]
        mkpts1: Matched keypoints in image 1 [N, 2]
        matches: Match indices [N, 2]
        depth0: Depth map for image 0
        depth1: Depth map for image 1
        K0: Intrinsics for image 0 [3, 3]
        K1: Intrinsics for image 1 [3, 3]
        T_0to1: Pose from image 0 to 1 [4, 4]

    Returns:
        3D reprojection errors [N] in meters
    """
    errors = []

    for i, (pt0, pt1) in enumerate(zip(mkpts0, mkpts1)):
        # Get depth values
        x0, y0 = int(pt0[0]), int(pt0[1])
        x1, y1 = int(pt1[0]), int(pt1[1])

        if (x0 < 0 or y0 < 0 or x0 >= depth0.shape[1] or y0 >= depth0.shape[0] or
            x1 < 0 or y1 < 0 or x1 >= depth1.shape[1] or y1 >= depth1.shape[0]):
            continue

        d0 = depth0[y0, x0]
        d1 = depth1[y1, x1]

        if d0 <= 0 or d1 <= 0:
            continue

        # Backproject to 3D
        pt0_3d = np.linalg.inv(K0) @ np.array([pt0[0], pt0[1], 1]) * d0
        pt1_3d = np.linalg.inv(K1) @ np.array([pt1[0], pt1[1], 1]) * d1

        # Transform point from camera 0 to camera 1
        pt0_in_cam1 = (T_0to1[:3, :3] @ pt0_3d + T_0to1[:3, 3])

        # Compute 3D error
        error = np.linalg.norm(pt0_in_cam1 - pt1_3d)
        errors.append(error)

    return np.array(errors) if errors else None


def evaluate_pose(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    matches: np.ndarray,
    K0: np.ndarray,
    K1: np.ndarray,
    T_gt: np.ndarray,
    ransac_threshold: float = 1.0,
) -> Dict[str, float]:
    """Evaluate pose estimation from matches.

    Args:
        kpts0: Keypoints in image 0 [N, 2]
        kpts1: Keypoints in image 1 [M, 2]
        matches: Match indices [K, 2]
        K0: Camera intrinsics for image 0 [3, 3]
        K1: Camera intrinsics for image 1 [3, 3]
        T_gt: Ground truth pose from image 0 to 1 [4, 4]
        ransac_threshold: RANSAC threshold in pixels

    Returns:
        Dictionary of metrics:
            - pose_estimated: Whether pose was successfully estimated
            - rotation_error: Rotation error in degrees
            - translation_error: Translation error in degrees
            - num_inliers: Number of RANSAC inliers
    """
    metrics = {
        "pose_estimated": False,
        "rotation_error": float("inf"),
        "translation_error": float("inf"),
        "num_inliers": 0,
    }

    if len(matches) < 8:
        return metrics

    # Get matched keypoints
    mkpts0 = kpts0[matches[:, 0]]
    mkpts1 = kpts1[matches[:, 1]]

    # Estimate essential matrix
    E, mask = cv2.findEssentialMat(
        mkpts0, mkpts1, K0, method=cv2.RANSAC,
        prob=0.9999, threshold=ransac_threshold
    )

    if E is None or mask is None:
        return metrics

    inliers = mask.ravel() == 1
    metrics["num_inliers"] = int(np.sum(inliers))

    if metrics["num_inliers"] < 8:
        return metrics

    # Recover pose
    _, R, t, _ = cv2.recoverPose(
        E, mkpts0[inliers], mkpts1[inliers], K0
    )

    # Construct estimated pose
    T_est = np.eye(4)
    T_est[:3, :3] = R
    T_est[:3, 3] = t.ravel()

    # Compute pose errors
    rot_err, trans_err = compute_pose_error(T_est, T_gt)

    metrics["pose_estimated"] = True
    metrics["rotation_error"] = rot_err
    metrics["translation_error"] = trans_err

    return metrics


def compute_pose_error(T_est: np.ndarray, T_gt: np.ndarray) -> Tuple[float, float]:
    """Compute rotation and translation error.

    Args:
        T_est: Estimated pose [4, 4]
        T_gt: Ground truth pose [4, 4]

    Returns:
        (rotation_error_deg, translation_error_deg)
    """
    # Rotation error
    R_est = T_est[:3, :3]
    R_gt = T_gt[:3, :3]
    R_err = R_est.T @ R_gt
    trace = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
    rot_err = np.degrees(np.arccos(trace))

    # Translation error (angular)
    t_est = T_est[:3, 3]
    t_gt = T_gt[:3, 3]

    # Normalize
    t_est_norm = t_est / (np.linalg.norm(t_est) + 1e-8)
    t_gt_norm = t_gt / (np.linalg.norm(t_gt) + 1e-8)

    # Angular error
    cos_angle = np.clip(np.dot(t_est_norm, t_gt_norm), -1, 1)
    trans_err = np.degrees(np.arccos(cos_angle))

    return rot_err, trans_err
