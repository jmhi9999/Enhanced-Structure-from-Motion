"""
Context-Aware Bundle Adjustment optimizer

Implements weighted BA optimization where observations are weighted by
camera and point confidence scores from scene graph analysis.

Objective:
    minimize Σ w_i × w_j × ||π(P_i, X_j) - x_ij||²

where:
    w_i = confidence(camera_i)
    w_j = confidence(point_j)
    π() = projection function
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import struct
from dataclasses import dataclass

from .scene_graph import SceneGraph, SceneGraphBuilder
from .confidence import ConfidenceCalculator, RuleBasedConfidence, HYBRID_AVAILABLE
from .config import ContextBAConfig

logger = logging.getLogger(__name__)


@dataclass
class ParameterLayout:
    """Helper structure describing parameter vector layout for BA."""

    camera_ids_sorted: List[int]
    camera_param_sizes: List[int]
    camera_param_offsets: List[int]
    camera_id_to_idx: Dict[int, int]
    image_keys_sorted: List[Any]
    image_key_to_idx: Dict[Any, int]
    image_param_size: int
    image_base_offset: int
    point_ids_sorted: List[int]
    point_id_to_idx: Dict[int, int]
    point_param_size: int
    point_base_offset: int
    total_params: int


class ContextAwareBundleAdjustment:
    """
    Context-Aware Bundle Adjustment optimizer

    Drop-in replacement for COLMAP's traditional BA with confidence weighting.
    """

    def __init__(self, config: Optional[ContextBAConfig] = None):
        """
        Args:
            config: ContextBAConfig or None (uses defaults)
        """
        self.config = config or ContextBAConfig()
        self.logger = logging.getLogger(__name__)

        # Configure logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))

        # Initialize confidence calculator
        if self.config.confidence_mode == "hybrid" and HYBRID_AVAILABLE:
            from .confidence import HybridConfidence
            self.confidence_calc = HybridConfidence(self.config)
        else:
            if self.config.confidence_mode == "hybrid":
                self.logger.warning(
                    "Hybrid mode requested but PyTorch not available. "
                    "Falling back to rule-based."
                )
            self.confidence_calc = RuleBasedConfidence(self.config)

    def optimize(
        self,
        features: Dict[str, Any],
        matches: Dict[Tuple[str, str], Any],
        image_dir: Path,
        database_path: Optional[Path] = None,
    ) -> Tuple[Dict[int, Any], Dict[int, Any], Dict[int, Any]]:
        """
        Run context-aware bundle adjustment

        Args:
            features: Feature extraction results
            matches: Feature matching results (after MAGSAC filtering)
            image_dir: Directory containing images
            database_path: Optional path to COLMAP database (for initialization)

        Returns:
            Tuple of (cameras, images, points3d)
        """
        self.logger.info("Starting Context-Aware Bundle Adjustment...")

        # Step 1: Build scene graph
        self.logger.info("Building scene graph...")
        graph_builder = SceneGraphBuilder(self.config.scene_graph)
        self.scene_graph = graph_builder.build(features, matches)

        # Step 2: Initialize reconstruction (triangulation)
        self.logger.info("Initializing reconstruction...")

        # Determine output path (use database_path's parent or create temp)
        if database_path:
            output_path = database_path.parent
        else:
            output_path = Path("./temp_colmap_init")
            output_path.mkdir(exist_ok=True)

        cameras, images, points3d = self._initialize_reconstruction(
            features, matches, image_dir, output_path
        )

        if len(points3d) == 0:
            self.logger.error("Initialization failed: no 3D points triangulated")
            return cameras, images, points3d

        self.logger.info(
            f"Initialized: {len(cameras)} cameras, {len(images)} images, "
            f"{len(points3d)} points"
        )

        # Step 3: Compute confidence scores
        self.logger.info("Computing confidence scores...")
        camera_confidences = self._compute_camera_confidences(features)
        point_confidences = self._compute_point_confidences(points3d, cameras, images)

        if camera_confidences.size:
            self.logger.info(
                f"Camera confidence: mean={camera_confidences.mean():.3f}, "
                f"std={camera_confidences.std():.3f}"
            )
        else:
            self.logger.info("Camera confidence: no cameras available")

        if point_confidences:
            point_conf_array = np.fromiter(point_confidences.values(), dtype=float)
            self.logger.info(
                f"Point confidence: mean={point_conf_array.mean():.3f}, "
                f"std={point_conf_array.std():.3f}"
            )
        else:
            self.logger.info("Point confidence: no points available")

        # Step 4: Run weighted bundle adjustment
        self.logger.info("Running weighted bundle adjustment...")
        cameras, images, points3d = self._bundle_adjustment(
            cameras, images, points3d, camera_confidences, point_confidences
        )

        self.logger.info("Context-Aware Bundle Adjustment completed")
        return cameras, images, points3d

    def _initialize_reconstruction(
        self,
        features: Dict[str, Any],
        matches: Dict[Tuple[str, str], Any],
        image_dir: Path,
        output_path: Path,
    ) -> Tuple[Dict[int, Any], Dict[int, Any], Dict[int, Any]]:
        """
        Initialize reconstruction using incremental SfM

        For now, delegates to COLMAP for initialization, but this could be
        replaced with custom initialization in the future.
        """
        # Import here to avoid circular dependency
        from ..colmap_binary import colmap_binary_reconstruction

        # Use COLMAP for initialization (note: returns sparse_points, cameras, images)
        sparse_points, cameras, images = colmap_binary_reconstruction(
            features, matches, output_path, image_dir
        )

        # Return in correct order for our usage
        return cameras, images, sparse_points

    def _compute_camera_confidences(
        self,
        features: Dict[str, Any],
    ) -> np.ndarray:
        """
        Compute confidence scores for all cameras

        Returns:
            Array of confidence scores, shape (num_cameras,)
        """
        confidences = self.confidence_calc.compute_all_camera_confidences(
            self.scene_graph, features
        )

        # Apply minimum threshold
        if self.config.enable_confidence_weighting:
            confidences = np.maximum(confidences, self.config.min_confidence_threshold)

        return confidences

    def _compute_point_confidences(
        self,
        points3d: Dict[int, Any],
        cameras: Dict[int, Any],
        images: Dict[int, Any],
    ) -> Dict[int, float]:
        """
        Compute confidence scores for all 3D points

        Returns:
            Dictionary mapping point_id to confidence score
        """
        if len(points3d) == 0:
            return {}

        confidences = self.confidence_calc.compute_all_point_confidences(
            points3d, cameras, images
        )

        # Apply minimum threshold
        if self.config.enable_confidence_weighting:
            for point_id in confidences:
                confidences[point_id] = max(confidences[point_id], self.config.min_confidence_threshold)

        return confidences

    def _build_parameter_layout(
        self,
        cameras: Dict[int, Any],
        images: Dict[int, Any],
        points3d: Dict[int, Any],
    ) -> ParameterLayout:
        """Construct layout describing how parameters are packed into a single vector."""
        camera_ids_sorted = sorted(cameras.keys())
        camera_param_sizes: List[int] = []
        camera_param_offsets: List[int] = []
        offset = 0

        for cam_id in camera_ids_sorted:
            camera = cameras[cam_id]
            params = camera.get("params", [])
            param_len = len(params)
            if param_len == 0:
                raise ValueError(f"Camera {cam_id} has no parameters; cannot optimize.")
            camera_param_sizes.append(param_len)
            camera_param_offsets.append(offset)
            offset += param_len

        camera_id_to_idx = {cam_id: idx for idx, cam_id in enumerate(camera_ids_sorted)}

        image_keys_sorted = sorted(images.keys())
        image_key_to_idx = {key: idx for idx, key in enumerate(image_keys_sorted)}
        image_param_size = 7  # qvec (4) + tvec (3)
        image_base_offset = offset
        offset += len(image_keys_sorted) * image_param_size

        point_ids_sorted = sorted(points3d.keys())
        point_id_to_idx = {pt_id: idx for idx, pt_id in enumerate(point_ids_sorted)}
        point_param_size = 3  # xyz
        point_base_offset = offset
        offset += len(point_ids_sorted) * point_param_size

        total_params = offset

        return ParameterLayout(
            camera_ids_sorted=camera_ids_sorted,
            camera_param_sizes=camera_param_sizes,
            camera_param_offsets=camera_param_offsets,
            camera_id_to_idx=camera_id_to_idx,
            image_keys_sorted=image_keys_sorted,
            image_key_to_idx=image_key_to_idx,
            image_param_size=image_param_size,
            image_base_offset=image_base_offset,
            point_ids_sorted=point_ids_sorted,
            point_id_to_idx=point_id_to_idx,
            point_param_size=point_param_size,
            point_base_offset=point_base_offset,
            total_params=total_params,
        )

    def _bundle_adjustment(
        self,
        cameras: Dict[int, Any],
        images: Dict[int, Any],
        points3d: Dict[int, Any],
        camera_confidences: np.ndarray,
        point_confidences: Dict[int, float],
    ) -> Tuple[Dict[int, Any], Dict[int, Any], Dict[int, Any]]:
        """
        Run weighted bundle adjustment optimization

        Args:
            cameras: Camera parameters
            images: Image poses (qvec, tvec, camera_id)
            points3d: 3D points
            camera_confidences: Confidence scores for cameras
            point_confidences: Confidence scores for points

        Returns:
            Optimized (cameras, images, points3d)
        """
        layout = self._build_parameter_layout(cameras, images, points3d)

        # Collect observations
        observations = []  # [(camera_idx, point_idx, image_idx, x, y), ...]

        colmap_id_to_image_key = {}
        for img_key in layout.image_keys_sorted:
            img_data = images[img_key]
            colmap_id = img_data.get("colmap_image_id")
            if colmap_id is not None:
                colmap_id_to_image_key[colmap_id] = img_key

        for point_id, point_data in points3d.items():
            point_idx = layout.point_id_to_idx.get(point_id)
            if point_idx is None:
                continue

            for img_id, pt2d_idx in zip(point_data['image_ids'], point_data['point2D_idxs']):
                if img_id in images:
                    image_key = img_id
                else:
                    image_key = colmap_id_to_image_key.get(int(img_id)) if img_id is not None else None
                if image_key not in images:
                    continue

                image_idx = layout.image_key_to_idx.get(image_key)
                if image_idx is None:
                    continue

                image = images[image_key]
                camera_id = image['camera_id']
                camera_idx = layout.camera_id_to_idx.get(camera_id)
                if camera_idx is None:
                    continue

                xys = image['xys']
                if pt2d_idx >= len(xys) or pt2d_idx < 0:
                    continue

                x, y = xys[pt2d_idx]
                observations.append((camera_idx, point_idx, image_idx, x, y))

        if len(observations) == 0:
            self.logger.warning("No valid observations for BA")
            return cameras, images, points3d

        self.logger.info(f"Total observations: {len(observations)}")

        # Convert to parameter vector
        params = self._pack_parameters(cameras, images, points3d, layout)

        # Compute observation weights
        weights = self._compute_observation_weights(
            observations, camera_confidences, point_confidences, layout
        )

        # Define residual function
        def residual_fn(params_vec):
            return self._compute_residuals(
                params_vec, observations, weights,
                cameras, images, points3d, layout
            )

        # Define Jacobian sparsity structure
        jacobian_sparsity = self._compute_jacobian_sparsity(
            observations, layout
        )

        # Run optimization
        self.logger.info("Running scipy least_squares optimization...")

        result = least_squares(
            residual_fn,
            params,
            jac_sparsity=jacobian_sparsity,
            verbose=self.config.optimizer.verbose,
            x_scale='jac',
            ftol=self.config.optimizer.ftol,
            xtol=self.config.optimizer.xtol,
            max_nfev=self.config.optimizer.max_iterations,
            method='trf',
            loss=self.config.optimizer.loss,
        )

        self.logger.info(
            f"Optimization finished: cost={result.cost:.4f}, "
            f"iterations={result.nfev}, success={result.success}"
        )

        # Unpack optimized parameters
        cameras_opt, images_opt, points3d_opt = self._unpack_parameters(
            result.x, cameras, images, points3d, layout
        )

        return cameras_opt, images_opt, points3d_opt

    def _pack_parameters(
        self,
        cameras: Dict[int, Any],
        images: Dict[int, Any],
        points3d: Dict[int, Any],
        layout: ParameterLayout,
    ) -> np.ndarray:
        """Pack cameras, images, and points into a single parameter vector using provided layout."""
        params_list: List[float] = []

        for cam_id in layout.camera_ids_sorted:
            camera = cameras[cam_id]
            params_list.extend(list(camera.get('params', [])))

        for img_key in layout.image_keys_sorted:
            image = images[img_key]
            params_list.extend(list(image['qvec']))
            params_list.extend(list(image['tvec']))

        for pt_id in layout.point_ids_sorted:
            point = points3d[pt_id]
            params_list.extend(list(point['xyz']))

        return np.array(params_list, dtype=np.float64)

    def _unpack_parameters(
        self,
        params: np.ndarray,
        cameras: Dict[int, Any],
        images: Dict[int, Any],
        points3d: Dict[int, Any],
        layout: ParameterLayout,
    ) -> Tuple[Dict[int, Any], Dict[int, Any], Dict[int, Any]]:
        """Unpack parameter vector back to cameras, images, points."""
        idx = 0

        cameras_new = {}
        for cam_id, param_len in zip(layout.camera_ids_sorted, layout.camera_param_sizes):
            camera = cameras[cam_id].copy()
            camera['params'] = params[idx:idx+param_len]
            cameras_new[cam_id] = camera
            idx += param_len

        images_new = {}
        for img_key in layout.image_keys_sorted:
            image = images[img_key].copy()
            image['qvec'] = params[idx:idx+4]
            image['tvec'] = params[idx+4:idx+7]
            images_new[img_key] = image
            idx += layout.image_param_size

        points3d_new = {}
        for pt_id in layout.point_ids_sorted:
            point = points3d[pt_id].copy()
            point['xyz'] = params[idx:idx+3]
            points3d_new[pt_id] = point
            idx += layout.point_param_size

        return cameras_new, images_new, points3d_new

    def _compute_observation_weights(
        self,
        observations: List[Tuple[int, int, int, float, float]],
        camera_confidences: np.ndarray,
        point_confidences: Dict[int, float],
        layout: ParameterLayout,
    ) -> np.ndarray:
        """
        Compute weights for each observation: w_ij = w_camera_i × w_point_j

        Returns:
            Weights array, shape (num_observations,)
        """
        if not self.config.enable_confidence_weighting:
            return np.ones(len(observations))

        weights = np.ones(len(observations))

        for obs_idx, (camera_idx, point_idx, image_idx, _, _) in enumerate(observations):
            cam_weight = 1.0
            if camera_confidences.size > 0:
                graph_cam_id = None
                if hasattr(self, "scene_graph") and self.scene_graph is not None:
                    image_key = layout.image_keys_sorted[image_idx]
                    graph_cam_id = self.scene_graph.image_to_id.get(image_key)

                if graph_cam_id is not None and graph_cam_id < camera_confidences.size:
                    cam_weight = camera_confidences[graph_cam_id]
                elif image_idx < camera_confidences.size:
                    cam_weight = camera_confidences[image_idx]
                else:
                    # Fallback: clamp using available confidence index
                    safe_idx = min(camera_idx, camera_confidences.size - 1)
                    cam_weight = camera_confidences[safe_idx]

            point_id = layout.point_ids_sorted[point_idx]
            point_weight = point_confidences.get(point_id, 0.5)

            weights[obs_idx] = cam_weight * point_weight

        return weights

    def _compute_residuals(
        self,
        params: np.ndarray,
        observations: List[Tuple[int, int, int, float, float]],
        weights: np.ndarray,
        cameras: Dict[int, Any],
        images: Dict[int, Any],
        points3d: Dict[int, Any],
        layout: ParameterLayout,
    ) -> np.ndarray:
        """
        Compute weighted reprojection residuals.

        Returns:
            Residuals array, shape (2 * num_observations,)
        """
        residuals = np.zeros(2 * len(observations))

        for obs_idx, (camera_idx, point_idx, image_idx, x_obs, y_obs) in enumerate(observations):
            cam_offset = layout.camera_param_offsets[camera_idx]
            cam_len = layout.camera_param_sizes[camera_idx]
            cam_params = params[cam_offset:cam_offset + cam_len]

            cam_id = layout.camera_ids_sorted[camera_idx]
            camera = cameras[cam_id]

            fx, fy, cx, cy = self._intrinsics_from_params(camera, cam_params)

            img_offset = layout.image_base_offset + image_idx * layout.image_param_size
            qvec = params[img_offset:img_offset+4]
            tvec = params[img_offset+4:img_offset+7]

            if qvec.size != 4 or tvec.size != 3:
                raise ValueError(
                    f"Invalid pose parameter lengths (qvec={qvec.size}, tvec={tvec.size}) "
                    f"for observation {obs_idx} (camera_idx={camera_idx}, image_idx={image_idx})"
                )

            pt_offset = layout.point_base_offset + point_idx * layout.point_param_size
            X = params[pt_offset:pt_offset+layout.point_param_size]
            if X.size != 3:
                raise ValueError(
                    f"Invalid point size {X.size} for observation {obs_idx} (point_idx={point_idx})"
                )

            x_proj, y_proj = self._project_point(
                X,
                qvec,
                tvec,
                fx,
                fy,
                cx,
                cy,
                camera.get('model', 'PINHOLE'),
                cam_params,
            )

            weight = np.sqrt(weights[obs_idx])  # sqrt because residual is squared later
            residuals[2*obs_idx] = weight * (x_proj - x_obs)
            residuals[2*obs_idx+1] = weight * (y_proj - y_obs)

        return residuals

    @staticmethod
    def _intrinsics_from_params(
        camera: Dict[int, Any],
        params: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """Derive fx, fy, cx, cy from COLMAP camera parameters for supported models."""
        model = camera.get('model', 'PINHOLE')
        params = np.asarray(params, dtype=np.float64)

        if params.size == 0:
            raise ValueError(f"Camera {camera} has no parameters.")

        simple_models = {
            "SIMPLE_PINHOLE",
            "SIMPLE_RADIAL",
            "SIMPLE_RADIAL_FISHEYE",
            "SIMPLE_EQUIRECTANGULAR",
            "SPHERICAL",
        }
        pinhole_like = {
            "PINHOLE",
            "RADIAL",
            "RADIAL_FISHEYE",
            "OPENCV",
            "OPENCV_FISHEYE",
            "FULL_OPENCV",
            "FOV",
            "THIN_PRISM_FISHEYE",
            "DUAL",
        }

        if model in simple_models:
            if params.size < 3:
                raise ValueError(f"Camera model {model} expects ≥3 parameters, got {params.size}")
            f, cx, cy = params[:3]
            return float(f), float(f), float(cx), float(cy)

        if model in pinhole_like or params.size >= 4:
            fx, fy, cx, cy = params[:4]
            return float(fx), float(fy), float(cx), float(cy)

        if params.size == 3:
            f, cx, cy = params
            return float(f), float(f), float(cx), float(cy)

        raise ValueError(f"Unsupported camera model {model} with parameters length {params.size}")

    @staticmethod
    def _project_point(
        X: np.ndarray,
        qvec: np.ndarray,
        tvec: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        camera_model: str,
        camera_params: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Project 3D point to 2D using camera parameters

        Args:
            X: 3D point in world coordinates (3,)
            qvec: Rotation as quaternion (w, x, y, z) (4,)
            tvec: Translation (3,)
            fx, fy, cx, cy: Camera intrinsics
            camera_model: COLMAP camera model string
            camera_params: Original camera parameter vector

        Returns:
            (x, y) projected 2D coordinates
        """
        # Convert quaternion to rotation matrix
        qw, qx, qy, qz = qvec
        R = np.array([
            [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
            [    2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2,     2*qy*qz - 2*qx*qw],
            [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])

        # Transform to camera coordinates
        X_cam = R @ X + tvec

        # Project to image plane
        x = X_cam[0] / X_cam[2]
        y = X_cam[1] / X_cam[2]

        x, y = ContextAwareBundleAdjustment._apply_distortion(
            x, y, camera_model, camera_params
        )

        # Apply intrinsics
        u = fx * x + cx
        v = fy * y + cy

        return u, v

    @staticmethod
    def _apply_distortion(
        x: float,
        y: float,
        camera_model: str,
        camera_params: np.ndarray,
    ) -> Tuple[float, float]:
        """Apply simple radial/tangential distortion models supported by COLMAP."""
        model = (camera_model or "PINHOLE").upper()
        params = np.asarray(camera_params, dtype=np.float64)

        # Models without distortion
        if model in {
            "SIMPLE_PINHOLE",
            "PINHOLE",
            "SIMPLE_EQUIRECTANGULAR",
            "SPHERICAL",
        }:
            return x, y

        r2 = x * x + y * y

        if model in {"SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"} and params.size >= 4:
            k = params[3]
            radial = 1.0 + k * r2
            return x * radial, y * radial

        if model == "RADIAL" and params.size >= 6:
            k1, k2 = params[4:6]
            radial = 1.0 + k1 * r2 + k2 * (r2 ** 2)
            return x * radial, y * radial

        if model in {"OPENCV", "FULL_OPENCV"} and params.size >= 8:
            k1, k2, p1, p2 = params[4:8]
            k3 = params[8] if params.size > 8 else 0.0
            k4 = params[9] if params.size > 9 else 0.0
            k5 = params[10] if params.size > 10 else 0.0
            k6 = params[11] if params.size > 11 else 0.0
            radial = 1.0 + k1 * r2 + k2 * (r2 ** 2) + k3 * (r2 ** 3)
            radial += k4 * (r2 ** 4) + k5 * (r2 ** 5) + k6 * (r2 ** 6)
            x_dist = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            y_dist = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            return x_dist, y_dist

        if model == "OPENCV_FISHEYE" and params.size >= 8:
            k1, k2, k3, k4 = params[4:8]
            r = np.sqrt(r2)
            if r < 1e-8:
                return x, y
            theta = np.arctan(r)
            theta_d = theta * (1 + k1 * theta ** 2 + k2 * theta ** 4 + k3 * theta ** 6 + k4 * theta ** 8)
            scale = theta_d / (r + 1e-12)
            return x * scale, y * scale

        if model == "FOV" and params.size >= 5:
            omega = params[4]
            if omega < 1e-8:
                return x, y
            r = np.sqrt(r2)
            if r < 1e-8:
                return x, y
            tan_half_omega = np.tan(omega / 2.0)
            rd = (1.0 / omega) * np.arctan(2.0 * r * tan_half_omega)
            return x * (rd / (r + 1e-12)), y * (rd / (r + 1e-12))

        # Default fallback: no distortion applied
        return x, y

    def _compute_jacobian_sparsity(
        self,
        observations: List[Tuple[int, int, int, float, float]],
        layout: ParameterLayout,
    ) -> lil_matrix:
        """
        Compute sparsity pattern of Jacobian matrix

        Each observation depends on:
        - 1 camera (variable params)
        - 1 image (7 params)
        - 1 point (3 params)
        """
        num_observations = len(observations)
        num_params = layout.total_params

        sparsity = lil_matrix((2 * num_observations, num_params), dtype=int)

        for obs_idx, (camera_idx, point_idx, image_idx, _, _) in enumerate(observations):
            # Camera parameters
            cam_offset = layout.camera_param_offsets[camera_idx]
            cam_size = layout.camera_param_sizes[camera_idx]
            sparsity[2*obs_idx, cam_offset:cam_offset+cam_size] = 1
            sparsity[2*obs_idx+1, cam_offset:cam_offset+cam_size] = 1

            # Image parameters
            img_offset = layout.image_base_offset + image_idx * layout.image_param_size
            sparsity[2*obs_idx, img_offset:img_offset+layout.image_param_size] = 1
            sparsity[2*obs_idx+1, img_offset:img_offset+layout.image_param_size] = 1

            # Point parameters
            pt_offset = layout.point_base_offset + point_idx * layout.point_param_size
            sparsity[2*obs_idx, pt_offset:pt_offset+layout.point_param_size] = 1
            sparsity[2*obs_idx+1, pt_offset:pt_offset+layout.point_param_size] = 1

        return sparsity
