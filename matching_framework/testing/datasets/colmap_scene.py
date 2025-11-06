"""Dataset loader that reuses COLMAP sparse reconstructions as ground truth."""

from __future__ import annotations

import logging
import struct
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .base import BaseDataset, DatasetSample

logger = logging.getLogger(__name__)

# Mapping from COLMAP camera model id to (model name, number of intrinsic params)
CAMERA_MODEL_PARAMS: Dict[int, Tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def _qvec_to_rotmat(qvec: Iterable[float]) -> np.ndarray:
    """Convert COLMAP quaternion (qw, qx, qy, qz) to rotation matrix."""
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ],
        dtype=np.float32,
    )


def _camera_intrinsics(model: str, params: Iterable[float]) -> np.ndarray:
    """Convert COLMAP intrinsic parameters to a 3x3 calibration matrix."""
    model = model.lower()
    params = list(params)

    fx = fy = cx = cy = None

    if model == "pinhole" and len(params) >= 4:
        fx, fy, cx, cy = params[:4]
    elif model in {"simple_pinhole", "simple_radial", "radial", "simple_radial_fisheye", "radial_fisheye"} and len(params) >= 3:
        fx = fy = params[0]
        cx, cy = params[1:3]
    elif len(params) >= 4:
        fx, fy, cx, cy = params[:4]
    elif len(params) >= 3:
        fx = fy = params[0]
        cx, cy = params[1:3]

    if None in {fx, fy, cx, cy}:
        raise ValueError(f"Cannot infer intrinsics for camera model {model} with params {params}")

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return K


def _read_cameras_binary(path: Path) -> Dict[int, Dict]:
    """Read cameras.bin with the official COLMAP binary layout."""
    cameras: Dict[int, Dict] = {}

    with open(path, "rb") as fid:
        num_cameras = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("<I", fid.read(4))[0]
            model_id = struct.unpack("<i", fid.read(4))[0]
            width = struct.unpack("<Q", fid.read(8))[0]
            height = struct.unpack("<Q", fid.read(8))[0]

            if model_id not in CAMERA_MODEL_PARAMS:
                raise ValueError(f"Unsupported COLMAP camera model id {model_id}")

            model, param_count = CAMERA_MODEL_PARAMS[model_id]
            params = struct.unpack(f"<{param_count}d", fid.read(8 * param_count))

            cameras[camera_id] = {
                "model": model,
                "width": int(width),
                "height": int(height),
                "params": params,
            }

    return cameras


def _read_images_binary(path: Path) -> Dict[int, Dict]:
    """Read images.bin with the official COLMAP binary layout."""
    images: Dict[int, Dict] = {}

    with open(path, "rb") as fid:
        num_images = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", fid.read(4))[0]
            qvec = struct.unpack("<4d", fid.read(32))
            tvec = struct.unpack("<3d", fid.read(24))
            camera_id = struct.unpack("<I", fid.read(4))[0]

            # Image name is a null-terminated string
            name_bytes = bytearray()
            while True:
                char = fid.read(1)
                if char == b"\x00":
                    break
                name_bytes.extend(char)
            name = name_bytes.decode("utf-8")

            num_points2d = struct.unpack("<Q", fid.read(8))[0]
            xys = np.zeros((num_points2d, 2), dtype=np.float32)
            point3d_ids = np.empty(num_points2d, dtype=np.int64)
            for i in range(num_points2d):
                xys[i] = struct.unpack("<2d", fid.read(16))
                point3d_ids[i] = struct.unpack("<q", fid.read(8))[0]

            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
                "xys": xys,
                "point3D_ids": point3d_ids,
            }

    return images


def _read_points3d_binary(path: Path) -> Dict[int, Dict]:
    """Read points3D.bin with the official COLMAP binary layout."""
    points3d: Dict[int, Dict] = {}

    with open(path, "rb") as fid:
        num_points = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_points):
            point_id = struct.unpack("<Q", fid.read(8))[0]
            xyz = struct.unpack("<3d", fid.read(24))
            rgb = struct.unpack("<3B", fid.read(3))
            error = struct.unpack("<d", fid.read(8))[0]
            track_length = struct.unpack("<Q", fid.read(8))[0]

            track: List[Tuple[int, int]] = []
            for _ in range(track_length):
                image_id = struct.unpack("<I", fid.read(4))[0]
                point2d_idx = struct.unpack("<I", fid.read(4))[0]
                track.append((image_id, point2d_idx))

            points3d[int(point_id)] = {
                "xyz": np.asarray(xyz, dtype=np.float32),
                "rgb": np.asarray(rgb, dtype=np.uint8),
                "error": float(error),
                "track": track,
            }

    return points3d


def _read_cameras_text(path: Path) -> Dict[int, Dict]:
    cameras: Dict[int, Dict] = {}
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            camera_id = int(elems[0])
            model = elems[1]
            width = int(elems[2])
            height = int(elems[3])
            params = [float(v) for v in elems[4:]]
            cameras[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def _read_images_text(path: Path) -> Dict[int, Dict]:
    images: Dict[int, Dict] = {}
    with open(path, "r") as fid:
        lines = fid.readlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        idx += 1
        if not line or line.startswith("#"):
            continue
        elems = line.split()
        image_id = int(elems[0])
        qvec = [float(v) for v in elems[1:5]]
        tvec = [float(v) for v in elems[5:8]]
        camera_id = int(elems[8])
        name = elems[9]
        xys = []
        point3d_ids = []
        if idx < len(lines):
            points_line = lines[idx].strip()
            idx += 1
            if points_line and not points_line.startswith("#"):
                parts = points_line.split()
                for i in range(0, len(parts), 3):
                    xys.append([float(parts[i]), float(parts[i + 1])])
                    point3d_ids.append(int(parts[i + 2]))
        images[image_id] = {
            "qvec": qvec,
            "tvec": tvec,
            "camera_id": camera_id,
            "name": name,
            "xys": np.asarray(xys, dtype=np.float32),
            "point3D_ids": np.asarray(point3d_ids, dtype=np.int64),
        }
    return images


def _read_points3d_text(path: Path) -> Dict[int, Dict]:
    points3d: Dict[int, Dict] = {}
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            point_id = int(elems[0])
            xyz = [float(v) for v in elems[1:4]]
            rgb = [int(v) for v in elems[4:7]]
            error = float(elems[7]) if len(elems) > 7 else 0.0
            track = []
            for i in range(8, len(elems), 2):
                track.append((int(elems[i]), int(elems[i + 1])))
            points3d[point_id] = {
                "xyz": np.asarray(xyz, dtype=np.float32),
                "rgb": np.asarray(rgb, dtype=np.uint8),
                "error": error,
                "track": track,
            }
    return points3d


def _load_colmap_model(sparse_root: Path) -> Tuple[Dict[int, Dict], Dict[int, Dict], Dict[int, Dict]]:
    """Load COLMAP cameras, images, and points from binary or text files."""
    if not sparse_root.exists():
        raise FileNotFoundError(f"Sparse directory {sparse_root} does not exist.")

    recon_dir = sparse_root
    if (sparse_root / "0").is_dir():
        recon_dir = sparse_root / "0"

    cameras_path = recon_dir / "cameras.bin"
    images_path = recon_dir / "images.bin"
    points_path = recon_dir / "points3D.bin"

    try:
        cameras = _read_cameras_binary(cameras_path)
        images = _read_images_binary(images_path)
        points3d = _read_points3d_binary(points_path)
        logger.info("Loaded COLMAP reconstruction from %s (binary format)", recon_dir)
        return cameras, images, points3d
    except FileNotFoundError as exc:
        logger.warning("Missing binary files (%s). Falling back to text.", exc)
    except Exception as exc:
        logger.warning("Failed to read binary COLMAP files: %s", exc)

    # Fallback to text format
    cameras_txt = recon_dir / "cameras.txt"
    images_txt = recon_dir / "images.txt"
    points_txt = recon_dir / "points3D.txt"

    cameras = _read_cameras_text(cameras_txt)
    images = _read_images_text(images_txt)
    points3d = _read_points3d_text(points_txt)
    logger.info("Loaded COLMAP reconstruction from %s (text format)", recon_dir)
    return cameras, images, points3d


def _contains_images(directory: Path) -> bool:
    """Return True if directory contains images with common extensions."""
    patterns = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG")
    for pattern in patterns:
        if next(directory.glob(pattern), None) is not None:
            return True
    return False


def _find_images_dir(root_path: Path) -> Path:
    """Try to locate folder that stores the RGB images."""
    candidate_names = [
        "images",
        "Images",
        "images_2",
        "Images_2",
        "rgb",
        "RGB",
    ]

    for name in candidate_names:
        candidate = root_path / name
        if candidate.exists() and _contains_images(candidate):
            return candidate

    for child in root_path.iterdir():
        if child.is_dir() and _contains_images(child):
            return child

    raise FileNotFoundError(f"Could not locate an images directory under {root_path}")


def _compose_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Create homogeneous transform from rotation and translation."""
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = t.astype(np.float32)
    return T


class ColmapSceneDataset(BaseDataset):
    """Dataset that reuses an existing COLMAP reconstruction as supervision."""

    def __init__(self, root_path: str, split: str = "test", min_shared_points: int = 30):
        self.min_shared_points = min_shared_points
        super().__init__(root_path, split)

    def _load_samples(self) -> List[DatasetSample]:
        cameras, images, points3d = _load_colmap_model(self.root_path / "sparse")
        if not cameras or not images:
            raise RuntimeError(f"Failed to load COLMAP model from {self.root_path}")

        images_dir = _find_images_dir(self.root_path)
        logger.info("Benchmark dataset images directory: %s", images_dir)

        camera_intrinsics: Dict[int, np.ndarray] = {}
        for cam_id, cam_data in cameras.items():
            try:
                camera_intrinsics[cam_id] = _camera_intrinsics(cam_data["model"], cam_data["params"])
            except ValueError as exc:
                logger.warning("Skipping camera %s: %s", cam_id, exc)

        image_entries: Dict[int, Dict] = {}
        for image_id, im_data in images.items():
            img_path = images_dir / im_data["name"]
            if not img_path.exists():
                logger.warning("Skipping image %s (missing %s)", im_data["name"], img_path)
                continue

            cam_id = im_data["camera_id"]
            if cam_id not in camera_intrinsics:
                logger.warning("Skipping image %s (no intrinsics for camera %s)", im_data["name"], cam_id)
                continue

            image_entries[image_id] = {
                "path": img_path,
                "camera_id": cam_id,
                "K": camera_intrinsics[cam_id],
                "R": _qvec_to_rotmat(im_data["qvec"]),
                "t": np.asarray(im_data["tvec"], dtype=np.float32),
            }

        if not image_entries:
            raise RuntimeError("No valid images found after intrinsics filtering.")

        pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        for point in points3d.values():
            track = point.get("track", [])
            if len(track) < 2:
                continue
            valid_ids = {img_id for img_id, _ in track if img_id in image_entries}
            valid_ids = sorted(valid_ids)
            for i in range(len(valid_ids)):
                for j in range(i + 1, len(valid_ids)):
                    a, b = valid_ids[i], valid_ids[j]
                    pair_counts[(a, b)] += 1

        samples: List[DatasetSample] = []
        for (id0, id1), shared in sorted(pair_counts.items()):
            if shared < self.min_shared_points:
                continue

            entry0 = image_entries.get(id0)
            entry1 = image_entries.get(id1)
            if entry0 is None or entry1 is None:
                continue

            T_w_c0 = _compose_transform(entry0["R"], entry0["t"])
            T_w_c1 = _compose_transform(entry1["R"], entry1["t"])
            T_c0_w = np.linalg.inv(T_w_c0)
            T_0to1 = T_w_c1 @ T_c0_w

            samples.append(
                DatasetSample(
                    image0=entry0["path"],
                    image1=entry1["path"],
                    K0=entry0["K"].copy(),
                    K1=entry1["K"].copy(),
                    T_0to1=T_0to1.astype(np.float32),
                    metadata={
                        "shared_track_count": shared,
                        "image0_id": id0,
                        "image1_id": id1,
                    },
                )
            )

        if not samples:
            raise RuntimeError(
                "No image pairs satisfied the shared track threshold "
                f"({self.min_shared_points}). Try lowering the threshold."
            )

        logger.info(
            "Prepared %d benchmark pairs (min_shared_points=%d)",
            len(samples),
            self.min_shared_points,
        )
        return samples
