"""
Transformer-based feature extractor built around DINO-style vision encoders.

The extractor supports both DINOv3 (default) and the previous DINOv2 models.
It exposes CLS tokens, patch embeddings, and attention maps that other parts
of the SfM pipeline can consume for retrieval, matching, and weighting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from sfm.core.feature_extractor import BaseFeatureExtractor

LOGGER = logging.getLogger(__name__)

try:
    import timm  # type: ignore

    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


@dataclass
class _PatchMeta:
    grid_hw: Tuple[int, int]
    patch_size: Tuple[int, int]
    pad_hw: Tuple[int, int]
    original_hw: Tuple[int, int]

    @property
    def total_patches(self) -> int:
        return self.grid_hw[0] * self.grid_hw[1]


class DINOFeatureExtractor(BaseFeatureExtractor):
    """
    Generalised DINO feature extractor.

    Args:
        device: torch device
        config: dictionary supporting:
            - model_family: "dinov3" (default) or "dinov2"
            - model_name: hub / timm model name
            - hub_repo: torch.hub repository (for dinov2/dinov3 official releases)
            - timm_model_name: optional timm model identifier for fallback
            - pretrained_path: optional local checkpoint (state dict) for custom models
            - patch_topk: number of attention-ranked patches to emit as keypoints
    """

    DEFAULT_MODEL_FAMILY = "dinov3"
    DEFAULT_MODEL_NAME = "dinov3_vitl14"
    DEFAULT_DINOV2_MODEL = "dinov2_vits14"
    DEFAULT_TIMM_MODEL = "vit_large_patch14_dinov2.lvd142m"

    def __init__(self, device: torch.device, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model_family = self.config.get("model_family", self.DEFAULT_MODEL_FAMILY)
        self.model_name = self.config.get("model_name")
        if self.model_name is None:
            self.model_name = (
                self.DEFAULT_DINOV2_MODEL
                if self.model_family == "dinov2"
                else self.DEFAULT_MODEL_NAME
            )

        self.timm_model_name = self.config.get("timm_model_name", self.DEFAULT_TIMM_MODEL)
        self.hub_repo = self.config.get(
            "hub_repo",
            "facebookresearch/dinov3"
            if self.model_family == "dinov3"
            else "facebookresearch/dinov2",
        )
        self.pretrained_path = self.config.get("pretrained_path")
        self.patch_topk = self.config.get("patch_topk", 800)
        self.max_keypoints = self.config.get("max_keypoints", 4096)
        self.normalize_descriptors = self.config.get("normalize_descriptors", True)
        self.return_all_patch_tokens = True

        super().__init__(device, config)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _setup_model(self) -> None:
        if self.model_family == "dinov3":
            self._setup_dinov3_model()
        else:
            self._setup_dinov2_model()

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.image_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.image_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self._last_attention: Optional[torch.Tensor] = None

        if hasattr(self.model, "blocks") and len(getattr(self.model, "blocks", [])) > 0:
            try:
                self.model.blocks[-1].attn.register_forward_hook(self._attention_hook)  # type: ignore[attr-defined]
            except Exception:
                LOGGER.warning("Unable to register attention hook; falling back to CLS similarity.")

    def _setup_dinov3_model(self) -> None:
        errors: List[str] = []

        # 1) Attempt to load from torch hub (expected official release path)
        try:
            LOGGER.info("Loading DINOv3 model %s from hub repo %s", self.model_name, self.hub_repo)
            self.model = torch.hub.load(self.hub_repo, self.model_name, pretrained=True)  # type: ignore[attr-defined]
            self.model.to(self.device)
            self.patch_size = self._infer_patch_size(self.model)
            self.embed_dim = self._infer_embed_dim(self.model)
            LOGGER.info("DINOv3 backbone initialised via torch.hub")
            return
        except Exception as exc:
            errors.append(f"torch.hub: {exc}")

        # 2) Attempt to load via timm fallback (user must supply valid model name)
        if TIMM_AVAILABLE:
            timm_name = self.config.get("timm_model_name", self.timm_model_name)
            try:
                LOGGER.info("Falling back to timm model %s for DINOv3 backbone", timm_name)
                self.model = timm.create_model(timm_name, pretrained=True)
                self.model.to(self.device)
                self.patch_size = self._infer_patch_size(self.model)
                self.embed_dim = self._infer_embed_dim(self.model)
                LOGGER.info("DINOv3 backbone initialised via timm fallback")
                return
            except Exception as exc:
                errors.append(f"timm: {exc}")

        # 3) Attempt to load from explicit checkpoint (requires code providing architecture)
        if self.pretrained_path:
            checkpoint_path = Path(self.pretrained_path)
            if checkpoint_path.exists():
                try:
                    LOGGER.info("Loading DINOv3 weights from %s", checkpoint_path)
                    state_dict = torch.load(checkpoint_path, map_location="cpu")
                    if "model" in state_dict:
                        state_dict = state_dict["model"]
                    # Expect a compatible architecture to be provided via config
                    architecture = self.config.get("architecture")
                    if architecture is None:
                        raise ValueError("`architecture` must be provided when using pretrained_path.")
                    if isinstance(architecture, torch.nn.Module):
                        self.model = architecture
                    else:
                        raise ValueError("`architecture` must be a torch.nn.Module instance.")
                    self.model.load_state_dict(state_dict, strict=False)
                    self.model.to(self.device)
                    self.patch_size = self._infer_patch_size(self.model)
                    self.embed_dim = self._infer_embed_dim(self.model)
                    LOGGER.info("DINOv3 backbone loaded from custom checkpoint")
                    return
                except Exception as exc:
                    errors.append(f"checkpoint: {exc}")

        raise ImportError(
            "Unable to initialise DINOv3 backbone. Attempted torch.hub, timm fallback, and optional "
            f"checkpoint loading. Errors: {errors}. Please ensure the official DINOv3 release is "
            "installed or provide a compatible timm model via `timm_model_name`."
        )

    def _setup_dinov2_model(self) -> None:
        try:
            LOGGER.info("Loading DINOv2 model %s from %s", self.model_name, self.hub_repo)
            self.model = torch.hub.load(self.hub_repo, self.model_name, pretrained=True)  # type: ignore[attr-defined]
            self.model.to(self.device)
            self.patch_size = self._infer_patch_size(self.model)
            self.embed_dim = self._infer_embed_dim(self.model)
        except Exception as exc:
            raise ImportError(
                f"Failed to load DINOv2 model '{self.model_name}' from repo '{self.hub_repo}'. "
                "Install the official DINOv2 release or pass a custom checkpoint."
            ) from exc

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def extract_features(
        self, images: List[Dict[str, Any]], batch_size: int = 4
    ) -> Dict[str, Dict[str, Any]]:
        features: Dict[str, Dict[str, Any]] = {}

        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            batch_tensor, paths, metas, originals = self._prepare_batch(batch)

            with torch.no_grad():
                outputs = self._forward_features(batch_tensor)

            for idx, path in enumerate(paths):
                meta = metas[idx]
                cls_token = outputs["cls"][idx].cpu().numpy()
                patch_tokens = outputs["patch_tokens"][idx]

                attn_map = self._compute_attention_map(idx, outputs, meta, patch_tokens)
                keypoint_data = self._select_keypoints(
                    patch_tokens, attn_map, meta, originals[idx]
                )

                features[path] = {
                    "keypoints": keypoint_data["keypoints"],
                    "descriptors": keypoint_data["descriptors"],
                    "scores": keypoint_data["scores"],
                    "image_shape": meta.original_hw,
                    "dino_cls": cls_token.astype(np.float32),
                    "dino_patch_tokens": patch_tokens.cpu().numpy().astype(np.float32),
                    "dino_patch_coords": keypoint_data["all_patch_coords"],
                    "dino_patch_attention": keypoint_data["all_patch_attention"],
                    "dino_patch_grid_hw": np.array(meta.grid_hw, dtype=np.int32),
                    "dino_attention_map": keypoint_data["attention_map"],
                    "dino_model_family": self.model_family,
                    "dino_model_name": self.model_name,
                }

            del batch_tensor, outputs
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return features

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _attention_hook(self, _module, _inputs, output):
        if isinstance(output, tuple):
            self._last_attention = output[0].detach()
        else:
            self._last_attention = output.detach()

    def _infer_patch_size(self, model: torch.nn.Module) -> Tuple[int, int]:
        patch_size = getattr(model, "patch_size", None)
        if patch_size is None and hasattr(model, "patch_embed"):
            patch_size = getattr(model.patch_embed, "patch_size", None)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if patch_size is None:
            patch_size = (14, 14)
        return tuple(int(x) for x in patch_size)

    def _infer_embed_dim(self, model: torch.nn.Module) -> int:
        if hasattr(model, "embed_dim"):
            return int(model.embed_dim)
        if hasattr(model, "num_features"):
            return int(model.num_features)
        if hasattr(model, "hidden_dim"):
            return int(model.hidden_dim)
        return 768

    def _prepare_batch(
        self, images: Iterable[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, List[str], List[_PatchMeta], List[np.ndarray]]:
        tensors: List[torch.Tensor] = []
        paths: List[str] = []
        metas: List[_PatchMeta] = []
        originals: List[np.ndarray] = []

        for img_data in images:
            img = img_data["image"]
            path = img_data["path"]

            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            originals.append(img)
            img_f = img.astype(np.float32) / 255.0
            tensor = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).to(self.device)

            _, _, h, w = tensor.shape
            pad_h = (self.patch_size[0] - (h % self.patch_size[0])) % self.patch_size[0]
            pad_w = (self.patch_size[1] - (w % self.patch_size[1])) % self.patch_size[1]

            if pad_h != 0 or pad_w != 0:
                tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

            tensor = (tensor - self.image_mean) / self.image_std
            padded_h, padded_w = tensor.shape[2], tensor.shape[3]
            grid_h = padded_h // self.patch_size[0]
            grid_w = padded_w // self.patch_size[1]

            metas.append(
                _PatchMeta(
                    grid_hw=(grid_h, grid_w),
                    patch_size=self.patch_size,
                    pad_hw=(pad_h, pad_w),
                    original_hw=(h, w),
                )
            )
            tensors.append(tensor)
            paths.append(path)

        batch_tensor = torch.cat(tensors, dim=0)
        return batch_tensor, paths, metas, originals

    def _forward_features(self, batch_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        results: Dict[str, torch.Tensor] = {}
        self._last_attention = None

        try:
            out = self.model.forward_features(batch_tensor, return_all_tokens=True)  # type: ignore[attr-defined]
        except TypeError:
            out = self.model.forward_features(batch_tensor)  # type: ignore[attr-defined]

        try:
            if isinstance(out, dict):
                cls_token = out.get("x_cls") or out.get("x_norm_clstoken") or out.get("cls_token")
                patch_tokens = out.get("x_patch") or out.get("x_norm_patchtokens") or out.get("patch_tokens")
                if cls_token is None or patch_tokens is None:
                    raise KeyError("Missing tokens from forward_features dictionary.")
            elif isinstance(out, tuple) and len(out) == 2:
                patch_tokens, cls_token = out
            elif isinstance(out, torch.Tensor) and out.dim() == 3:
                cls_token = out[:, 0:1, :]
                patch_tokens = out[:, 1:, :]
            else:
                raise ValueError("Unexpected forward_features output format.")
        except Exception:
            if hasattr(self.model, "get_intermediate_layers"):
                LOGGER.debug("Falling back to get_intermediate_layers for token extraction.")
                intermediate = self.model.get_intermediate_layers(  # type: ignore[attr-defined]
                    batch_tensor, n=1, return_class_token=True, return_patch_tokens=True
                )
                if isinstance(intermediate, list):
                    patch_tokens, cls_token = intermediate[0][0], intermediate[0][1]
                else:
                    patch_tokens, cls_token = intermediate
            else:
                raise RuntimeError(
                    "Unable to retrieve token embeddings from the DINO backbone. "
                    "Ensure the selected architecture exposes either `forward_features` with "
                    "return_all_tokens=True` support or `get_intermediate_layers`."
                )

        if self.normalize_descriptors:
            cls_token = F.normalize(cls_token, dim=-1)
            patch_tokens = F.normalize(patch_tokens, dim=-1)

        results["cls"] = cls_token
        results["patch_tokens"] = patch_tokens
        return results

    def _compute_attention_map(
        self,
        batch_index: int,
        outputs: Dict[str, torch.Tensor],
        meta: _PatchMeta,
        patch_tokens: torch.Tensor,
    ) -> np.ndarray:
        if self._last_attention is not None:
            attn = self._last_attention[batch_index]
            attn_cls = attn[:, 0, 1:]
            attn_map = attn_cls.mean(dim=0)
        else:
            cls_token = outputs["cls"][batch_index].unsqueeze(0)
            patches = patch_tokens
            attn_map = torch.einsum("nc,mc->n", patches, cls_token)
            attn_map = attn_map - attn_map.min()
            if attn_map.max() > 0:
                attn_map = attn_map / attn_map.max()

        attn_map = attn_map.reshape(meta.grid_hw[0], meta.grid_hw[1])
        attn_np = attn_map.detach().cpu().numpy().astype(np.float32)

        pad_h, pad_w = meta.pad_hw
        if pad_h > 0:
            attn_np = attn_np[: -(pad_h // meta.patch_size[0]), :]
        if pad_w > 0:
            attn_np = attn_np[:, : -(pad_w // meta.patch_size[1])]

        return attn_np

    def _select_keypoints(
        self,
        patch_tokens: torch.Tensor,
        attention_map: np.ndarray,
        meta: _PatchMeta,
        original_image: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        grid_h, grid_w = attention_map.shape
        total_valid = grid_h * grid_w

        all_coords = self._compute_patch_centers(grid_h, grid_w, meta)
        all_attention = attention_map.reshape(-1)

        patch_tokens_np = patch_tokens[: total_valid].cpu().numpy().astype(np.float32)
        if patch_tokens_np.shape[0] != total_valid:
            patch_tokens_np = patch_tokens_np[:total_valid]

        max_kp = min(self.max_keypoints, total_valid)
        top_indices = np.argsort(all_attention)[::-1][:max_kp]

        keypoints = all_coords[top_indices]
        descriptors = patch_tokens_np[top_indices]
        scores = all_attention[top_indices]
        if scores.max() > 0:
            scores = scores / (scores.max() + 1e-6)

        return {
            "keypoints": keypoints.astype(np.float32),
            "descriptors": descriptors.astype(np.float32),
            "scores": scores.astype(np.float32),
            "all_patch_coords": all_coords.astype(np.float32),
            "all_patch_attention": all_attention.astype(np.float32),
            "attention_map": attention_map.astype(np.float32),
        }

    def _compute_patch_centers(
        self, grid_h: int, grid_w: int, meta: _PatchMeta
    ) -> np.ndarray:
        patch_h, patch_w = meta.patch_size
        ys = (np.arange(grid_h) + 0.5) * patch_h
        xs = (np.arange(grid_w) + 0.5) * patch_w
        coords = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
        coords[:, 0] = np.clip(coords[:, 0], 0, meta.original_hw[1] - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, meta.original_hw[0] - 1)
        return coords
