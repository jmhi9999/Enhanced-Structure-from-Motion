"""
Loop Closure Detection using Global Descriptors

Detects loop closures (similar images that are temporally far apart) to improve
reconstruction quality and enable component merging.

Uses feature pooling by default (no additional model needed).
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class LoopClosureDetector:
    """
    Detect loop closures using global image descriptors

    Finds pairs of images that are:
    1. Visually similar (high descriptor similarity)
    2. Temporally distant (not sequential neighbors)
    3. Not already connected in the scene graph
    """

    def __init__(
        self,
        descriptor_method: str = 'feature_pooling',
        similarity_threshold: float = 0.70,
        min_temporal_gap: int = 30,
        min_matches_for_verification: int = 15
    ):
        """
        Args:
            descriptor_method: 'feature_pooling' (default) or 'dinov2'
            similarity_threshold: Cosine similarity threshold (0.7 = high similarity)
            min_temporal_gap: Minimum temporal gap to avoid sequential matches
            min_matches_for_verification: Minimum matches for geometric verification
        """
        self.descriptor_method = descriptor_method
        self.similarity_threshold = similarity_threshold
        self.min_temporal_gap = min_temporal_gap
        self.min_matches_for_verification = min_matches_for_verification

        logger.info(f"Loop Closure Detector initialized:")
        logger.info(f"  Method: {descriptor_method}")
        logger.info(f"  Similarity threshold: {similarity_threshold}")
        logger.info(f"  Min temporal gap: {min_temporal_gap}")

    def extract_global_descriptors(
        self,
        features: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Extract global descriptor for each image from local features

        Args:
            features: Feature dictionary from feature extraction
                     {image_path: {'descriptors': [D, N], ...}}

        Returns:
            Dictionary mapping image_path to global descriptor [D]
        """
        logger.info("Extracting global descriptors from local features...")

        global_descriptors = {}
        descriptor_dim = None

        # First pass to determine descriptor dimension
        for feat_data in features.values():
            if 'descriptors' in feat_data and feat_data['descriptors'].shape[0] > 0:
                descriptor_dim = feat_data['descriptors'].shape[0]
                break
        
        if descriptor_dim is None:
            logger.warning("Could not determine descriptor dimension. Loop closure detection may fail.")
            # Fallback to a common dimension, though this is not ideal
            descriptor_dim = 256 

        logger.info(f"Determined descriptor dimension: {descriptor_dim}")

        for img_path, feat_data in features.items():
            descriptors = feat_data.get('descriptors')

            if descriptors is None or descriptors.shape[1] == 0:
                # No keypoints - use zero vector
                global_desc = np.zeros(descriptor_dim, dtype=np.float32)
            else:
                # Max pooling across all keypoints
                global_desc = np.max(descriptors, axis=1)  # [D]

                # L2 normalization
                norm = np.linalg.norm(global_desc)
                if norm > 1e-6:
                    global_desc = global_desc / norm

            global_descriptors[img_path] = global_desc

        logger.info(f"Extracted {len(global_descriptors)} global descriptors")
        return global_descriptors

    def detect_loop_closures(
        self,
        global_descriptors: Dict[str, np.ndarray],
        existing_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> List[Tuple[str, str, float]]:
        """
        Detect potential loop closures based on descriptor similarity

        Args:
            global_descriptors: Global descriptors from extract_global_descriptors()
            existing_pairs: Existing image pairs to avoid (e.g., from sequential/vocab tree)

        Returns:
            List of (img1, img2, similarity) tuples for loop closure candidates
        """
        logger.info("Detecting loop closures...")

        # Convert existing pairs to set for fast lookup
        existing_set = set()
        if existing_pairs:
            for img1, img2 in existing_pairs:
                existing_set.add((min(img1, img2), max(img1, img2)))

        # Get sorted image paths (for temporal ordering)
        image_paths = sorted(global_descriptors.keys())
        logger.info(f"Processing {len(image_paths)} images")

        loop_closures = []

        # Compare each image with temporally distant images
        for i, img1 in enumerate(image_paths):
            desc1 = global_descriptors[img1]

            # Only check images that are temporally far away
            for j in range(i + self.min_temporal_gap, len(image_paths)):
                img2 = image_paths[j]

                # Skip if already connected
                pair_key = (min(img1, img2), max(img1, img2))
                if pair_key in existing_set:
                    continue

                desc2 = global_descriptors[img2]

                # Compute cosine similarity
                similarity = float(np.dot(desc1, desc2))

                # Check if similarity exceeds threshold
                if similarity >= self.similarity_threshold:
                    loop_closures.append((img1, img2, similarity))
                    logger.debug(
                        f"Loop closure candidate: {Path(img1).name} ↔ {Path(img2).name} "
                        f"(similarity={similarity:.3f}, gap={j-i})"
                    )

        logger.info(f"Found {len(loop_closures)} loop closure candidates")

        # Sort by similarity (highest first)
        loop_closures.sort(key=lambda x: x[2], reverse=True)

        return loop_closures

    def verify_loop_closures(
        self,
        loop_closures: List[Tuple[str, str, float]],
        features: Dict[str, Any],
        matcher: Any,
        max_loops_to_verify: int = 100
    ) -> List[Tuple[str, str, Dict]]:
        """
        Verify loop closures using feature matching and geometric verification

        Args:
            loop_closures: Candidate loop closures from detect_loop_closures()
            features: Feature dictionary for matching
            matcher: Feature matcher instance (e.g., EnhancedLightGlueMatcher)
            max_loops_to_verify: Maximum number of loops to verify (for efficiency)

        Returns:
            List of verified loop closures with match data: [(img1, img2, match_dict), ...]
        """
        logger.info(f"Verifying loop closures (max: {max_loops_to_verify})...")

        verified_loops = []

        # Limit number of loops to verify
        loops_to_verify = loop_closures[:max_loops_to_verify]

        for img1, img2, similarity in loops_to_verify:
            try:
                # Prepare features for matching
                feat1 = features[img1]
                feat2 = features[img2]

                # Match features
                if hasattr(matcher, 'match_pair'):
                    # Use match_pair if available
                    match_result = matcher.match_pair(feat1, feat2)
                else:
                    # Fallback: use match_features with subset
                    match_result = matcher.match_features({img1: feat1, img2: feat2})
                    if (img1, img2) in match_result:
                        match_result = match_result[(img1, img2)]
                    elif (img2, img1) in match_result:
                        match_result = match_result[(img2, img1)]
                    else:
                        match_result = None

                if match_result is None:
                    continue

                # Extract matched keypoints
                if 'mkpts0' in match_result and 'mkpts1' in match_result:
                    mkpts0 = match_result['mkpts0']
                    mkpts1 = match_result['mkpts1']
                elif 'matches0' in match_result:
                    # Convert match indices to keypoints
                    matches0 = match_result['matches0']
                    valid = matches0 >= 0
                    kpts0 = feat1['keypoints'][valid]
                    kpts1 = feat2['keypoints'][matches0[valid]]

                    match_result['mkpts0'] = kpts0
                    match_result['mkpts1'] = kpts1
                    mkpts0 = kpts0
                    mkpts1 = kpts1
                else:
                    logger.warning(f"Unexpected match result format for {img1} ↔ {img2}")
                    continue

                num_matches = len(mkpts0)

                # Check if enough matches for reliable verification
                if num_matches >= self.min_matches_for_verification:
                    verified_loops.append((img1, img2, match_result))
                    logger.info(
                        f"✓ Verified loop: {Path(img1).name} ↔ {Path(img2).name} "
                        f"({num_matches} matches, sim={similarity:.3f})"
                    )
                else:
                    logger.debug(
                        f"✗ Rejected loop: {Path(img1).name} ↔ {Path(img2).name} "
                        f"(only {num_matches} matches, need {self.min_matches_for_verification})"
                    )

            except Exception as e:
                logger.warning(f"Error verifying loop {img1} ↔ {img2}: {e}")
                continue

        logger.info(f"Verified {len(verified_loops)}/{len(loops_to_verify)} loop closures")

        return verified_loops

    def add_loops_to_matches(
        self,
        verified_loops: List[Tuple[str, str, Dict]],
        matches: Dict[Tuple[str, str], Any]
    ) -> Dict[Tuple[str, str], Any]:
        """
        Add verified loop closures to the matches dictionary

        Args:
            verified_loops: Verified loop closures with match data
            matches: Existing matches dictionary (will be modified in-place)

        Returns:
            Updated matches dictionary
        """
        logger.info(f"Adding {len(verified_loops)} loop closures to matches...")

        num_added = 0
        for img1, img2, match_data in verified_loops:
            pair_key = (img1, img2)

            # Check if pair already exists
            if pair_key in matches or (img2, img1) in matches:
                logger.debug(f"Loop already in matches: {Path(img1).name} ↔ {Path(img2).name}")
                continue

            # Add to matches
            matches[pair_key] = match_data
            num_added += 1

        logger.info(f"Added {num_added} new loop closure edges to matches")
        logger.info(f"Total matches: {len(matches)} pairs")

        return matches


def detect_and_add_loop_closures(
    features: Dict[str, Any],
    matches: Dict[Tuple[str, str], Any],
    matcher: Any,
    config: Optional[Dict[str, Any]] = None
) -> Dict[Tuple[str, str], Any]:
    """
    Convenience function to detect and add loop closures to matches

    Args:
        features: Feature dictionary from feature extraction
        matches: Existing matches dictionary (will be modified)
        matcher: Feature matcher instance
        config: Optional configuration dict with keys:
                - similarity_threshold (default: 0.70)
                - min_temporal_gap (default: 30)
                - min_matches_for_verification (default: 15)
                - max_loops_to_verify (default: 100)

    Returns:
        Updated matches dictionary with loop closures added
    """
    if config is None:
        config = {}

    # Initialize detector
    detector = LoopClosureDetector(
        descriptor_method='feature_pooling',
        similarity_threshold=config.get('similarity_threshold', 0.70),
        min_temporal_gap=config.get('min_temporal_gap', 30),
        min_matches_for_verification=config.get('min_matches_for_verification', 15)
    )

    # Extract global descriptors
    global_descriptors = detector.extract_global_descriptors(features)

    # Detect loop closures
    existing_pairs = list(matches.keys())
    loop_candidates = detector.detect_loop_closures(
        global_descriptors,
        existing_pairs=existing_pairs
    )

    if len(loop_candidates) == 0:
        logger.info("No loop closures detected")
        return matches

    # Verify loop closures
    verified_loops = detector.verify_loop_closures(
        loop_candidates,
        features,
        matcher,
        max_loops_to_verify=config.get('max_loops_to_verify', 100)
    )

    if len(verified_loops) == 0:
        logger.info("No loop closures verified")
        return matches

    # Add to matches
    updated_matches = detector.add_loops_to_matches(verified_loops, matches)

    return updated_matches
