import argparse
from pathlib import Path
import sys
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from extractors import ExtractorFactory, ExtractorConfig
from matchers import MatcherFactory, MatcherConfig
from pair_selectors import PairSelectorFactory, PairSelectorConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Modular feature matching framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output
    parser.add_argument("--images", type=str, required=True, help="Image directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")

    # Feature extractor
    parser.add_argument(
        "--extractor",
        type=str,
        default="superpoint",
        choices=ExtractorFactory.list_extractors(),
        help="Feature extractor"
    )
    parser.add_argument("--max_keypoints", type=int, default=4096, help="Max keypoints per image")
    parser.add_argument("--resize_max", type=int, default=None, help="Resize max dimension")

    # Matcher
    parser.add_argument(
        "--matcher",
        type=str,
        default="lightglue",
        choices=MatcherFactory.list_matchers(),
        help="Feature matcher"
    )
    parser.add_argument("--distance_threshold", type=float, default=0.7, help="Distance threshold")
    parser.add_argument("--no_mutual_check", action="store_true", help="Disable mutual check")

    # Pair selector
    parser.add_argument(
        "--pair_selector",
        type=str,
        default="exhaustive",
        choices=PairSelectorFactory.list_selectors(),
        help="Pair selection strategy"
    )
    parser.add_argument("--max_pairs_per_image", type=int, default=20, help="Max pairs per image")

    # Geometric verification
    parser.add_argument("--geometric_verification", action="store_true", help="Run RANSAC")
    parser.add_argument("--ransac_threshold", type=float, default=1.0, help="RANSAC threshold (px)")

    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    # Misc
    parser.add_argument("--save_visualization", action="store_true", help="Save match visualizations")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup paths
    image_dir = Path(args.images)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get image list
    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    image_list = sorted([
        p for p in image_dir.iterdir()
        if p.suffix in image_extensions
    ])

    logger.info(f"Found {len(image_list)} images")

    # ========== STEP 1: Feature Extraction ==========
    logger.info(f"Extracting features with {args.extractor}...")

    extractor_config = ExtractorConfig(
        max_keypoints=args.max_keypoints,
        resize_max=args.resize_max,
        device=args.device,
    )

    extractor = ExtractorFactory.create(args.extractor, extractor_config)
    features_dict = extractor.extract_batch(image_list)

    logger.info(f"Extracted features from {len(features_dict)} images")

    # Save features
    import pickle
    features_path = output_dir / "features.pkl"
    with open(features_path, "wb") as f:
        pickle.dump(features_dict, f)
    logger.info(f"Saved features to {features_path}")

    # ========== STEP 2: Pair Selection ==========
    logger.info(f"Selecting pairs with {args.pair_selector}...")

    pair_selector_config = PairSelectorConfig(
        max_pairs_per_image=args.max_pairs_per_image,
        device=args.device,
        extra={
            "knn_k": 30,
            "descriptor_metric": "cosine" if args.extractor != "sift" else "l2"
        }
    )

    pair_selector = PairSelectorFactory.create(args.pair_selector, pair_selector_config)

    # MPA needs features for selection
    if args.pair_selector == "mpa":
        pairs = pair_selector.select(image_list, features=features_dict)
    else:
        pairs = pair_selector.select(image_list)

    logger.info(f"Selected {len(pairs)} pairs")

    # Statistics
    stats = pair_selector.get_statistics(image_list, pairs)
    logger.info(f"Pair statistics: {stats}")

    # ========== STEP 3: Feature Matching ==========
    logger.info(f"Matching features with {args.matcher}...")

    matcher_config = MatcherConfig(
        distance_threshold=args.distance_threshold,
        mutual_check=not args.no_mutual_check,
        device=args.device,
    )

    matcher = MatcherFactory.create(
        args.matcher,
        matcher_config,
        extractor.get_config_dict()
    )

    matches_dict = matcher.match_pairs(features_dict, pairs)

    logger.info(f"Matched {len(matches_dict)} pairs")

    # ========== STEP 4: Geometric Verification (optional) ==========
    if args.geometric_verification:
        logger.info("Running geometric verification...")

        verified_matches = {}
        for (path0, path1), matches in matches_dict.items():
            verified, F = matcher.geometric_verification(
                features_dict[path0],
                features_dict[path1],
                matches,
                threshold=args.ransac_threshold
            )
            verified_matches[(path0, path1)] = verified
            logger.info(
                f"{path0.name} <-> {path1.name}: "
                f"{len(matches)} -> {len(verified)} matches"
            )

        matches_dict = verified_matches

    # Save matches
    matches_path = output_dir / "matches.pkl"
    with open(matches_path, "wb") as f:
        pickle.dump(matches_dict, f)
    logger.info(f"Saved matches to {matches_path}")

    # ========== STEP 5: Visualization (optional) ==========
    if args.save_visualization:
        logger.info("Generating visualizations...")
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)

        # Visualize top pairs
        import cv2
        import numpy as np

        for (path0, path1), matches in list(matches_dict.items())[:10]:
            img0 = cv2.imread(str(path0))
            img1 = cv2.imread(str(path1))

            kpts0, kpts1 = matches.to_keypoints(
                features_dict[path0].keypoints,
                features_dict[path1].keypoints
            )

            # Draw matches
            h0, w0 = img0.shape[:2]
            h1, w1 = img1.shape[:2]
            h = max(h0, h1)
            vis = np.zeros((h, w0 + w1, 3), dtype=np.uint8)
            vis[:h0, :w0] = img0
            vis[:h1, w0:] = img1

            for (x0, y0), (x1, y1) in zip(kpts0, kpts1):
                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.circle(vis, (int(x0), int(y0)), 3, color, -1)
                cv2.circle(vis, (int(x1 + w0), int(y1)), 3, color, -1)
                cv2.line(vis, (int(x0), int(y0)), (int(x1 + w0), int(y1)), color, 1)

            vis_path = vis_dir / f"{path0.stem}_{path1.stem}.jpg"
            cv2.imwrite(str(vis_path), vis)

        logger.info(f"Saved visualizations to {vis_dir}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
