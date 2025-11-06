#!/usr/bin/env python3
"""
Quick example demonstrating the modular matching framework.

This script shows how to:
1. Extract features with different extractors
2. Match features with different matchers
3. Switch components dynamically
"""

from pathlib import Path
from matching_framework import (
    ExtractorFactory,
    ExtractorConfig,
    MatcherFactory,
    MatcherConfig,
    PairSelectorFactory,
    PairSelectorConfig,
)


def main():
    # ========== Setup ==========
    print("=" * 60)
    print("Modular Feature Matching Framework - Quick Example")
    print("=" * 60)

    # Paths (adjust these to your data)
    image_dir = Path("data/images")
    output_dir = Path("results/example")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get image list
    image_list = sorted(image_dir.glob("*.jpg"))[:10]  # First 10 images
    print(f"\nFound {len(image_list)} images")

    if len(image_list) < 2:
        print("Error: Need at least 2 images in data/images/")
        return

    # ========== Example 1: SIFT + NN Matcher ==========
    print("\n" + "-" * 60)
    print("Example 1: SIFT + NN Matcher (CPU-friendly)")
    print("-" * 60)

    # Extract SIFT features
    extractor_config = ExtractorConfig(max_keypoints=2048, device="cpu")
    extractor = ExtractorFactory.create("sift", extractor_config)

    print("Extracting SIFT features...")
    features_dict = extractor.extract_batch(image_list)
    print(f"Extracted features from {len(features_dict)} images")

    # Select pairs (exhaustive)
    pair_selector_config = PairSelectorConfig()
    pair_selector = PairSelectorFactory.create("exhaustive", pair_selector_config)

    pairs = pair_selector.select(image_list)
    print(f"Generated {len(pairs)} pairs")

    # Match with NN
    matcher_config = MatcherConfig(distance_threshold=0.7, device="cpu")
    matcher = MatcherFactory.create("nn", matcher_config, extractor.get_config_dict())

    print("Matching features...")
    matches_dict = matcher.match_pairs(features_dict, pairs)

    # Print results
    for (path0, path1), matches in list(matches_dict.items())[:3]:
        print(f"  {path0.name} <-> {path1.name}: {len(matches.matches0)} matches")

    # ========== Example 2: SuperPoint + LightGlue ==========
    print("\n" + "-" * 60)
    print("Example 2: SuperPoint + LightGlue (GPU-accelerated)")
    print("-" * 60)

    try:
        # Extract SuperPoint features
        extractor_config = ExtractorConfig(max_keypoints=4096, device="cuda")
        extractor = ExtractorFactory.create("superpoint", extractor_config)

        print("Extracting SuperPoint features...")
        features_dict = extractor.extract_batch(image_list)
        print(f"Extracted features from {len(features_dict)} images")

        # Match with LightGlue
        matcher_config = MatcherConfig(device="cuda")
        matcher = MatcherFactory.create(
            "lightglue", matcher_config, extractor.get_config_dict()
        )

        print("Matching with LightGlue...")
        matches_dict = matcher.match_pairs(features_dict, pairs)

        # Print results
        for (path0, path1), matches in list(matches_dict.items())[:3]:
            print(f"  {path0.name} <-> {path1.name}: {len(matches.matches0)} matches")

    except Exception as e:
        print(f"Skipping GPU example (CUDA not available or LightGlue not installed)")
        print(f"Error: {e}")

    # ========== Example 3: Dynamic Switching ==========
    print("\n" + "-" * 60)
    print("Example 3: Comparing Different Extractors")
    print("-" * 60)

    extractors_to_test = ["sift", "orb"]
    config = ExtractorConfig(max_keypoints=2048, device="cpu")

    for extractor_name in extractors_to_test:
        print(f"\nTesting {extractor_name.upper()}:")

        # Create extractor
        extractor = ExtractorFactory.create(extractor_name, config)

        # Extract features from first image
        features = extractor.extract(image_list[0])

        print(f"  Keypoints: {len(features.keypoints)}")
        print(f"  Descriptor dim: {extractor.descriptor_dim}")
        print(f"  Descriptor type: {extractor.descriptor_type}")

    # ========== Summary ==========
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print("Available extractors:", ExtractorFactory.list_extractors())
    print("Available matchers:", MatcherFactory.list_matchers())
    print("Available pair selectors:", PairSelectorFactory.list_selectors())
    print("\nFramework is ready to use!")
    print("=" * 60)


if __name__ == "__main__":
    main()
