"""
Benchmark algebraic pre-filtering for graph construction.

Tests how fast we can check orientation compatibility between image pairs
to enable exhaustive graph construction without brute force matching cost.
"""

import time
import numpy as np
from typing import Tuple


def orientation_compatibility_check(
    kpts1: np.ndarray,
    kpts2: np.ndarray,
    ori1: np.ndarray,
    ori2: np.ndarray,
    tau: float = np.radians(30),
    sample_size: int = 100
) -> Tuple[bool, float]:
    """
    Fast orientation compatibility check between two images.

    Args:
        kpts1, kpts2: Keypoint arrays (N×2)
        ori1, ori2: Orientation arrays (N,)
        tau: Orientation threshold (radians)
        sample_size: Number of samples to check

    Returns:
        (compatible, confidence_score)
    """
    n1 = min(sample_size, len(kpts1))
    n2 = min(sample_size, len(kpts2))

    # Sample features
    idx1 = np.random.choice(len(kpts1), n1, replace=False)
    idx2 = np.random.choice(len(kpts2), n2, replace=False)

    # Compute orientation differences for sampled pairs
    ori_diffs = []
    for i in idx1:
        for j in idx2:
            diff = np.abs(ori1[i] - ori2[j])
            diff = min(diff, 2*np.pi - diff)  # Wrap to [0, π]
            ori_diffs.append(diff)

    ori_diffs = np.array(ori_diffs)

    # Compatibility criterion: median orientation difference < tau
    median_diff = np.median(ori_diffs)
    compatible = median_diff < tau

    # Confidence: proportion of pairs within tau
    confidence = np.mean(ori_diffs < tau)

    return compatible, confidence


def benchmark_single_pair():
    """Benchmark a single pair compatibility check."""
    n_features = 2000

    # Simulate features
    kpts1 = np.random.rand(n_features, 2) * 640
    kpts2 = np.random.rand(n_features, 2) * 640
    ori1 = np.random.rand(n_features) * 2 * np.pi
    ori2 = np.random.rand(n_features) * 2 * np.pi

    start = time.time()
    compatible, confidence = orientation_compatibility_check(kpts1, kpts2, ori1, ori2)
    elapsed = time.time() - start

    return elapsed, compatible, confidence


def benchmark_graph_construction(n_images: int):
    """
    Benchmark exhaustive graph construction with algebraic pre-filtering.

    Args:
        n_images: Number of images

    Returns:
        Statistics dict
    """
    n_pairs = n_images * (n_images - 1) // 2
    n_features = 2000

    print(f"\n{'='*60}")
    print(f"Benchmarking graph construction for {n_images} images")
    print(f"Total pairs to check: {n_pairs:,}")
    print(f"{'='*60}\n")

    # Simulate image features
    print("Generating synthetic features...")
    all_kpts = []
    all_ori = []
    for i in range(n_images):
        kpts = np.random.rand(n_features, 2) * 640
        ori = np.random.rand(n_features) * 2 * np.pi
        all_kpts.append(kpts)
        all_ori.append(ori)

    # Pre-filtering phase
    print("Running algebraic pre-filtering...")
    start = time.time()

    compatible_pairs = []
    confidences = []

    for i in range(n_images):
        for j in range(i+1, n_images):
            compatible, confidence = orientation_compatibility_check(
                all_kpts[i], all_kpts[j],
                all_ori[i], all_ori[j],
                sample_size=50  # Smaller sample for speed
            )

            if compatible:
                compatible_pairs.append((i, j))
                confidences.append(confidence)

    prefilter_time = time.time() - start

    # Statistics
    rejection_rate = 1 - (len(compatible_pairs) / n_pairs)
    avg_confidence = np.mean(confidences) if confidences else 0

    print(f"\nPre-filtering Results:")
    print(f"  Total time: {prefilter_time:.2f}s")
    print(f"  Time per pair: {prefilter_time/n_pairs*1000:.3f}ms")
    print(f"  Compatible pairs: {len(compatible_pairs):,} / {n_pairs:,}")
    print(f"  Rejection rate: {rejection_rate*100:.1f}%")
    print(f"  Avg confidence: {avg_confidence:.3f}")

    # Estimate downstream cost
    print(f"\nDownstream Cost Estimates:")

    # Scenario 1: Naive brute force (LightGlue on all pairs)
    naive_matching_time = n_pairs * 0.05  # 50ms per pair
    print(f"  Naive brute force (LightGlue all pairs): {naive_matching_time:.1f}s")

    # Scenario 2: Algebraic graph (LightGlue on compatible pairs only)
    smart_matching_time = len(compatible_pairs) * 0.05
    total_smart_time = prefilter_time + smart_matching_time
    print(f"  Algebraic graph (pre-filter + LightGlue compatible): {total_smart_time:.1f}s")
    print(f"    - Pre-filtering: {prefilter_time:.1f}s")
    print(f"    - Matching: {smart_matching_time:.1f}s")

    # Scenario 3: Vocab tree baseline
    vocab_pairs = min(200, n_pairs)  # Top-200 from vocab tree
    vocab_matching_time = vocab_pairs * 0.05
    vocab_total = 0.5 + vocab_matching_time  # 0.5s for vocab tree
    print(f"  Vocab tree baseline (top-200): {vocab_total:.1f}s")

    speedup_vs_naive = naive_matching_time / total_smart_time
    speedup_vs_vocab = vocab_total / total_smart_time

    print(f"\nSpeedup:")
    print(f"  vs Naive brute force: {speedup_vs_naive:.1f}x faster")
    print(f"  vs Vocab tree: {speedup_vs_vocab:.2f}x {'faster' if speedup_vs_vocab > 1 else 'slower'}")

    return {
        'n_images': n_images,
        'n_pairs': n_pairs,
        'compatible_pairs': len(compatible_pairs),
        'rejection_rate': rejection_rate,
        'prefilter_time': prefilter_time,
        'time_per_pair_ms': prefilter_time/n_pairs*1000,
        'total_smart_time': total_smart_time,
        'speedup_vs_naive': speedup_vs_naive,
        'speedup_vs_vocab': speedup_vs_vocab
    }


if __name__ == "__main__":
    print("="*60)
    print("Algebraic Pre-filtering Benchmark")
    print("="*60)

    # Warmup
    print("\nWarmup run...")
    for _ in range(10):
        benchmark_single_pair()

    # Single pair benchmark
    print("\n" + "="*60)
    print("Single Pair Benchmark (100 trials)")
    print("="*60)

    times = []
    for _ in range(100):
        t, _, _ = benchmark_single_pair()
        times.append(t)

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"\nAverage time per pair: {avg_time*1000:.3f} ± {std_time*1000:.3f}ms")
    print(f"Throughput: {1/avg_time:.0f} pairs/second")

    # Graph construction benchmarks
    for n_images in [50, 100, 200, 500]:
        stats = benchmark_graph_construction(n_images)

        # Stop if getting too slow
        if stats['total_smart_time'] > 60:
            print(f"\n⚠ Stopping benchmark (>60s for {n_images} images)")
            break

    print("\n" + "="*60)
    print("Benchmark Complete")
    print("="*60)
