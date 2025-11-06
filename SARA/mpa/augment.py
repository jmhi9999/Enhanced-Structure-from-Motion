from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import networkx as nx

Edge = Tuple[str, str, float]


def _canonical(u: str, v: str) -> Tuple[str, str]:
    return (u, v) if u <= v else (v, u)


def leaf_augment(
    nodes: Sequence[str],
    all_edges: Iterable[Edge],
    initial_edges: Sequence[Edge],
    deg_cap: int | None,
) -> List[Edge]:
    """Add high-score edges incident to leaves to reduce degree-1 nodes."""
    selected: List[Edge] = list(initial_edges)
    selected_set = {_canonical(u, v) for u, v, _ in selected}
    degree = Counter()
    for u, v, _ in selected:
        degree[u] += 1
        degree[v] += 1

    candidate_by_leaf = {}
    sorted_edges = sorted(all_edges, key=lambda e: e[2], reverse=True)
    for u, v, w in sorted_edges:
        key = _canonical(u, v)
        if key in selected_set or u == v:
            continue
        candidate_by_leaf.setdefault(u, []).append((u, v, w))
        candidate_by_leaf.setdefault(v, []).append((u, v, w))

    leaves = [node for node in nodes if degree[node] <= 1]
    for leaf in leaves:
        if leaf not in candidate_by_leaf:
            continue
        for u, v, w in candidate_by_leaf[leaf]:
            other = v if leaf == u else u
            if _canonical(u, v) in selected_set:
                continue
            if deg_cap is not None:
                if degree[u] >= deg_cap or degree[v] >= deg_cap:
                    continue
            selected.append((u, v, w))
            selected_set.add(_canonical(u, v))
            degree[u] += 1
            degree[v] += 1
            break

    return selected


def triangle_gain_augment(
    nodes: Sequence[str],
    all_edges: Iterable[Edge],
    existing_edges: Sequence[Edge],
    budget: int,
) -> List[Edge]:
    """Add a limited number of high-parallax loop edges."""
    if budget <= 0:
        return list(existing_edges)

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for u, v, w in existing_edges:
        graph.add_edge(u, v, weight=w)

    selected_set = {_canonical(u, v) for u, v, _ in existing_edges}
    result = list(existing_edges)
    added = 0

    candidate_edges = sorted(all_edges, key=lambda e: e[2], reverse=True)
    for u, v, w in candidate_edges:
        if added >= budget:
            break
        key = _canonical(u, v)
        if key in selected_set or not graph.has_node(u) or not graph.has_node(v):
            continue
        if graph.has_edge(u, v):
            continue
        try:
            path_length = nx.shortest_path_length(graph, u, v)
        except nx.NetworkXNoPath:
            path_length = float("inf")
        if path_length <= 1:
            continue
        graph.add_edge(u, v, weight=w)
        result.append((u, v, w))
        selected_set.add(key)
        added += 1

    return result


def multi_scale_loop_augmentation(
    nodes: Sequence[str],
    all_edges: Iterable[Edge],
    existing_edges: Sequence[Edge],
    budget: int,
    small_ratio: float = 0.5,
    medium_ratio: float = 0.3,
    large_ratio: float = 0.2,
) -> List[Edge]:
    """
    Add loops of different path lengths for multi-scale robustness.

    Args:
        nodes: All graph nodes
        all_edges: All candidate edges with scores
        existing_edges: Current edges in graph
        budget: Total loop budget
        small_ratio: Fraction for small loops (path length 2)
        medium_ratio: Fraction for medium loops (path length 3-4)
        large_ratio: Fraction for large loops (path length 5+)

    Returns:
        Existing edges + selected multi-scale loops
    """
    if budget <= 0:
        return list(existing_edges)

    # Build graph from existing edges
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for u, v, w in existing_edges:
        graph.add_edge(u, v, weight=w)

    selected_set = {_canonical(u, v) for u, v, _ in existing_edges}

    # Categorize candidate loops by path length
    small_loops: List[Edge] = []
    medium_loops: List[Edge] = []
    large_loops: List[Edge] = []

    candidate_edges = sorted(all_edges, key=lambda e: e[2], reverse=True)

    for u, v, w in candidate_edges:
        key = _canonical(u, v)
        if key in selected_set or not graph.has_node(u) or not graph.has_node(v):
            continue
        if graph.has_edge(u, v):
            continue

        try:
            path_length = nx.shortest_path_length(graph, u, v)
        except nx.NetworkXNoPath:
            continue

        if path_length == 2:
            small_loops.append((u, v, w))
        elif path_length in [3, 4]:
            medium_loops.append((u, v, w))
        elif path_length >= 5:
            large_loops.append((u, v, w))

    # Allocate budget
    n_small = int(budget * small_ratio)
    n_medium = int(budget * medium_ratio)
    n_large = int(budget * large_ratio)

    # Ensure we use the full budget
    allocated = n_small + n_medium + n_large
    if allocated < budget:
        n_small += budget - allocated

    # Select loops
    result = list(existing_edges)
    added_count = 0

    for edge in small_loops[:n_small]:
        u, v, w = edge
        if not graph.has_edge(u, v):
            result.append(edge)
            graph.add_edge(u, v, weight=w)
            selected_set.add(_canonical(u, v))
            added_count += 1

    for edge in medium_loops[:n_medium]:
        u, v, w = edge
        if not graph.has_edge(u, v):
            result.append(edge)
            graph.add_edge(u, v, weight=w)
            selected_set.add(_canonical(u, v))
            added_count += 1

    for edge in large_loops[:n_large]:
        u, v, w = edge
        if not graph.has_edge(u, v):
            result.append(edge)
            graph.add_edge(u, v, weight=w)
            selected_set.add(_canonical(u, v))
            added_count += 1

    return result


def add_long_baseline_anchors(
    nodes: Sequence[str],
    all_edges: Iterable[Edge],
    existing_edges: Sequence[Edge],
    embeddings: Dict[str, np.ndarray],
    anchor_count: int = 10,
    percentile: float = 0.95,
) -> List[Edge]:
    """
    Add long-baseline edges for scale stability.

    Args:
        nodes: All graph nodes
        all_edges: All candidate edges with scores
        existing_edges: Current edges in graph
        embeddings: DINO embeddings for computing baseline proxy
        anchor_count: Number of anchor edges to add
        percentile: Percentile threshold for baseline length

    Returns:
        Existing edges + anchor edges
    """
    if anchor_count <= 0:
        return list(existing_edges)

    # Build graph
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for u, v, _ in existing_edges:
        graph.add_edge(u, v)

    selected_set = {_canonical(u, v) for u, v, _ in existing_edges}

    # Compute baseline proxy (inverse of DINO cosine similarity)
    anchor_candidates: List[Tuple[str, str, float, float]] = []

    for u, v, score in all_edges:
        key = _canonical(u, v)
        if key in selected_set:
            continue
        if graph.has_edge(u, v):
            continue

        # Baseline proxy: inverse of cosine similarity
        if u in embeddings and v in embeddings:
            emb_u = embeddings[u] / (np.linalg.norm(embeddings[u]) + 1e-8)
            emb_v = embeddings[v] / (np.linalg.norm(embeddings[v]) + 1e-8)
            cosine_sim = np.dot(emb_u, emb_v)
            baseline_proxy = 1.0 - cosine_sim
        else:
            baseline_proxy = 0.0

        anchor_candidates.append((u, v, score, baseline_proxy))

    if not anchor_candidates:
        return list(existing_edges)

    # Filter by baseline percentile
    baseline_values = [b for _, _, _, b in anchor_candidates]
    threshold = np.percentile(baseline_values, percentile * 100)

    long_baseline_edges = [
        (u, v, score) for u, v, score, baseline in anchor_candidates
        if baseline >= threshold
    ]

    # Sort by score and take top-k
    long_baseline_edges.sort(key=lambda e: e[2], reverse=True)
    anchors = long_baseline_edges[:anchor_count]

    result = list(existing_edges)
    result.extend(anchors)

    return result


def reinforce_weak_views(
    nodes: Sequence[str],
    all_edges: Iterable[Edge],
    existing_edges: Sequence[Edge],
    features: Dict[str, Dict[str, np.ndarray]],
    percentile: float = 0.20,
    extra_edges: int = 2,
) -> List[Edge]:
    """
    Add extra edges to weak views (low feature quality).

    Args:
        nodes: All graph nodes
        all_edges: All candidate edges with scores
        existing_edges: Current edges in graph
        features: Feature data with keypoints and scores
        percentile: Percentile threshold for weak views
        extra_edges: Number of additional edges per weak view

    Returns:
        Existing edges + weak view reinforcement edges
    """
    if extra_edges <= 0:
        return list(existing_edges)

    # Compute feature strength per view
    feature_strength: Dict[str, float] = {}
    for node in nodes:
        if node not in features:
            feature_strength[node] = 0.0
            continue

        feat = features[node]
        kpts = feat.get("kpt", feat.get("keypoints", np.array([])))
        scores = feat.get("score", feat.get("scores", np.array([])))

        if len(kpts) == 0 or len(scores) == 0:
            strength = 0.0
        else:
            # Strength = number of keypoints × average score
            strength = float(len(kpts) * np.mean(scores))

        feature_strength[node] = strength

    # Identify weak views (bottom percentile)
    if not feature_strength:
        return list(existing_edges)

    strength_values = list(feature_strength.values())
    threshold = np.percentile(strength_values, percentile * 100)
    weak_views = [node for node, strength in feature_strength.items()
                  if strength <= threshold]

    if not weak_views:
        return list(existing_edges)

    # Build graph to check current degree
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for u, v, _ in existing_edges:
        graph.add_edge(u, v)

    selected_set = {_canonical(u, v) for u, v, _ in existing_edges}

    # Add extra edges for each weak view
    result = list(existing_edges)

    for weak_node in weak_views:
        # Find top-scoring incident edges not yet selected
        incident_edges: List[Edge] = []
        for u, v, score in all_edges:
            if u != weak_node and v != weak_node:
                continue
            key = _canonical(u, v)
            if key in selected_set:
                continue
            if graph.has_edge(u, v):
                continue
            incident_edges.append((u, v, score))

        # Sort by score and take top-k
        incident_edges.sort(key=lambda e: e[2], reverse=True)

        added = 0
        for edge in incident_edges:
            if added >= extra_edges:
                break
            u, v, score = edge
            result.append(edge)
            graph.add_edge(u, v)
            selected_set.add(_canonical(u, v))
            added += 1

    return result
