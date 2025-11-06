from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import networkx as nx


def maximum_spanning_tree(
    nodes: Sequence[str],
    weighted_edges: Iterable[Tuple[str, str, float]],
) -> List[Tuple[str, str, float]]:
    """Compute maximum spanning tree over provided nodes."""
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for u, v, w in weighted_edges:
        if u == v:
            continue
        graph.add_edge(u, v, weight=float(w))

    tree = nx.maximum_spanning_tree(graph, weight="weight")
    result = []
    for u, v, data in tree.edges(data=True):
        result.append((u, v, float(data.get("weight", 0.0))))
    return sorted(result)
