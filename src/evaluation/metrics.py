from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set


def precision_at_k(recommended: Sequence[str], relevant: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    recs = recommended[:k]
    if not recs:
        return 0.0
    hits = sum(1 for r in recs if r in relevant)
    return hits / k


def recall_at_k(recommended: Sequence[str], relevant: Set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    recs = recommended[:k]
    hits = sum(1 for r in recs if r in relevant)
    return hits / len(relevant)


def average_precision(recommended: Sequence[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = 0
    precisions: List[float] = []
    for idx, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            precisions.append(hits / (idx + 1))
    if not precisions:
        return 0.0
    return sum(precisions) / min(len(relevant), k)


def mean_average_precision(rec_map: Mapping[str, Sequence[str]], rel_map: Mapping[str, Set[str]], k: int) -> float:
    if not rec_map:
        return 0.0
    aps = []
    for user, recs in rec_map.items():
        rel = rel_map.get(user, set())
        aps.append(average_precision(recs, rel, k))
    return sum(aps) / len(aps) if aps else 0.0


def ndcg_at_k(recommended: Sequence[str], relevant: Set[str], k: int) -> float:
    dcg = 0.0
    for idx, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / math.log2(idx + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def coverage(rec_map: Mapping[str, Sequence[str]], catalog: Set[str]) -> float:
    if not catalog:
        return 0.0
    recommended_items = {item for recs in rec_map.values() for item in recs}
    return len(recommended_items) / len(catalog)


def diversity(rec_map: Mapping[str, Sequence[str]], item_features: Optional[Mapping[str, Set[str]]] = None) -> Optional[float]:
    """Average pairwise dissimilarity (1 - Jaccard) across recommendations.

    If item_features is None, returns None.
    """
    if item_features is None:
        return None

    pair_scores: List[float] = []
    for recs in rec_map.values():
        n = len(recs)
        if n < 2:
            continue
        for i in range(n):
            fi = item_features.get(recs[i]) or set()
            for j in range(i + 1, n):
                fj = item_features.get(recs[j]) or set()
                if not fi and not fj:
                    pair_scores.append(1.0)
                    continue
                inter = len(fi & fj)
                union = len(fi | fj)
                sim = inter / union if union else 0.0
                pair_scores.append(1.0 - sim)
    if not pair_scores:
        return None
    return sum(pair_scores) / len(pair_scores)


def evaluate(
    rec_map: Mapping[str, Sequence[str]],
    rel_map: Mapping[str, Set[str]],
    k: int = 10,
    catalog: Optional[Set[str]] = None,
    item_features: Optional[Mapping[str, Set[str]]] = None,
) -> Dict[str, float]:
    """Compute offline metrics given recommendations and ground truth.

    rec_map: user -> ranked list of event_ids
    rel_map: user -> set of relevant event_ids (e.g., attended in holdout)
    """
    users = list(rec_map.keys())
    if not users:
        return {
            "precision@k": 0.0,
            "recall@k": 0.0,
            "map": 0.0,
            "ndcg": 0.0,
            "coverage": 0.0,
            "diversity": 0.0,
        }

    precisions = []
    recalls = []
    ndcgs = []
    for u in users:
        recs = rec_map[u]
        rel = rel_map.get(u, set())
        precisions.append(precision_at_k(recs, rel, k))
        recalls.append(recall_at_k(recs, rel, k))
        ndcgs.append(ndcg_at_k(recs, rel, k))

    metrics = {
        "precision@k": sum(precisions) / len(precisions),
        "recall@k": sum(recalls) / len(recalls),
        "map": mean_average_precision(rec_map, rel_map, k),
        "ndcg": sum(ndcgs) / len(ndcgs),
    }

    if catalog is not None:
        metrics["coverage"] = coverage(rec_map, catalog)
    else:
        metrics["coverage"] = 0.0

    div = diversity(rec_map, item_features)
    metrics["diversity"] = div if div is not None else 0.0

    return metrics


if __name__ == "__main__":
    # Example offline evaluation with toy data
    recs = {
        "U1": ["E1", "E2", "E3"],
        "U2": ["E2", "E4"],
    }
    truth = {
        "U1": {"E2", "E3"},
        "U2": {"E4", "E5"},
    }
    catalog = {"E1", "E2", "E3", "E4", "E5", "E6"}
    features = {
        "E1": {"music"},
        "E2": {"music", "dance"},
        "E3": {"dance"},
        "E4": {"theatre"},
        "E5": {"music"},
    }
    out = evaluate(recs, truth, k=2, catalog=catalog, item_features=features)
    print(out)
