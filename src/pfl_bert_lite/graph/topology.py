from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from pfl_bert_lite.config import GraphWeights


@dataclass(frozen=True)
class ClientProfile:
    client_id: str
    sample_count: int
    label_histogram: Sequence[float]
    structure_signature: Sequence[float]
    gradient_vector: Sequence[float] | None = None


@dataclass(frozen=True)
class EdgeScore:
    dataset_scale: float
    label_similarity: float
    structural_similarity: float
    gradient_alignment: float
    total: float



def _safe_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))



def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = _safe_norm(left)
    right_norm = _safe_norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot = sum(x * y for x, y in zip(left, right))
    return dot / (left_norm * right_norm)



def _normalize_distribution(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    total = sum(values)
    if total <= 0.0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in values]



def _js_divergence(left: Sequence[float], right: Sequence[float], eps: float = 1e-12) -> float:
    p = _normalize_distribution(left)
    q = _normalize_distribution(right)
    width = max(len(p), len(q))
    p = p + [0.0] * (width - len(p))
    q = q + [0.0] * (width - len(q))
    midpoint = [(x + y) / 2.0 for x, y in zip(p, q)]

    def _kl(base: Sequence[float], ref: Sequence[float]) -> float:
        total = 0.0
        for base_value, ref_value in zip(base, ref):
            if base_value <= 0.0:
                continue
            total += base_value * math.log((base_value + eps) / (ref_value + eps), 2)
        return total

    return 0.5 * (_kl(p, midpoint) + _kl(q, midpoint))



def edge_score(
    left: ClientProfile,
    right: ClientProfile,
    weights: GraphWeights,
) -> EdgeScore:
    scale_ratio = min(left.sample_count, right.sample_count) / max(
        max(left.sample_count, right.sample_count), 1
    )
    label_similarity = max(0.0, 1.0 - _js_divergence(left.label_histogram, right.label_histogram))
    structural_similarity = max(0.0, _cosine_similarity(left.structure_signature, right.structure_signature))
    gradient_alignment = 0.0
    if left.gradient_vector and right.gradient_vector:
        gradient_alignment = max(0.0, _cosine_similarity(left.gradient_vector, right.gradient_vector))

    total = (
        weights.dataset_scale * scale_ratio
        + weights.label_similarity * label_similarity
        + weights.structural_similarity * structural_similarity
        + weights.gradient_alignment * gradient_alignment
    )
    return EdgeScore(
        dataset_scale=scale_ratio,
        label_similarity=label_similarity,
        structural_similarity=structural_similarity,
        gradient_alignment=gradient_alignment,
        total=total,
    )



def build_dynamic_graph(
    profiles: Sequence[ClientProfile],
    weights: GraphWeights,
    top_k: int = 2,
    minimum_edge: float = 0.10,
) -> list[list[float]]:
    size = len(profiles)
    adjacency = [[0.0 for _ in range(size)] for _ in range(size)]
    for index in range(size):
        adjacency[index][index] = 1.0

    for source in range(size):
        candidates: list[tuple[int, float]] = []
        for target in range(size):
            if source == target:
                continue
            score = edge_score(profiles[source], profiles[target], weights).total
            candidates.append((target, score))
        candidates.sort(key=lambda item: item[1], reverse=True)
        selected = candidates[: max(1, top_k)]
        for target, score in selected:
            if score >= minimum_edge or not any(value > 0.0 for _, value in selected[:-1]):
                adjacency[source][target] = score
                adjacency[target][source] = max(adjacency[target][source], score)

    return adjacency
