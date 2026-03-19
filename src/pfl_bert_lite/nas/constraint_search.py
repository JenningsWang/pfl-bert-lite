from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SubnetworkCandidate:
    name: str
    layers: int
    hidden_size: int
    intermediate_size: int
    attention_heads: int
    lora_rank: int
    estimated_latency_ms: float
    estimated_params_m: float

    @classmethod
    def from_mapping(cls, values: dict) -> "SubnetworkCandidate":
        return cls(
            name=str(values["name"]),
            layers=int(values["layers"]),
            hidden_size=int(values["hidden_size"]),
            intermediate_size=int(values["intermediate_size"]),
            attention_heads=int(values["attention_heads"]),
            lora_rank=int(values["lora_rank"]),
            estimated_latency_ms=float(values["estimated_latency_ms"]),
            estimated_params_m=float(values["estimated_params_m"]),
        )



def default_candidates() -> list[SubnetworkCandidate]:
    return [
        SubnetworkCandidate("bert-mini-lora4", 4, 256, 1024, 4, 4, 22.0, 4.8),
        SubnetworkCandidate("bert-small-lora8", 6, 384, 1536, 6, 8, 38.0, 11.2),
        SubnetworkCandidate("bert-base-lite-lora8", 8, 512, 2048, 8, 8, 57.0, 18.7),
    ]



def score_candidate(
    candidate: SubnetworkCandidate,
    client_scale: float,
    topology_bonus: float,
) -> float:
    capacity = (
        0.35 * candidate.layers / 12.0
        + 0.30 * candidate.hidden_size / 768.0
        + 0.20 * candidate.intermediate_size / 3072.0
        + 0.15 * candidate.lora_rank / 16.0
    )
    return capacity * (0.70 + 0.30 * client_scale) + 0.10 * topology_bonus



def choose_candidate(
    candidates: Iterable[SubnetworkCandidate],
    latency_budget_ms: float,
    parameter_budget_m: float,
    client_scale: float,
    topology_bonus: float,
) -> SubnetworkCandidate:
    candidates_list = list(candidates)
    feasible = [
        candidate
        for candidate in candidates_list
        if candidate.estimated_latency_ms <= latency_budget_ms
        and candidate.estimated_params_m <= parameter_budget_m
    ]
    if feasible:
        return max(
            feasible,
            key=lambda candidate: score_candidate(candidate, client_scale, topology_bonus),
        )

    return min(
        candidates_list,
        key=lambda candidate: (
            max(0.0, candidate.estimated_latency_ms - latency_budget_ms)
            + max(0.0, candidate.estimated_params_m - parameter_budget_m),
            candidate.estimated_latency_ms,
        ),
    )
