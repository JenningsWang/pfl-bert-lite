from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GraphWeights:
    dataset_scale: float = 0.35
    label_similarity: float = 0.30
    structural_similarity: float = 0.20
    gradient_alignment: float = 0.15

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> "GraphWeights":
        defaults = cls()
        if not values:
            return defaults
        return cls(
            dataset_scale=float(values.get("dataset_scale", defaults.dataset_scale)),
            label_similarity=float(values.get("label_similarity", defaults.label_similarity)),
            structural_similarity=float(
                values.get("structural_similarity", defaults.structural_similarity)
            ),
            gradient_alignment=float(
                values.get("gradient_alignment", defaults.gradient_alignment)
            ),
        )


@dataclass
class ExperimentConfig:
    seed: int = 42
    rounds: int = 3
    local_steps: int = 10
    client_top_k: int = 2
    consensus_steps: int = 2
    latency_budget_ms: float = 55.0
    parameter_budget_m: float = 15.0
    graph_weights: GraphWeights = field(default_factory=GraphWeights)
    candidate_pool: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        defaults = cls()
        return cls(
            seed=int(payload.get("seed", defaults.seed)),
            rounds=int(payload.get("rounds", defaults.rounds)),
            local_steps=int(payload.get("local_steps", defaults.local_steps)),
            client_top_k=int(payload.get("client_top_k", defaults.client_top_k)),
            consensus_steps=int(payload.get("consensus_steps", defaults.consensus_steps)),
            latency_budget_ms=float(
                payload.get("latency_budget_ms", defaults.latency_budget_ms)
            ),
            parameter_budget_m=float(
                payload.get("parameter_budget_m", defaults.parameter_budget_m)
            ),
            graph_weights=GraphWeights.from_mapping(payload.get("graph_weights")),
            candidate_pool=list(payload.get("candidate_pool", defaults.candidate_pool)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

