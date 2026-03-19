from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pfl_bert_lite.aggregation.consensus import (
    apply_consensus,
    matrix_power,
    metropolis_hastings_weights,
)
from pfl_bert_lite.config import ExperimentConfig
from pfl_bert_lite.graph.topology import ClientProfile, build_dynamic_graph
from pfl_bert_lite.nas.constraint_search import (
    SubnetworkCandidate,
    choose_candidate,
    default_candidates,
)


@dataclass
class ClientRoundInput:
    profile: ClientProfile
    local_update: dict[str, Any]
    latency_budget_ms: float
    parameter_budget_m: float


@dataclass
class RoundArtifacts:
    adjacency: list[list[float]]
    consensus_matrix: list[list[float]]
    selected_subnetworks: dict[str, str]
    aggregated_updates: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "adjacency": self.adjacency,
            "consensus_matrix": self.consensus_matrix,
            "selected_subnetworks": self.selected_subnetworks,
            "aggregated_updates": self.aggregated_updates,
        }


class PersonalizedFederatedTrainer:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.candidates = self._load_candidates()

    def _load_candidates(self) -> list[SubnetworkCandidate]:
        if self.config.candidate_pool:
            return [SubnetworkCandidate.from_mapping(item) for item in self.config.candidate_pool]
        return default_candidates()

    def _select_subnetworks(
        self,
        inputs: list[ClientRoundInput],
        adjacency: list[list[float]],
    ) -> dict[str, str]:
        max_samples = max(client.profile.sample_count for client in inputs)
        selected: dict[str, str] = {}
        for index, client in enumerate(inputs):
            client_scale = client.profile.sample_count / max(max_samples, 1)
            topology_bonus = max(
                (value for offset, value in enumerate(adjacency[index]) if offset != index),
                default=0.0,
            )
            candidate = choose_candidate(
                candidates=self.candidates,
                latency_budget_ms=client.latency_budget_ms,
                parameter_budget_m=client.parameter_budget_m,
                client_scale=client_scale,
                topology_bonus=topology_bonus,
            )
            selected[client.profile.client_id] = candidate.name
        return selected

    def run_round(self, inputs: list[ClientRoundInput]) -> RoundArtifacts:
        profiles = [item.profile for item in inputs]
        adjacency = build_dynamic_graph(
            profiles=profiles,
            weights=self.config.graph_weights,
            top_k=self.config.client_top_k,
        )
        consensus = metropolis_hastings_weights(adjacency)
        consensus = matrix_power(consensus, self.config.consensus_steps)
        selected = self._select_subnetworks(inputs, adjacency)
        aggregated_updates = apply_consensus(
            matrix=consensus,
            values=[item.local_update for item in inputs],
        )
        return RoundArtifacts(
            adjacency=adjacency,
            consensus_matrix=consensus,
            selected_subnetworks=selected,
            aggregated_updates=aggregated_updates,
        )
