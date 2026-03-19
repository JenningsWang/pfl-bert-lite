from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pfl_bert_lite.config import ExperimentConfig
from pfl_bert_lite.graph.topology import ClientProfile
from pfl_bert_lite.training.federated_trainer import (
    ClientRoundInput,
    PersonalizedFederatedTrainer,
)
from pfl_bert_lite.training.repro import set_global_seed



def build_demo_round_inputs(config: ExperimentConfig) -> list[ClientRoundInput]:
    return [
        ClientRoundInput(
            profile=ClientProfile(
                client_id="client_0",
                sample_count=240,
                label_histogram=[0.82, 0.18],
                structure_signature=[6.0, 384.0, 1536.0, 6.0],
                gradient_vector=[0.41, -0.09, 0.18, 0.22],
            ),
            local_update={
                "classifier.weight": [0.18, -0.03, 0.07],
                "classifier.bias": [0.01, -0.01],
            },
            latency_budget_ms=config.latency_budget_ms,
            parameter_budget_m=config.parameter_budget_m,
        ),
        ClientRoundInput(
            profile=ClientProfile(
                client_id="client_1",
                sample_count=170,
                label_histogram=[0.55, 0.45],
                structure_signature=[6.0, 384.0, 1536.0, 6.0],
                gradient_vector=[0.38, -0.04, 0.20, 0.19],
            ),
            local_update={
                "classifier.weight": [0.11, -0.02, 0.05],
                "classifier.bias": [0.00, 0.01],
            },
            latency_budget_ms=config.latency_budget_ms - 4.0,
            parameter_budget_m=config.parameter_budget_m,
        ),
        ClientRoundInput(
            profile=ClientProfile(
                client_id="client_2",
                sample_count=96,
                label_histogram=[0.21, 0.79],
                structure_signature=[4.0, 256.0, 1024.0, 4.0],
                gradient_vector=[-0.12, 0.17, -0.25, 0.08],
            ),
            local_update={
                "classifier.weight": [-0.09, 0.04, -0.06],
                "classifier.bias": [-0.01, 0.02],
            },
            latency_budget_ms=config.latency_budget_ms - 15.0,
            parameter_budget_m=config.parameter_budget_m - 4.0,
        ),
    ]



def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight personalized FL dry-run.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    parser.add_argument(
        "--output",
        default=str(ROOT / "artifacts" / "demo_round.json"),
        help="Where to write the JSON artifact.",
    )
    args = parser.parse_args()

    config = ExperimentConfig.from_json(args.config)
    set_global_seed(config.seed)
    trainer = PersonalizedFederatedTrainer(config)
    inputs = build_demo_round_inputs(config)
    artifacts = trainer.run_round(inputs)

    payload = {
        "config": config.to_dict(),
        "round_artifacts": artifacts.to_dict(),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload["round_artifacts"]["selected_subnetworks"], indent=2))
    print(f"Wrote demo artifact to {output_path}")


if __name__ == "__main__":
    main()
