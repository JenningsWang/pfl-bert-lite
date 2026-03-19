from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("query", "key", "value", "dense")


@dataclass(frozen=True)
class BERTSubnetwork:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int



def estimate_subnetwork_cost(subnetwork: BERTSubnetwork, sequence_length: int = 128) -> dict[str, float]:
    params_m = (
        subnetwork.num_hidden_layers
        * (4 * subnetwork.hidden_size * subnetwork.hidden_size + 2 * subnetwork.hidden_size * subnetwork.intermediate_size)
    ) / 1000000
    flops_m = params_m * sequence_length * 2.0
    return {"params_m": round(params_m, 3), "flops_m": round(flops_m, 3)}



def build_personalized_model(
    base_model_name: str,
    num_labels: int,
    lora_config: LoRAConfig,
    subnetwork: BERTSubnetwork,
) -> Any:
    try:
        from transformers import AutoConfig, AutoModelForSequenceClassification
    except ImportError as exc:
        raise ImportError(
            "transformers is required. Install dependencies from requirements.txt."
        ) from exc

    try:
        from peft import LoraConfig as PeftLoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError("peft is required. Install dependencies from requirements.txt.") from exc

    config = AutoConfig.from_pretrained(base_model_name)
    config.num_labels = num_labels
    config.hidden_size = subnetwork.hidden_size
    config.num_hidden_layers = subnetwork.num_hidden_layers
    config.num_attention_heads = subnetwork.num_attention_heads
    config.intermediate_size = subnetwork.intermediate_size

    model = AutoModelForSequenceClassification.from_config(config)
    peft_config = PeftLoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_config.rank,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=list(lora_config.target_modules),
    )
    return get_peft_model(model, peft_config)
