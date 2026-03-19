from __future__ import annotations

from typing import Any



def run_local_lora_finetune(
    model: Any,
    dataloader: Any,
    optimizer: Any,
    device: str,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for local fine-tuning.") from exc

    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    num_steps = 0

    for step, batch in enumerate(dataloader, start=1):
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / grad_accum_steps
        loss.backward()

        if step % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += float(loss.item())
        num_steps += 1

    return {"loss": total_loss / max(num_steps, 1)}
