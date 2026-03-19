from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Iterable



def clean_text(text: str) -> str:
    lowered = text.lower()
    normalized = "".join(char if char.isalnum() or char.isspace() else " " for char in lowered)
    return " ".join(normalized.split())



def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows



def label_histogram(labels: Iterable[int], num_classes: int | None = None) -> list[float]:
    labels_list = list(labels)
    if not labels_list and not num_classes:
        return []
    width = (max(labels_list) + 1) if labels_list else int(num_classes or 0)
    width = max(width, int(num_classes or 0))
    counts = [0] * width
    for label in labels_list:
        counts[int(label)] += 1
    total = sum(counts) or 1
    return [count / total for count in counts]



def _sample_dirichlet(alpha: float, size: int, rng: random.Random) -> list[float]:
    raw = [rng.gammavariate(alpha, 1.0) for _ in range(size)]
    total = sum(raw) or 1.0
    return [value / total for value in raw]



def dirichlet_partition(
    labels: Iterable[int],
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> list[list[int]]:
    label_to_indices: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        label_to_indices[int(label)].append(index)

    rng = random.Random(seed)
    client_partitions: list[list[int]] = [[] for _ in range(num_clients)]

    for indices in label_to_indices.values():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        shares = _sample_dirichlet(alpha=alpha, size=num_clients, rng=rng)
        cut_points: list[int] = []
        running_share = 0.0
        for share in shares[:-1]:
            running_share += share
            cut_points.append(int(round(running_share * len(shuffled))))

        start = 0
        for client_id, end in enumerate(cut_points + [len(shuffled)]):
            client_partitions[client_id].extend(shuffled[start:end])
            start = end

    for partition in client_partitions:
        partition.sort()
    return client_partitions
