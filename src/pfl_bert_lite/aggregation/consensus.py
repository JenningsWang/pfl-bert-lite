from __future__ import annotations

from typing import Any, Sequence



def metropolis_hastings_weights(adjacency: Sequence[Sequence[float]]) -> list[list[float]]:
    size = len(adjacency)
    degrees = [
        sum(1 for j, value in enumerate(row) if i != j and value > 0.0)
        for i, row in enumerate(adjacency)
    ]
    weights = [[0.0 for _ in range(size)] for _ in range(size)]

    for i in range(size):
        off_diagonal_sum = 0.0
        for j in range(size):
            if i == j or adjacency[i][j] <= 0.0:
                continue
            weight = 1.0 / (1.0 + max(degrees[i], degrees[j], 1))
            weights[i][j] = weight
            off_diagonal_sum += weight
        weights[i][i] = 1.0 - off_diagonal_sum
    return weights



def matrix_multiply(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> list[list[float]]:
    rows = len(left)
    inner = len(right)
    cols = len(right[0]) if right else 0
    product = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for k in range(inner):
            if left[i][k] == 0.0:
                continue
            for j in range(cols):
                product[i][j] += left[i][k] * right[k][j]
    return product



def matrix_power(matrix: Sequence[Sequence[float]], power: int) -> list[list[float]]:
    if power <= 1:
        return [list(row) for row in matrix]
    result = [list(row) for row in matrix]
    for _ in range(power - 1):
        result = matrix_multiply(result, matrix)
    return result



def _scale_value(value: Any, factor: float) -> Any:
    if isinstance(value, (int, float)):
        return value * factor
    if isinstance(value, list):
        return [_scale_value(item, factor) for item in value]
    if isinstance(value, tuple):
        return tuple(_scale_value(item, factor) for item in value)
    if isinstance(value, dict):
        return {key: _scale_value(item, factor) for key, item in value.items()}
    return value * factor



def _add_values(left: Any, right: Any) -> Any:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left + right
    if isinstance(left, list) and isinstance(right, list):
        return [_add_values(a, b) for a, b in zip(left, right)]
    if isinstance(left, tuple) and isinstance(right, tuple):
        return tuple(_add_values(a, b) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        keys = set(left) | set(right)
        return {
            key: _add_values(left.get(key, 0.0), right.get(key, 0.0))
            for key in keys
        }
    return left + right



def apply_consensus(matrix: Sequence[Sequence[float]], values: Sequence[Any]) -> list[Any]:
    aggregated: list[Any] = []
    for row in matrix:
        mixed = None
        for weight, value in zip(row, values):
            if weight == 0.0:
                continue
            weighted_value = _scale_value(value, weight)
            mixed = weighted_value if mixed is None else _add_values(mixed, weighted_value)
        aggregated.append(mixed)
    return aggregated



def row_sums(matrix: Sequence[Sequence[float]]) -> list[float]:
    return [sum(row) for row in matrix]
