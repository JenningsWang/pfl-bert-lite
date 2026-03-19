from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pfl_bert_lite.aggregation.consensus import (
    apply_consensus,
    metropolis_hastings_weights,
    row_sums,
)


class ConsensusTests(unittest.TestCase):
    def test_complete_graph_reaches_average_in_one_step(self) -> None:
        adjacency = [
            [1.0, 0.8, 0.8],
            [0.8, 1.0, 0.8],
            [0.8, 0.8, 1.0],
        ]
        weights = metropolis_hastings_weights(adjacency)
        mixed = apply_consensus(weights, [1.0, 4.0, 7.0])
        for value in mixed:
            self.assertAlmostEqual(value, 4.0, places=7)

    def test_rows_sum_to_one(self) -> None:
        adjacency = [
            [1.0, 0.6, 0.0],
            [0.6, 1.0, 0.5],
            [0.0, 0.5, 1.0],
        ]
        weights = metropolis_hastings_weights(adjacency)
        for row_sum in row_sums(weights):
            self.assertAlmostEqual(row_sum, 1.0, places=7)


if __name__ == "__main__":
    unittest.main()
