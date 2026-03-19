from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pfl_bert_lite.data.partition import clean_text, dirichlet_partition


class PartitionTests(unittest.TestCase):
    def test_clean_text_removes_punctuation_and_collapses_spaces(self) -> None:
        self.assertEqual(clean_text("Great!!!  Federated,   BERT?"), "great federated bert")

    def test_dirichlet_partition_is_a_cover(self) -> None:
        labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        partitions = dirichlet_partition(labels, num_clients=3, alpha=0.5, seed=7)
        covered = sorted(index for bucket in partitions for index in bucket)
        self.assertEqual(covered, list(range(len(labels))))


if __name__ == "__main__":
    unittest.main()
