from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pfl_bert_lite.config import GraphWeights
from pfl_bert_lite.graph.topology import ClientProfile, build_dynamic_graph, edge_score


class TopologyTests(unittest.TestCase):
    def test_more_similar_clients_receive_higher_edge_score(self) -> None:
        weights = GraphWeights()
        anchor = ClientProfile("a", 100, [0.8, 0.2], [6, 384, 1536, 6], [0.4, -0.2, 0.1])
        similar = ClientProfile("b", 120, [0.75, 0.25], [6, 384, 1536, 6], [0.39, -0.18, 0.12])
        different = ClientProfile("c", 60, [0.1, 0.9], [4, 256, 1024, 4], [-0.2, 0.3, -0.1])
        self.assertGreater(edge_score(anchor, similar, weights).total, edge_score(anchor, different, weights).total)

    def test_graph_is_symmetric(self) -> None:
        profiles = [
            ClientProfile("a", 100, [0.8, 0.2], [6, 384, 1536, 6], [0.4, -0.2, 0.1]),
            ClientProfile("b", 120, [0.75, 0.25], [6, 384, 1536, 6], [0.39, -0.18, 0.12]),
            ClientProfile("c", 60, [0.1, 0.9], [4, 256, 1024, 4], [-0.2, 0.3, -0.1]),
        ]
        adjacency = build_dynamic_graph(profiles, GraphWeights(), top_k=1)
        self.assertEqual(adjacency[0][1], adjacency[1][0])


if __name__ == "__main__":
    unittest.main()
