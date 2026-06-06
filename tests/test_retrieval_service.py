import unittest

from services.retrieval_service import retrieve_relevant_blocks


class RetrievalServiceTests(unittest.TestCase):
    def test_retrieve_relevant_blocks_ranks_best_match_first(self) -> None:
        items = [
            {"content": "产品最大功率为 120W，额定电压 220V。", "source": "A"},
            {"content": "设备外观为银色金属外壳。", "source": "B"},
            {"content": "包装箱内含说明书和电源线。", "source": "C"},
        ]

        results = retrieve_relevant_blocks("最大功率是多少", items, top_k=2)

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["source"], "A")

    def test_retrieve_relevant_blocks_skips_low_quality_text(self) -> None:
        items = [
            {
                "content": "/G1/G1/G1/G1/G1 /c1/c2/c3/c4/c5/c6/c7",
                "source": "noise",
                "type": "text",
            },
            {
                "content": "Word2vec 包含 CBOW 和 Skip-gram 两种训练模型。",
                "source": "clean",
                "type": "text",
            },
        ]

        results = retrieve_relevant_blocks("Word2vec 模型", items, top_k=2)

        self.assertTrue(results)
        self.assertEqual(results[0]["source"], "clean")


if __name__ == "__main__":
    unittest.main()
