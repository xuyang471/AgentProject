import unittest

from parsers.pdf_parser import (
    _attach_caption_metadata,
    _attach_nearby_text_metadata,
    _bbox_overlap_ratio,
    _caption_crop_bounds,
    _contains_figure_caption,
    _find_nearby_text_block,
    _filter_layout_blocks_against_tables,
    _is_near_duplicate_text,
    _is_noise_image,
    _looks_like_scanned_page,
    _merge_page_text,
    _normalize_bbox,
    _normalize_layout_text_blocks,
)


class PdfParserHelpersTests(unittest.TestCase):
    def test_contains_figure_caption_detects_chinese_caption(self) -> None:
        self.assertTrue(_contains_figure_caption("\u56fe1 \u795e\u7ecf\u8bed\u8a00\u6a21\u578b"))
        self.assertTrue(_contains_figure_caption("图1 神经语言模型"))

    def test_contains_figure_caption_detects_english_caption(self) -> None:
        self.assertTrue(_contains_figure_caption("Figure 2 Neural Language Model"))

    def test_noise_image_filter_marks_small_blank_image(self) -> None:
        block = {
            "image_width": 62,
            "image_height": 63,
            "ocr_text": "",
            "description": "图片文件，尺寸约为 62x63，格式为 PNG。 一张纯白色背景的图片，无明显内容。",
        }
        self.assertTrue(_is_noise_image(block))

    def test_caption_crop_bounds_avoids_full_page_crop(self) -> None:
        x0, y0, x1, y1 = _caption_crop_bounds(600, 800, (180, 420, 420, 450))

        self.assertEqual(x0, 24)
        self.assertEqual(x1, 576)
        self.assertGreater(y0, 0)
        self.assertLess(y1 - y0, 800)

    def test_merge_page_text_prefers_fallback_when_primary_is_low_quality(self) -> None:
        merged = _merge_page_text("/G1/G1/G1/G1/G1 /c1/c2/c3/c4/c5", "Word2vec 的工作原理及应用探究")

        self.assertEqual(merged, "Word2vec 的工作原理及应用探究")

    def test_merge_page_text_keeps_longer_version_for_near_duplicates(self) -> None:
        merged = _merge_page_text(
            "图1 神经语言模型",
            "图1 神经语言模型 该模型用于根据上下文预测目标词",
        )

        self.assertEqual(merged, "图1 神经语言模型 该模型用于根据上下文预测目标词")

    def test_merge_page_text_combines_complementary_text(self) -> None:
        merged = _merge_page_text("2.3 神经网络语言模型", "Word2vec 包含 CBOW 与 Skip-gram 两种模型")

        self.assertIn("2.3 神经网络语言模型", merged)
        self.assertIn("Word2vec 包含 CBOW 与 Skip-gram 两种模型", merged)

    def test_near_duplicate_text_detects_whitespace_and_punctuation_variants(self) -> None:
        self.assertTrue(_is_near_duplicate_text("Figure 1: Neural model", "Figure 1 Neural model"))

    def test_normalize_layout_text_blocks_filters_noise_and_duplicates(self) -> None:
        normalized = _normalize_layout_text_blocks(
            [
                {"content": "/G1/G1/G1/G1 /c1/c2/c3/c4", "bbox": _normalize_bbox((0, 0, 10, 10)), "is_caption": False},
                {"content": "图1 神经语言模型", "bbox": _normalize_bbox((10, 20, 30, 40)), "is_caption": True},
                {"content": "图1 神经语言模型", "bbox": _normalize_bbox((10, 20, 30, 40)), "is_caption": True},
                {
                    "content": "Word2vec 包含 CBOW 和 Skip-gram 两种核心模型",
                    "bbox": _normalize_bbox((20, 30, 200, 80)),
                    "is_caption": False,
                },
                {"content": "a", "bbox": _normalize_bbox((0, 0, 5, 5)), "is_caption": False},
            ]
        )

        self.assertEqual(
            normalized,
            [
                {
                    "content": "图1 神经语言模型",
                    "bbox": {"x0": 10.0, "y0": 20.0, "x1": 30.0, "y1": 40.0},
                    "is_caption": True,
                },
                {
                    "content": "Word2vec 包含 CBOW 和 Skip-gram 两种核心模型",
                    "bbox": {"x0": 20.0, "y0": 30.0, "x1": 200.0, "y1": 80.0},
                    "is_caption": False,
                },
            ],
        )

    def test_attach_caption_metadata_adds_caption_fields_to_image_block(self) -> None:
        image_block = {"type": "image", "source": "demo.pdf 第 2 页 图片 1"}
        caption_block = {
            "content": "图1 神经语言模型",
            "bbox": {"x0": 15.0, "y0": 300.0, "x1": 120.0, "y1": 320.0},
            "is_caption": True,
        }

        enriched = _attach_caption_metadata(image_block, caption_block, caption_index=1, match_method="page_order")

        self.assertEqual(enriched["caption_text"], "图1 神经语言模型")
        self.assertEqual(enriched["caption_bbox"]["y0"], 300.0)
        self.assertEqual(enriched["matched_caption_index"], 1)
        self.assertEqual(enriched["caption_match_method"], "page_order")

    def test_find_nearby_text_block_prefers_below_caption_with_overlap(self) -> None:
        caption_block = {
            "content": "Figure 1 Neural Language Model",
            "bbox": {"x0": 100.0, "y0": 300.0, "x1": 260.0, "y1": 320.0},
            "is_caption": True,
            "text_block_index": 2,
            "source": "demo.pdf 第 2 页 文本块 2",
        }
        layout_blocks = [
            caption_block,
            {
                "content": "This paragraph explains how the neural language model predicts the next token.",
                "bbox": {"x0": 96.0, "y0": 332.0, "x1": 360.0, "y1": 372.0},
                "is_caption": False,
                "text_block_index": 3,
                "source": "demo.pdf 第 2 页 文本块 3",
            },
            {
                "content": "A distant unrelated paragraph on another topic.",
                "bbox": {"x0": 40.0, "y0": 430.0, "x1": 360.0, "y1": 470.0},
                "is_caption": False,
                "text_block_index": 4,
                "source": "demo.pdf 第 2 页 文本块 4",
            },
            {
                "content": "A short line above the figure.",
                "bbox": {"x0": 110.0, "y0": 250.0, "x1": 260.0, "y1": 272.0},
                "is_caption": False,
                "text_block_index": 1,
                "source": "demo.pdf 第 2 页 文本块 1",
            },
        ]

        nearby = _find_nearby_text_block(caption_block, layout_blocks)

        self.assertIsNotNone(nearby)
        self.assertEqual(nearby["text_block_index"], 3)

    def test_attach_nearby_text_metadata_adds_neighbor_fields(self) -> None:
        image_block = {"type": "image", "source": "demo.pdf 第 2 页 图片 1"}
        nearby_text_block = {
            "content": "This paragraph explains the figure.",
            "bbox": {"x0": 96.0, "y0": 332.0, "x1": 360.0, "y1": 372.0},
            "text_block_index": 3,
            "source": "demo.pdf 第 2 页 文本块 3",
        }

        enriched = _attach_nearby_text_metadata(image_block, nearby_text_block, match_method="caption_neighbor")

        self.assertEqual(enriched["nearby_text"], "This paragraph explains the figure.")
        self.assertEqual(enriched["nearby_text_bbox"]["y0"], 332.0)
        self.assertEqual(enriched["nearby_text_block_index"], 3)
        self.assertEqual(enriched["nearby_text_match_method"], "caption_neighbor")

    def test_looks_like_scanned_page_detects_low_quality_page(self) -> None:
        detected = _looks_like_scanned_page(
            merged_text="/G1/G1/G1 /c1/c2/c3",
            layout_blocks=[],
            page_images=[object()],
            raw_text="/G1/G1",
            fallback_text="",
        )

        self.assertTrue(detected)

    def test_looks_like_scanned_page_skips_normal_text_page(self) -> None:
        detected = _looks_like_scanned_page(
            merged_text="This page contains enough readable text to avoid whole-page OCR fallback. " * 3,
            layout_blocks=[
                {"content": "Paragraph one", "bbox": {"x0": 0, "y0": 0, "x1": 100, "y1": 40}, "is_caption": False},
                {"content": "Paragraph two", "bbox": {"x0": 0, "y0": 50, "x1": 100, "y1": 90}, "is_caption": False},
            ],
            page_images=[],
            raw_text="Readable text",
            fallback_text="Readable text",
        )

        self.assertFalse(detected)

    def test_bbox_overlap_ratio_returns_expected_fraction(self) -> None:
        source_bbox = {"x0": 0.0, "y0": 0.0, "x1": 100.0, "y1": 100.0}
        target_bbox = {"x0": 50.0, "y0": 0.0, "x1": 100.0, "y1": 100.0}

        self.assertEqual(_bbox_overlap_ratio(source_bbox, target_bbox), 0.5)

    def test_filter_layout_blocks_against_tables_removes_table_overlaps(self) -> None:
        layout_blocks = [
            {
                "content": "参数 值 速度 120",
                "bbox": {"x0": 10.0, "y0": 100.0, "x1": 210.0, "y1": 180.0},
                "is_caption": False,
            },
            {
                "content": "本文提出一种改进方法。",
                "bbox": {"x0": 10.0, "y0": 220.0, "x1": 210.0, "y1": 260.0},
                "is_caption": False,
            },
            {
                "content": "表1 参数配置",
                "bbox": {"x0": 10.0, "y0": 80.0, "x1": 120.0, "y1": 95.0},
                "is_caption": True,
            },
        ]
        page_tables = [
            {
                "content": "| 参数 | 值 |\n| --- | --- |\n| 速度 | 120 |",
                "bbox": {"x0": 0.0, "y0": 90.0, "x1": 220.0, "y1": 190.0},
            }
        ]

        filtered = _filter_layout_blocks_against_tables(layout_blocks, page_tables)

        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["content"], "本文提出一种改进方法。")
        self.assertEqual(filtered[1]["content"], "表1 参数配置")


if __name__ == "__main__":
    unittest.main()
