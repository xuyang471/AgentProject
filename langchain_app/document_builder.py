from __future__ import annotations

from typing import List

from services.text_quality_service import clean_text_artifacts, is_low_quality_text


def build_langchain_documents(parsed_documents: List[dict]) -> list:
    try:
        from langchain_core.documents import Document
    except ModuleNotFoundError:
        return []

    documents = []
    for document in parsed_documents:
        file_name = document.get("file_name", "")
        file_type = document.get("file_type", "")
        for index, block in enumerate(document.get("blocks", []), start=1):
            block_type = block.get("type", "text")
            content = clean_text_artifacts(block.get("content", "")).strip()
            if not content:
                continue
            if block_type == "text" and is_low_quality_text(content):
                continue

            metadata = {
                "file_name": file_name,
                "file_type": file_type,
                "block_type": block_type,
                "source": block.get("source", file_name),
                "page": block.get("page"),
                "image_index": block.get("image_index"),
                "table_index": block.get("table_index"),
                "table_bbox": block.get("table_bbox"),
                "table_image_path": block.get("table_image_path", ""),
                "image_path": block.get("image_path", ""),
                "ocr_text": block.get("ocr_text", ""),
                "description": block.get("description", ""),
                "layout_bbox": block.get("layout_bbox"),
                "text_block_index": block.get("text_block_index"),
                "is_caption": block.get("is_caption", False),
                "caption_text": block.get("caption_text", ""),
                "caption_bbox": block.get("caption_bbox"),
                "caption_source": block.get("caption_source", ""),
                "matched_caption_index": block.get("matched_caption_index"),
                "caption_match_method": block.get("caption_match_method", ""),
                "nearby_text": block.get("nearby_text", ""),
                "nearby_text_bbox": block.get("nearby_text_bbox"),
                "nearby_text_source": block.get("nearby_text_source", ""),
                "nearby_text_block_index": block.get("nearby_text_block_index"),
                "nearby_text_match_method": block.get("nearby_text_match_method", ""),
                "crop_bbox": block.get("crop_bbox"),
                "block_index": index,
            }
            documents.append(Document(page_content=content, metadata=metadata))

    return documents
