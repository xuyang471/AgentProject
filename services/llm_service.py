from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Iterable

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None


PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_TEXT_MODEL = "qwen-plus"
DEFAULT_VISION_MODEL = "qwen3-vl-flash"


def _get_client() -> OpenAI | None:
    if OpenAI is None:
        return None

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return None

    base_url = os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base_url)


def _text_model_name() -> str:
    return os.getenv("QWEN_TEXT_MODEL", DEFAULT_TEXT_MODEL)


def _vision_model_name() -> str:
    return os.getenv("QWEN_VISION_MODEL", DEFAULT_VISION_MODEL)


def load_prompt(name: str) -> str:
    prompt_path = PROMPTS_DIR / name
    return prompt_path.read_text(encoding="utf-8")


def extract_report_insights(texts: Iterable[str]) -> dict | None:
    merged = "\n".join(text.strip() for text in texts if text.strip()).strip()
    if not merged:
        return None

    client = _get_client()
    if client is None:
        return None

    prompt = load_prompt("report_insight_prompt.txt")
    try:
        response = client.chat.completions.create(
            model=_text_model_name(),
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"请基于以下文档内容抽取结构化信息：\n\n{merged[:12000]}",
                },
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        content = (response.choices[0].message.content or "").strip()
        return json.loads(content) if content else None
    except Exception:
        return None


def summarize_texts(texts: Iterable[str]) -> str:
    merged = " ".join(texts).strip()
    if not merged:
        return "未提取到有效文本内容。"

    client = _get_client()
    if client is None:
        return merged[:150]

    prompt = load_prompt("summary_prompt.txt")
    try:
        response = client.chat.completions.create(
            model=_text_model_name(),
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"请将以下文档内容总结为 150 字以内的中文摘要：\n\n{merged[:8000]}",
                },
            ],
            temperature=0.2,
        )
        return (response.choices[0].message.content or "").strip() or merged[:150]
    except Exception:
        return merged[:150]


def answer_from_context(question: str, context: str) -> str:
    if not context.strip():
        return "未找到可用于回答的问题上下文。"

    client = _get_client()
    if client is None:
        return f"问题：{question}\n\n基于当前已提取内容，最相关信息如下：\n{context}"

    prompt = load_prompt("qa_prompt.txt")
    try:
        response = client.chat.completions.create(
            model=_text_model_name(),
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"上下文如下：\n{context}\n\n问题：{question}\n\n请用简洁中文回答，并优先引用上下文中的事实。",
                },
            ],
            temperature=0.2,
        )
        return (response.choices[0].message.content or "").strip() or "模型未返回有效答案。"
    except Exception:
        return f"问题：{question}\n\n基于当前已提取内容，最相关信息如下：\n{context}"


def describe_image(image_path: str | Path) -> str:
    analysis = analyze_image(image_path)
    return analysis["description"]


def analyze_image(image_path: str | Path, local_ocr_text: str = "") -> dict:
    client = _get_client()
    if client is None:
        return {
            "ocr_text": local_ocr_text.strip(),
            "description": "当前版本未配置 DASHSCOPE_API_KEY，图片仅保留基础元信息，未进行云端视觉理解。",
        }

    image_path = Path(image_path)
    mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    data_url = f"data:{mime_type};base64,{encoded}"

    prompt_text = (
        "请分析这张图片，并返回严格 JSON，字段必须为 ocr_text 和 description。"
        "ocr_text: 提取图片中可见的关键文字；如果几乎没有可辨识文字则返回空字符串。"
        "description: 用中文简洁描述图片内容、图表含义或场景。"
        f"已知本地 OCR 结果：{local_ocr_text or '无'}。"
    )

    try:
        response = client.chat.completions.create(
            model=_vision_model_name(),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = (response.choices[0].message.content or "").strip()
        payload = json.loads(content) if content else {}
        cloud_ocr_text = str(payload.get("ocr_text", "")).strip()
        description = str(payload.get("description", "")).strip()
        merged_ocr_text = local_ocr_text.strip() or cloud_ocr_text
        return {
            "ocr_text": merged_ocr_text,
            "description": description or "云端视觉模型未返回有效描述。",
        }
    except Exception:
        return {
            "ocr_text": local_ocr_text.strip(),
            "description": "云端视觉理解调用失败，已保留基础图片信息。",
        }
