from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"


def create_qwen_embeddings():
    try:
        from langchain_openai import OpenAIEmbeddings
    except ModuleNotFoundError:
        return None

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return None

    return OpenAIEmbeddings(
        model=os.getenv("QWEN_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        api_key=api_key,
        base_url=os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL),
    )
