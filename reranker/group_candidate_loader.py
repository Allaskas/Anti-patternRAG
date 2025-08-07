import json
from pathlib import Path
from typing import Optional


def load_group_chunks_by_type(group_id: str, chunk_type: str, embedding_base_dir: Path) -> Optional[str]:
    """
    给定 group_id 和 chunk_type，加载该 chunk_type 下的 chunk 文本

    Returns:
        对应的 chunk 文本 or None
    """
    chunk_dir = embedding_base_dir / chunk_type
    content_file = chunk_dir / "content.json"

    if not content_file.exists():
        return None

    try:
        with open(content_file, "r", encoding="utf-8") as f:
            docs = json.load(f)
    except Exception as e:
        print(f"[ERROR] Load failed: {e}")
        return None

    for doc in docs:
        metadata = doc.get("metadata", {})
        if metadata.get("group_id") == group_id:
            return doc.get("page_content", "")

    return None
