from pathlib import Path
from typing import Union

from embedding_utils import (
    load_chunks_from_json,
    build_documents,
    init_embedding_model,
    store_to_chroma,
    get_persist_dir_from_chunk_path,
)
from config.settings import CODE_EMBEDDING_MODEL, TEXT_EMBEDDING_MODEL
from embeddings.build_code_embedding import build_code_embedding
from embeddings.build_text_embedding import build_text_embedding


def run_embedding_pipeline(chunks_json_path: Union[str, Path]):
    chunks_json_path = Path(chunks_json_path)

    if not chunks_json_path.exists():
        raise FileNotFoundError(f"Chunk JSON file does not exist: {chunks_json_path}")

    build_code_embedding(chunks_json_path)
    build_text_embedding(chunks_json_path)


if __name__ == "__main__":
    # ✅ 示例调用
    run_embedding_pipeline("/data/sanglei/Anti-patternRAG/data/CH/kafka/commit_1000/6/kafka_6_CH_chunk.json")
