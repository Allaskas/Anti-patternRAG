from pathlib import Path
from typing import Union

from config.settings import TEXT_EMBEDDING_MODEL, VECTOR_STORE_TEXT_DIR
from embedding_utils import (
    load_chunks_from_json,
    build_documents,
    init_embedding_model,
    store_to_chroma, get_persist_dir_from_chunk_path
)


def build_text_embedding(chunks_json_path: Union[str, Path]):
    chunks = load_chunks_from_json(Path(chunks_json_path))
    documents = build_documents(chunks, content_key="llm_description")
    embedding_model = init_embedding_model(TEXT_EMBEDDING_MODEL)
    persist_dir = get_persist_dir_from_chunk_path(VECTOR_STORE_TEXT_DIR, Path(chunks_json_path))
    store_to_chroma(documents, embedding_model, persist_dir=persist_dir)

