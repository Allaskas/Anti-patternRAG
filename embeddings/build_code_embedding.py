from pathlib import Path
from typing import Union

from transformers import AutoTokenizer

from config.settings import CODE_EMBEDDING_MODEL, VECTOR_STORE_CODE_DIR
from embeddings.embedding_utils import (
    load_chunks_from_json,
    build_documents,
    init_embedding_model,
    store_to_chroma,
    get_persist_dir_from_chunk_path, get_max_token_length, check_documents_exceed_max_len, get_query_vectorstore_dir,
)
from splitter.utils import split_ast_documents


def build_code_embedding(chunks_json_path: Union[str, Path]):
    chunks = load_chunks_from_json(Path(chunks_json_path))
    group_id = chunks["group_id"]
    documents = build_documents(chunks, content_key="ast_subtree")
    embedding_model = init_embedding_model(CODE_EMBEDDING_MODEL, default_task="code.passage")
    tokenizer = AutoTokenizer.from_pretrained(CODE_EMBEDDING_MODEL, trust_remote_code=True)
    model_max_len = get_max_token_length(tokenizer)
    if group_id < 0:
        persist_dir = get_query_vectorstore_dir(chunks_json_path, "CODE")
    else:
        persist_dir = get_persist_dir_from_chunk_path(VECTOR_STORE_CODE_DIR, Path(chunks_json_path))
    match CODE_EMBEDDING_MODEL:
        case m if "jinaai/jina-embeddings-v4" in m:
            valid_documents, exceeding_documents = check_documents_exceed_max_len(documents, tokenizer, model_max_len)
            if len(exceeding_documents) > 0:
                exceeding_documents = split_ast_documents(exceeding_documents, tokenizer,model_max_len)
            documents = valid_documents + exceeding_documents
        case _:
            pass
    store_to_chroma(documents, embedding_model, persist_dir=persist_dir)
    return persist_dir
