import json
from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def load_chunks_from_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_documents(chunks_json: dict, content_key: str) -> List[Document]:
    """
    构建 Document 列表：
    - page_content 来源于每个 chunk 的 content_key 字段（如 ast_subtree / llm_description）
    - metadata 合并 chunk 字段和顶层元信息（antipattern_type, project_name 等）

    :param chunks_json: load 后的整个 JSON 内容（包含顶层元数据和 chunks）
    :param content_key: 使用哪个字段作为向量内容，如 "ast_subtree" 或 "llm_description"
    """
    all_documents = []
    case_metadata = {k: v for k, v in chunks_json.items() if k != "chunks"}
    chunks = chunks_json["chunks"]

    for chunk in chunks:
        if content_key not in chunk:
            continue

        content = chunk[content_key]
        chunk_metadata = {k: v for k, v in chunk.items() if k != content_key}
        full_metadata = {**case_metadata, **chunk_metadata}

        all_documents.append(Document(page_content=content, metadata=full_metadata))

    return all_documents


def init_embedding_model(model_name: str, device: str = "cpu", normalize: bool = False):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device, 'trust_remote_code': True},
        encode_kwargs={"normalize_embeddings": normalize},
    )


def store_to_chroma(documents: List[Document], embedding_model, persist_dir):
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_dir,
    )
    vectorstore.persist()
    print(f"[✓] Stored {len(documents)} documents to Chroma: {persist_dir}")
    return vectorstore


def get_persist_dir_from_chunk_path(vector_store_dir: str, chunk_json_path: Path) -> Path:
    # 解析倒数四级路径部分
    parts = chunk_json_path.parts[-5:-1]
    persist_dir = Path(vector_store_dir).joinpath(*parts)
    return persist_dir
