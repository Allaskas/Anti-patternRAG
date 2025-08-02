from pathlib import Path
from typing import Union
from embeddings.build_code_embedding import build_code_embedding
from embeddings.build_text_embedding import build_text_embedding


def run_embedding_pipeline(chunks_json_path: Union[str, Path]):
    chunks_json_path = Path(chunks_json_path)

    if not chunks_json_path.exists():
        raise FileNotFoundError(f"Chunk JSON file does not exist: {chunks_json_path}")

    print(" start run     build_code_embedding(chunks_json_path) ")
    build_code_embedding(chunks_json_path)
    print("✅ run over    build_code_embedding(chunks_json_path) ")
    print(" start run     build_text_embedding(chunks_json_path) ")
    build_text_embedding(chunks_json_path)
    print("✅ run over    build_text_embedding(chunks_json_path) ")


if __name__ == "__main__":
    # ✅ 示例调用
    run_embedding_pipeline("/data/sanglei/Anti-patternRAG/data/CH/kafka/commit_1000/6/kafka_6_CH_chunk.json")
