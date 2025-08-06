from pathlib import Path
from typing import Dict

from chromadb import Embeddings
from langchain_community.vectorstores import Chroma

from config.settings import CODE_EMBEDDING_MODEL, TEXT_EMBEDDING_MODEL, VECTOR_STORE_CODE_DIR
from embeddings.embedding_utils import init_embedding_model


class MultiChunkRetriever:
    def __init__(
        self,
        vectorstore_root: str,
        code_embedding_model: Embeddings,
        text_embedding_model: Embeddings,
    ):
        self.vectorstore_root = Path(vectorstore_root)
        self.code_embedding_model = code_embedding_model
        self.text_embedding_model = text_embedding_model

        self.retrievers: Dict[str, Dict[str, Chroma]] = {"CODE": {}, "TEXT": {}}
        self._load_all_vectorstores()

    def _load_all_vectorstores(self):
        for category in ["CODE", "TEXT"]:
            category_dir = self.vectorstore_root / category
            if not category_dir.exists():
                continue
            for chunk_dir in category_dir.iterdir():
                if not chunk_dir.is_dir():
                    continue

                model = self.code_embedding_model if category == "CODE" else self.text_embedding_model
                retriever = Chroma(
                    persist_directory=str(chunk_dir),
                    embedding_function=model,
                ).as_retriever()

                self.retrievers[category][chunk_dir.name] = retriever
                print(f"[LOAD] Loaded retriever: {category}/{chunk_dir.name}")

    def query_by_chunk_type(self, chunk_type: str, query: str, category: str = "CODE", k: int = 5) -> List[Document]:
        retriever = self.retrievers.get(category, {}).get(chunk_type)
        if not retriever:
            print(f"[WARN] Retriever not found for {category}/{chunk_type}")
            return []
        return retriever.get_relevant_documents(query, k=k)

    def query_all(self, query: str, k: int = 5):
        all_results = {}
        for category in self.retrievers:
            for chunk_type, retriever in self.retrievers[category].items():
                docs = retriever.get_relevant_documents(query, k=k)
                all_results[f"{category}/{chunk_type}"] = docs
        return all_results


# 初始化 embedding 模型
code_model = init_embedding_model(CODE_EMBEDDING_MODEL, default_task="code.passage")
text_model = init_embedding_model(TEXT_EMBEDDING_MODEL)

retriever = MultiChunkRetriever(
    vectorstore_root=str(Path(VECTOR_STORE_CODE_DIR).parent),
    code_embedding_model=code_model,
    text_embedding_model=text_model,
)

# 单个 chunk_type 查询
docs = retriever.query_by_chunk_type("parent_method", "how does the child call parent?", category="CODE")

# 所有 chunk_type 查询
all_results = retriever.query_all("find related file structure")
for name, docs in all_results.items():
    print(f"== Results from {name} ==")
    for doc in docs:
        print(doc.metadata, doc.page_content[:100])
