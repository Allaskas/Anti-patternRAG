from typing import List, Union

class JinaCodeEmbeddingWrapper:
    """
    Wrapper for HuggingFaceEmbeddings / SentenceTransformer models,
    ensuring outputs are Python lists for Chroma, and supports 'code' task.
    """

    def __init__(self, hf_model):
        self.model = hf_model
        # 使用 _client 确保能传 task 参数
        if not hasattr(hf_model, "_client"):
            raise RuntimeError("HuggingFaceEmbeddings has no _client attribute")
        self._client = hf_model._client

    def embed_documents(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_emb = self._client.encode(batch_texts, task="code")
            # 转成 list[list[float]]
            if hasattr(batch_emb, "tolist"):
                batch_emb = batch_emb.tolist()
            embeddings.extend(batch_emb)
        return embeddings

    def embed_query(self, text_or_texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        single_input = isinstance(text_or_texts, str)
        texts = [text_or_texts] if single_input else text_or_texts
        emb = self._client.encode(texts, task="code")
        if hasattr(emb, "tolist"):
            emb = emb.tolist()
        return emb[0] if single_input else emb
