import faiss
import pickle
import numpy as np

# ---------- 1️⃣ 载入 FAISS 索引 ----------
index_path = "faiss_index.idx"
index = faiss.read_index(index_path)

print(f"[i] FAISS index loaded: {index.ntotal} vectors, dimension {index.d}")

# ---------- 2️⃣ 载入 metadata ----------
metadata_path = "metadata.pkl"
with open(metadata_path, "rb") as f:
    metadatas = pickle.load(f)

print(f"[i] Metadata loaded: {len(metadatas)} documents")

# ---------- 3️⃣ 打印前几个文档的原文和向量 ----------
num_to_show = len(metadatas)  # 可以修改想看多少条
for i in range(num_to_show):
    data = metadatas[i]
    print(f"Document {i}:")

    # 原文
    print("Page content:", data.get("page_content", "[No page_content stored]"))

    # 其它 metadata
    other_meta = {k: v for k, v in data.items() if k != "embedding" and k != "page_content"}
    if other_meta:
        print("Other metadata:", other_meta)

    # 对应向量
    vec = index.reconstruct(i)
    print("Embedding vector (first 10 dims):", vec[:10], "...")  # 只打印前10维

    print("-" * 60)
