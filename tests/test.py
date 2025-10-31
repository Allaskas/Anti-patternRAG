# from langchain_ollama import OllamaLLM
#
# llm = OllamaLLM(model="gemma3:1b")
# print(llm.invoke("你好"))


# from langchain_huggingface import HuggingFaceEmbeddings
#
# model_name1 = "Qwen/Qwen3-Embedding-8B"
# model_name2 = "jinaai/jina-embeddings-v4"
# model_kwargs = {'device': 'cpu', 'trust_remote_code': True}
# encode_kwargs = {'normalize_embeddings': False}
# print("run model1")
# hf1 = HuggingFaceEmbeddings(
#     model_name=model_name1,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs
# )
# print("run model2")
# hf2 = HuggingFaceEmbeddings(
#     model_name=model_name2,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs
# )

# sanglei@amax:~/Anti-patternRAG$ rm -f /data/sanglei/.cache/huggingface/hub/.locks/models--Qwen--Qwen3-Embedding-8B
# rm: cannot remove '/data/sanglei/.cache/huggingface/hub/.locks/models--Qwen--Qwen3-Embedding-8B': Is a directory
# sanglei@amax:~/Anti-patternRAG$
# sanglei@amax:~/Anti-patternRAG$ rm -rf /data/sanglei/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-8B
# sanglei@amax:~/Anti-patternRAG$ rm -rf /data/sanglei/.cache/huggingface/hub/.locks/models--Qwen--Qwen3-Embedding-8B
# sanglei@amax:~/Anti-patternRAG$ find "$HOME/.cache/huggingface/hub" -type d -name "*Qwen3-Embedding-8B*"
# sanglei@amax:~/Anti-patternRAG$ find "$HOME/.cache/huggingface/hub" -type d -name "*jina-embeddings-v4*"
# /data/sanglei/.cache/huggingface/hub/.locks/models--jinaai--jina-embeddings-v4
# /data/sanglei/.cache/huggingface/hub/models--jinaai--jina-embeddings-v4
# sanglei@amax:~/Anti-patternRAG$ rm -rf /data/sanglei/.cache/huggingface/hub/.locks/models--jinaai--jina-embeddings-v4
# sanglei@amax:~/Anti-patternRAG$ rm -rf /data/sanglei/.cache/huggingface/hub/models--jinaai--jina-embeddings-v4
# sanglei@amax:~/Anti-patternRAG$ cat test/py
# cat: test/py: No such file or directory



from langchain_community.vectorstores import Chroma

# 指定 persist_directory
persist_dir = "/Users/moncheri/Downloads/main/重构/反模式修复数据集构建/RefactorRAG/Anti-PatternRAG"

# 载入已有的 Chroma 向量库
vectorstore = Chroma(persist_directory=persist_dir)

# 查看所有文档数量
print("Number of documents:", len(vectorstore._collection.get(include=["metadatas"])["metadatas"]))

# 获取向量、文本、metadata
data = vectorstore._collection.get(include=["embeddings", "documents", "metadatas"])
