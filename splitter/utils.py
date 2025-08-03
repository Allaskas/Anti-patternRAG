from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def read_limited_text(file_path: str, max_chars: int) -> str:
    """
    读取文件内容，最多读取 max_chars 个字符。
    用于防止单个 Java 文件内容过长。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        return content[:max_chars] if len(content) > max_chars else content


def parse_line_range(range_str: str) -> tuple[int, int]:
    """
    将形如 "12-18" 或 "112–300"（含不同破折号）字符串解析为整数元组 (start, end)
    """
    try:
        # 替换常见的各种破折号为普通减号
        normalized = range_str.replace("–", "-").replace("—", "-").replace("−", "-")
        start_str, end_str = normalized.strip().split("-")
        start, end = int(start_str), int(end_str)
        return start, end
    except Exception:
        raise ValueError(f"Invalid line range string: '{range_str}'. Expected format 'start-end'")


# QWen3 的 Embedding Prompt 所要用到 instruct 和 query 的格式，保留此格式所进行的拆分
def split_documents_with_instruction_context(documents: list[Document], tokenizer, max_token_length: int, chunk_overlap: int = 100) -> list[Document]:
    new_documents = []
    for doc in documents:
        content = doc.page_content
        try:
            instruct_part, query_part = content.split("Query:", maxsplit=1)
        except ValueError:
            # 格式不对，跳过或保留原文
            continue

        instruct_part = instruct_part.strip()
        query_part = query_part.strip()

        # 计算 instruct 部分的 token 长度
        instruct_tokens = tokenizer.encode(instruct_part)
        instruct_len = len(instruct_tokens)

        # 剩余可用于 query 的长度
        max_query_chunk_len = max_token_length - instruct_len

        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=max_query_chunk_len,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

        query_chunks = splitter.split_text(query_part)

        for i, query_chunk in enumerate(query_chunks):
            new_content = f"{instruct_part}\n\nQuery: {query_chunk}"
            new_metadata = doc.metadata.copy()
            new_metadata["split_index"] = i
            new_documents.append(Document(page_content=new_content, metadata=new_metadata))

    return new_documents


# jina 的 Code Embedding 过程中可能需要拆分 ast_subtree
def split_ast_documents(documents: list[Document], tokenizer, max_token_length: int, chunk_overlap: int = 100,) -> list[Document]:
    """
    拆分 AST 类型的文档，确保每个 chunk 不超过最大 token 数量，保留结构语义。
    """

    new_documents = []

    # 拆分器：用于 Java AST 拆分
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=max_token_length,
        chunk_overlap=chunk_overlap,
        separators=[") (", ")(", ") ", "(", ")", " "],  # 结构化拆分优先
    )

    for doc in documents:
        content = doc.page_content.strip()

        # 判断是否超长（可选，直接拆分也可）
        token_count = len(tokenizer.encode(content))
        if token_count <= max_token_length:
            new_documents.append(doc)
            continue

        chunks = splitter.split_text(content)

        for i, chunk in enumerate(chunks):
            new_metadata = doc.metadata.copy()
            new_metadata["split_index"] = i
            new_documents.append(Document(page_content=chunk.strip(), metadata=new_metadata))

    return new_documents
