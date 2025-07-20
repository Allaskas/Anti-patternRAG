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
