from pathlib import Path
from collections import defaultdict
import json

from reranker.group_candidate_loader import load_group_chunks_by_type
from reranker.rerank_model import rerank_with_cross_encoder


def rerank_and_aggregate(
    query: str,
    score_files: list[Path],
    embedding_base_dir: Path,
    weight_file: Path,
    top_k: int = 5,
    rerank_model: str = "Qwen/Qwen3-Reranker-8B"
):
    """
    每个 chunk_type 独立 rerank → 按权重聚合得分 → 输出 top-k group_id

    Args:
        query: query 文本
        score_files: 每个 chunk_type 的匹配得分 json 文件（如 parent_method.json）
        embedding_base_dir: 各 chunk_type 对应 embedding 内容根路径
        weight_file: chunk_type 权重文件路径
        top_k: 最终输出的 group 数量
        rerank_model: cross-encoder 模型

    Returns:
        List of (group_id, aggregated_score)
    """

    # 1. 加载 chunk_type 权重
    with open(weight_file, "r", encoding="utf-8") as f:
        chunk_weights = json.load(f)

    rerank_scores_by_group = defaultdict(float)

    for score_file in score_files:
        chunk_type = score_file.stem  # e.g., parent_method
        weight = chunk_weights.get(chunk_type, 0.1)

        with open(score_file, "r", encoding="utf-8") as f:
            match_scores = json.load(f)

        # 2. 提取 top-N 组做 rerank（避免每个 chunk_type 太多）
        sorted_group_ids = sorted(match_scores.items(), key=lambda x: x[1], reverse=True)
        top_group_ids = [gid for gid, _ in sorted_group_ids[:top_k * 2]]

        # 3. 加载文本并进行 rerank
        candidates = []
        for group_id in top_group_ids:
            chunk = load_group_chunks_by_type(group_id, chunk_type, embedding_base_dir)
            if chunk:
                candidates.append((group_id, chunk))

        if not candidates:
            continue

        reranked = rerank_with_cross_encoder(query, candidates, model_name=rerank_model)

        # 4. 按 chunk_type 权重叠加得分
        for group_id, score in reranked:
            rerank_scores_by_group[group_id] += score * weight

    final_sorted = sorted(rerank_scores_by_group.items(), key=lambda x: x[1], reverse=True)
    return final_sorted[:top_k]
