from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


def rerank_with_cross_encoder(
    query: str,
    candidates: List[Tuple[str, str]],
    model_name: str = "Qwen/Qwen3-Reranker-8B",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> List[Tuple[str, float]]:
    """
    用 CrossEncoder 模型对 query + candidate_text 进行打分排序

    Args:
        query: 原始查询
        candidates: List of (group_id, merged_text)
        model_name: reranker 模型名称（默认 Qwen）
        device: cuda or cpu

    Returns:
        List of (group_id, score) sorted by score desc
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    inputs = tokenizer(
        [f"{query} [SEP] {text}" for _, text in candidates],
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits.squeeze(-1).tolist()

    if isinstance(scores, float):
        scores = [scores]

    return sorted(zip([gid for gid, _ in candidates], scores), key=lambda x: x[1], reverse=True)
