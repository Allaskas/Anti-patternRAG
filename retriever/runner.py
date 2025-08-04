from langchain.schema import BaseRetriever


def run_query_matching_pipeling(merge_vectorstore_dir: str, query_data_dir: str, top_k: int):
    """
    1. 从 query_project_dir 中提取文本/代码块
    2 对其进行 chunk → embedding → 存储为临时 query_vectorstore
    3 遍历其中的每个 chunk_type，加载对应的向量
    4 与 merged_vectorstore_dir 中的 candidate chunks 做相似度匹配
    5 聚合相似度结果，按 group_id 打分
    6 每个 chunk_type 保存一个 match_scores.json 到 query vectorstore 的路径下
    :param merge_vectorstore_dir:
    :param query_data_dir:
    :param top_k:
    :return:
    """