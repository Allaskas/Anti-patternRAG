def load_splitter_by_mode(mode):
    if mode == "java":
        from splitter.java_only_splitter import build_chunks
    elif mode == "json":
        from splitter.json_only_splitter import build_chunks
    elif mode == "json_java":
        from splitter.json_java_splitter import build_chunks
    elif mode == "ast":
        from splitter.ch_ast_splitter.ast_case_splitter import build_chunks
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return build_chunks
