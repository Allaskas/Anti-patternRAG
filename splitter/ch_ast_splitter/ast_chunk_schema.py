from dataclasses import dataclass
from enum import Enum


class ASTChunkType(Enum):
    PARENT_METHOD = "parent_method"
    PARENT_CALL_CHILD = "parent_call_child"
    CHILD_METHOD = "child_method"
    PARENT_FILE_STRUCTURE = "parent_file_structure"
    CHILD_FILE_STRUCTURE = "child_file_structure"


@dataclass
class ASTChunk:
    file_path: str
    chunk_type: ASTChunkType
    ast_subtree: str
