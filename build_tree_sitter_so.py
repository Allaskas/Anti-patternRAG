from tree_sitter import Language, Parser

# 构建 so 文件，生成包含 Java 语言解析器的动态库
Language.build_library(
    # 生成的 so 文件路径
    'build/my-languages.so',
    # 语言语法库的路径列表
    [
        'vendor/tree-sitter-java'
    ]
)

# 加载刚生成的 so 文件
JAVA_LANGUAGE = Language('build/my-languages.so', 'java')

# 初始化解析器
parser = Parser()
parser.set_language(JAVA_LANGUAGE)
