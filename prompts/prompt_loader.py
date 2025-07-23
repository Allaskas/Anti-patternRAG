import os


def load_prompt(name: str) -> str:
    path = os.path.join("prompts", f"{name}.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_prompt(prompt_template: str) -> tuple[str, str]:
    """
    拆分包含 ### SYSTEM 和 ### USER 的 prompt_template 字符串，
    返回 (system_prompt, user_prompt)。

    参数：
        prompt_template (str): 包含 SYSTEM 和 USER 的完整模板。

    返回：
        Tuple[str, str]: (system_prompt, user_prompt)
    """
    system_marker = "### SYSTEM"
    user_marker = "### USER"

    if system_marker not in prompt_template or user_marker not in prompt_template:
        raise ValueError("Prompt must contain both '### SYSTEM' and '### USER' markers.")

    system_start = prompt_template.index(system_marker) + len(system_marker)
    user_start = prompt_template.index(user_marker) + len(user_marker)

    system_prompt = prompt_template[system_start:prompt_template.index(user_marker)].strip()
    user_prompt = prompt_template[user_start:].strip()

    return system_prompt, user_prompt
