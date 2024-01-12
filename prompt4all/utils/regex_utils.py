import regex

__all__ = [
    "choice_pattern", "delta_pattern", "json_pattern", 'numbered_list_member_pattern', 'unordered_listitem_pattern',
    "replace_special_chars", 'extract_score', 'extract_code', 'triplequote_pattern', 'is_numbered_list_member',
    'is_unordered_list_member', 'extract_numbered_list_member', 'find_all_placeholders', 'md_table_pattern']

choice_pattern = regex.compile(r'"choices":\s*\[(\{.*?\})\]')

delta_pattern = regex.compile(r'"delta":\s*{"content":"([^"]*)"}')

json_pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')

triplequote_pattern = regex.compile(r"```(.*)```")

unordered_listitem_pattern = regex.compile(r"\s*([-*+])\s+(.*)$")
numbered_list_member_pattern = regex.compile(r'\s*(\d+(\.\d+)*\.?)(?=\s)')

code_pattern = regex.compile(r'```(.*?)```')

md_table_pattern = regex.compile(r"^(\|.*\|)\r?\n\|([ :-]+)\|\r?\n(\|(.*\|)+)", regex.MULTILINE)


def find_all_placeholders(text):
    """
    從給定的文本中提取所有的占位符。

    :param text: 包含占位符的字符串。
    :return: 包含所有占位符的列表。

    Examples:
        >>> find_all_placeholders('Where is the @placeholder??  @Placeholder(getdiagram_result220102), can you tell me?')
        ['@Placeholder(get_diagram_result_220102)']

    """
    pattern = r"@Placeholder\((\w+)\)"
    # pattern = r'\s*@Placeholder(.*?)\s*'  # 建立正則表達式的規則
    matches = regex.findall(pattern, text)  # 使用 re.findall() 方法找出所有匹配的字串

    return matches


def replace_special_chars(input_str):
    # 匹配除了英文、數字、漢字以外的字符
    pattern = r"[^a-zA-Z0-9\u4e00-\u9fa5]"

    # 使用 "_" 替換匹配到的字符
    result = regex.sub(pattern, "_", input_str)

    return result


def extract_score(text):
    pattern = r"(\d+)分"
    result = regex.search(pattern, text)
    if result:
        return int(result.group(1))
    else:
        return None


def extract_code(text):
    """
    從給定的文本中提取代碼區塊。

    :param text: 包含代碼區塊的字符串。
    :return: 包含所有代碼區塊的列表。
    """
    code_blocks = regex.findall(r'```(.*?)```', text, regex.DOTALL)
    if code_blocks and len(code_blocks) > 0:
        return code_blocks[0][4:]
    elif text.lower().startswith('select'):
        return text
    return code_blocks


def is_numbered_list_member(string):
    return bool(regex.match(numbered_list_member_pattern, string))


def is_unordered_list_member(string):
    return bool(regex.match(unordered_listitem_pattern, string))


def extract_numbered_list_member(string):
    """

    :param string:
    :return:

    Examples:
        >>> print(extract_numbered_list_member("1. This is a numbered list member."))
        1.
        >>> print(extract_numbered_list_member("19. This is a numbered list member."))
        19.
        >>> print(extract_numbered_list_member("    1.3.1 This is a numbered list member."))
        1.3.1
        >>> print(extract_numbered_list_member("   1.2 This is a numbered list member."))
        1.2
        >>> print(extract_numbered_list_member("    1.2 This is a numbered list member."))
        1.2
        >>> print(extract_numbered_list_member("    1This is a numbered list member."))
        None
    """
    match = regex.search(numbered_list_member_pattern, string)
    if match:
        return match.group(1)
    else:
        return ''


def extract_unordered_list_member(string):
    """

    :param string:
    :return:

    Examples:
        >>> print(extract_unordered_list_member("1. This is a numbered list member."))
        <BLANKLINE>
        >>> print(extract_unordered_list_member("- This is a numbered list member."))
        -
        >>> print(extract_unordered_list_member("    - This is a numbered list member."))
        -
        >>> print(extract_unordered_list_member("   -This is a numbered list member."))
        -
        >>> print(extract_unordered_list_member("    + This is a numbered list member."))
        +
        >>> print(extract_unordered_list_member("This is a numbered list member."))
        <BLANKLINE>
    """
    match = regex.search(unordered_listitem_pattern, string)
    if match:
        return match.group(1)
    else:
        return ''
