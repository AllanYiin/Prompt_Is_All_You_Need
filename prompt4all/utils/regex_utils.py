
import regex

__all__ = [
    "choice_pattern","delta_pattern","json_pattern",'numbered_list_member_pattern','unordered_listitem_pattern',"replace_special_chars",'extract_score','triplequote_pattern','is_numbered_list_member','is_unordered_list_member','extract_numbered_list_member']
choice_pattern =regex.compile(r'"choices":\s*\[(\{.*?\})\]')

delta_pattern = regex.compile(r'"delta":\s*{"content":"([^"]*)"}')

json_pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')

triplequote_pattern=regex.compile(r"```(.*)```")

unordered_listitem_pattern=regex.compile(r"\s*([-*+])\s+(.*)$")
numbered_list_member_pattern=regex.compile(r'\s*(\d+(\.\d+)*\.?)(?=\s)')



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