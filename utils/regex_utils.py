
import regex


choice_pattern =regex.compile(r'"choices":\[(\{.*?\})\]')


def replace_special_chars(input_str):
    # 匹配除了英文、數字、漢字以外的字符
    pattern = r"[^a-zA-Z0-9\u4e00-\u9fa5]"

    # 使用 "_" 替換匹配到的字符
    result = regex.sub(pattern, "_", input_str)

    return result

