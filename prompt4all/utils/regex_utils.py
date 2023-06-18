
import regex

__all__ = [
    "choice_pattern","delta_pattern","replace_special_chars",'extract_score']
choice_pattern =regex.compile(r'"choices":\s*\[(\{.*?\})\]')

delta_pattern = regex.compile(r'"delta":\s*{"content":"([^"]*)"}')



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