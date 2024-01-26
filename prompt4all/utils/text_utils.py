import re
from typing import List, TypedDict
from itertools import repeat
from functools import partial
import string

_alphabets = "([A-Za-z])"
_numbers = "([0-9])"
_prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
_suffixes = "(Inc|Ltd|Jr|Sr|Co)"
_starters = "(Mr|Mrs|Ms|Dr|He\\s|She\\s|It\\s|They\\s|Their\\s|Our\\s|We\\s|But\\s|However\\s|That\\s|This\\s|Where\\s|When\\s|Who\\s|Why\\s)"
_acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_websites = "[.](com|net|org|io|gov)"
__all__ = [
    "seg_as_sentence", "chinese_full2half", "chinese_half2full", "convert_data"]


def seg_as_sentence(txt: str) -> list[str]:
    """
    將文字按照語意結束來截斷。

    Args:       txt (str): 輸入字串。

    Returns:
        list[str]: 切割開的文字清單。

    Examples:
        >>> seg_as_sentence("IC 設計大廠聯發科 (2454-TW) 今 (8) 日公告 11 月營收 430.71 億元，創近 14 個月新高，月增 0.6%、年增 19.23%；累計前 11 月營收 3897.66 億元，年減 23.59%。跟據聯發科法說展望，第四季營收以美元兑新台幣匯率 1 比 32 計算，第四季營收 1200-1266 億元，季增 9-15%，年增 11-17%，達 1200 億元以上，將創五季來新高，單季也轉為年成長。聯發科第四季受惠新一代旗艦晶片天璣 9300 開始出貨，帶動手機業務營收強勁成長，抵銷智慧裝置平台季節性下滑影響。不過，Wi-Fi 7 解決方案已獲高階路由器、高階筆電和寬頻設備採用，預期明年會有更多採用 Wi-Fi 7 產品推出。")
        ['IC 設計大廠聯發科 (2454-TW) 今 (8) 日公告 11 月營收 430.71 億元，創近 14 個月新高，月增 0.6%、年增 19.23%；累計前 11 月營收 3897.66 億元，年減 23.59%。', '跟據聯發科法說展望，第四季營收以美元兑新台幣匯率 1 比 32 計算，第四季營收 1200-1266 億元，季增 9-15%，年增 11-17%，達 1200 億元以上，將創五季來新高，單季也轉為年成長。', '聯發科第四季受惠新一代旗艦晶片天璣 9300 開始出貨，帶動手機業務營收強勁成長，抵銷智慧裝置平台季節性下滑影響。', '不過，Wi-Fi 7 解決方案已獲高階路由器、高階筆電和寬頻設備採用，預期明年會有更多採用 Wi-Fi 7 產品推出。']

    """
    # 將前綴後的點替換成特殊標記<prd>，以避免被錯誤切分
    txt = re.sub(_prefixes, "\\1<prd>", txt)

    # 對於網址中的點進行相同的處理
    txt = re.sub(_websites, "<prd>\\1", txt)

    # 特殊處理"Ph.D."，以免被錯誤切分
    if "Ph.D" in txt:
        txt = txt.replace("Ph.D.", "Ph<prd>D<prd>")

    # 替換英文字母後的點
    txt = re.sub("\s" + _alphabets + "[.] ", " \\1<prd> ", txt)

    # 處理縮寫詞後跟著句首詞的情況
    txt = re.sub(_acronyms + " " + _starters, "\\1<stop> \\2", txt)

    # 替換連續的英文字母和點的組合
    txt = re.sub(_alphabets + "[.]" + _alphabets + "[.]" + _alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", txt)
    txt = re.sub(_alphabets + "[.]" + _alphabets + "[.]", "\\1<prd>\\2<prd>", txt)

    # 數字後的點也進行特殊處理
    txt = re.sub(_numbers + "[.]" + _numbers, "\\1<prd>\\2", txt)

    # 處理後綴和句首詞的組合
    txt = re.sub(" " + _suffixes + "[.] " + _starters, " \\1<stop> \\2", txt)
    txt = re.sub(" " + _suffixes + "[.]", " \\1<prd>", txt)
    txt = re.sub(" " + _alphabets + "[.]", " \\1<prd>", txt)

    # 處理引號和句末標點的組合
    if "”" in txt:
        txt = txt.replace(".”", "”.")
    if "\"" in txt:
        txt = txt.replace(".\"", "\".")
    if "!" in txt:
        txt = txt.replace("!\"", "\"!")
    if "?" in txt:
        txt = txt.replace("?\"", "\"?")

    # 替換省略號和接下來的字符
    txt = re.sub("(\.\.\.)([^”’])", "<prd><prd><prd><stop>", txt)

    # 將句末的點、問號、驚嘆號替換為<stop>
    txt = txt.replace(".", ".<stop>")
    txt = txt.replace("?", "?<stop>")
    txt = txt.replace("!", "!<stop>")

    # 恢復先前替換的<prd>為正常的點
    txt = txt.replace("<prd>", ".")

    # 對中文的標點進行處理
    txt = re.sub('([。！？\?])([^”’])', r"\1<stop>\2", txt)
    txt = re.sub('(\.{6})([^”’])', r"\1\<stop>\2", txt)
    txt = re.sub('([。！？\?][”’])([^。！？\?])', r"\1\<stop>\2", txt)

    # 移除尾部空白並按<stop>分割
    txt = txt.replace('\n', "<stop>")
    txt = txt.rstrip()
    sentences: list[str] = txt.split("<stop>")
    return [s for s in sentences if len(s) > 0]


def chinese_full2half(input_str):
    """Convert all fullwidth Chinese characters to halfwidth .

    Returns:

    """
    rstring = ""
    for uchar in input_str:
        u_code = ord(uchar)
        if u_code == 0x3000 or u_code == 12288 or uchar == string.whitespace:
            u_code = 32
        elif 65281 <= u_code <= 65374:
            u_code -= 65248
        rstring += chr(u_code)
    return rstring


def chinese_half2full(input_str):
    """Convert all halfwidth Chinese characters to fullwidth .

    Returns:

    """
    rstring = ""
    for uchar in input_str:
        u_code = ord(uchar)
        if u_code == 32:
            u_code = 12288
        elif 33 <= u_code <= 126:
            u_code += 65248
        rstring += chr(u_code)
    return rstring


def convert_data(text):
    """
        Return the first number in the given text for any locale.
        TODO we actually don't take into account spaces for only
        3-digited numbers (like "1 000") so, for now, "1 0" is 10.
        TODO parse cases like "125,000.1,0.2" (125000.1).

        :example:
        >>> convert_data(chinese_full2half('−1.5284'))
        -1.5284
       >>> convert_data("1190,00 €")
       '1190,00 €'
        >>> convert_data("1,190.00 €")
        1190
        >>> convert_data("$1190.00")
        1190
        >>> convert_data("rrr1rrr")
        'rrr1rrr'
        >>> convert_data("$.3")
        0.3
        >>> convert_data("125,00 €")
        '125,00 €'
        >>> convert_data("100,000,000")
        100000000
        >>> convert_data("a 125,00 €")
        'a 125,00 €'
        >>> convert_data("100.000,000")
        '100.000,000'
        >>> convert_data("100 000,000")
        '100 000,000'
        >>> convert_data("100 000 000")
        '100 000 000'
        >>> convert_data("100.001 ")
        100.001
        >>> convert_data(".003")
        0.003
        >>> convert_data(".003 ")
        0.003
        >>> convert_data("3 005")
        '3 005'
        >>> convert_data("1.190,00 €")
        '1.190,00 €'
        >>> convert_data("$1 190.99")
        '$1 190.99'
        >>> convert_data("$-1 190.99")
        '$-1 190.99'
        >>> convert_data("1 000 000.3")
        '1 000 000.3'
        >>> convert_data('-151.744122')
        -151.744122
        >>> convert_data('-1')
        -1
        >>> convert_data('1e-3')
        0.001
        >>> convert_data("1 0002,1.2")
        '1 0002,1.2'
        >>> convert_data("")
        >>> convert_data(None)

        >>> convert_data(1)
        1
        >>> convert_data("rrr1,.2o")
        'rrr1,.2o'
        >>> convert_data("rrr ,.o")
        'rrr ,.o'
    """
    # First we return None if we don't have something in the text:
    if text is None:
        return None
    if isinstance(text, int) or isinstance(text, float):
        return text
    # 去除字串前後的空白與換行符號
    orig_string = chinese_full2half(text.strip()).replace('−', '-')
    s = orig_string
    if s == "":
        return None
    # 檢查字串是否為有效數字
    # 使用正則表達式匹配數字區域
    pattern = r"([-+]?[^\d\s]*)(\s*)([\d,]*\.?\d*)(\s*)([^\d\s]*)(\s*)([eE][-+]?\d+)?"
    match = re.fullmatch(pattern, s)
    # 如果匹配成功，則進一步檢查數字區域的細節
    if match:
        # 獲取數字區域前後的貨幣符號或負號
        prefix = match.group(1)
        suffix = match.group(5)

        # 檢查數字區域前後是否只出現一次貨幣符號
        if len(re.findall(r"\$", prefix + suffix)) > 1:
            return orig_string  # 不是有效數字，返回字串

        # 檢查數字區域中的comma符號是否每3位出現一次
        digits = match.group(3)
        # 如果有小數點，則在小數點前面補上一個0
        if prefix and prefix[-1] == ".":
            digits = "0." + digits

        test_prefix = prefix.strip('¥$€£₩+-. ')
        test_suffix = suffix.strip('¥$€£₩+-. ')
        if len(test_prefix) > 0 and len(test_suffix) > 0:
            return orig_string
        prefix = prefix.strip('¥$€£₩+. ')

        if ',' in digits:
            if '.' in digits:
                parts = '.'.join(digits.split('.')[:-1]).split(",")
            else:
                parts = digits.split(",")
            for part in parts[1:]:
                if len(part) != 3:
                    return orig_string  # 不是有效數字，返回字串
            # 檢查數字區域中的小數點是否在comma之後
            if "." in digits and "," in digits:
                if digits.index(".") < digits.index(","):
                    return orig_string  # 不是有效數字，返回字串
            # 如果通過以上檢查，則將數字區域轉換為float或integer
            # 去除comma符號
            digits = digits.replace(",", "")

        # 獲取科學符號
        exponent = match.group(7) or ""
        number = prefix + digits + exponent
        # 轉換為float或integer
        try:
            number = float(number)
            if number.is_integer():
                number = int(number)
            return number  # 返回數字
        except Exception as e:
            print(e)
            return orig_string  # 轉換失敗，返回字串
    else:
        return orig_string  # 不匹配正則表達式，返回字串

    # try:
    #     # First we return None if we don't have something in the text:
    #     if text is None:
    #         return None
    #     if isinstance(text, int) or isinstance(text, float):
    #         return text
    #     text = text.strip()
    #     if text == "":
    #         return None
    #     # Next we get the first "[0-9,. ]+":
    #     # Check if text is a number
    #     if re.match(r'^[-¥$€£₩]?(\d+\.?\d*|\.\d+)[¥$€£₩]?$', text):
    #         # Find the first number in text
    #         n = re.search(r'[-¥$€£₩]?(\d+\.?\d*|\.\d+)[¥$€£₩]?', text).group(0)
    #         n.strip()
    #         # Remove currency symbols
    #         n = re.sub(r'[¥$€£₩]', '', n)
    #         n = re.sub(r',(?=\d{3})', '', n)
    #         # Remove spaces
    #         n = re.sub(r'\s+', '', n)
    #         # And we cast the text to float or int:
    #         n = float(n)
    #         if n.is_integer():
    #             return int(n)
    #         else:
    #             return n
    #     else:
    #         # Return the original text
    #         return text
    # except:
    #     pass
    # return text
