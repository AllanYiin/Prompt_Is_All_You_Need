import re
from typing import List

_alphabets = "([A-Za-z])"
_numbers = "([0-9])"
_prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
_suffixes = "(Inc|Ltd|Jr|Sr|Co)"
_starters = "(Mr|Mrs|Ms|Dr|He\\s|She\\s|It\\s|They\\s|Their\\s|Our\\s|We\\s|But\\s|However\\s|That\\s|This\\s|Where\\s|When\\s|Who\\s|Why\\s)"
_acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_websites = "[.](com|net|org|io|gov)"


def seg_as_sentence(txt: str) -> list[str]:
    """
    將文字按照語意結束來截斷。

    Args:
        txt (str): 輸入字串。

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
    txt = re.sub('([。！？\?][”’])([^，。！？\?])', r"\1\<stop>\2", txt)

    # 移除尾部空白並按<stop>分割
    txt = txt.rstrip()
    sentences: list[str] = txt.split("<stop>")
    return sentences
