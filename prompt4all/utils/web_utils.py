import builtins
import json
import logging
import re
import random
import time
import uuid
import numpy as np
from collections import OrderedDict
from urllib.parse import urlencode, unquote
import markdownify
from scipy.ndimage import gaussian_filter
import requests
from bs4 import BeautifulSoup
from prompt4all.common import *
from prompt4all.utils.text_utils import seg_as_sentence

__all__ = ["search_google", "search_bing"]

# import chromedriver_binary
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options

"""
https://stackoverflow.com/questions/45034227/html-to-markdown-with-html2text
https://beautiful-soup-4.readthedocs.io/en/latest/#multi-valued-attributes
https://beautiful-soup-4.readthedocs.io/en/latest/#contents-and-children
"""


class CustomMarkdownConverter(markdownify.MarkdownConverter):
    def convert_a(self, el, text, convert_as_inline):
        classList = el.get("class")
        if classList and "searched_found" in classList:
            # custom transformation
            # unwrap child nodes of <a class="searched_found">
            text = ""
            for child in el.children:
                text += super().process_tag(child, convert_as_inline)
            return text
        # default transformation
        return super().convert_a(el, text, convert_as_inline)


# Create shorthand method for conversion
def md4html(html, **options):
    return CustomMarkdownConverter(**options).convert(html)


def check_useful_html_tag(tag):
    tag_list = ['header', 'copyright', 'footer', 'telephone', 'breadcrumb', 'crumb', 'menu', 'accordion', 'modal',
                'loading', 'shopping_cart']
    if 'class' in tag.attrs:
        _class = ''.join(tag.attrs['class']).lower()
        for t in tag_list:
            if t in _class:
                return False
    if 'id' in tag.attrs:
        _id = tag.attrs['id'].lower()
        for t in tag_list:
            if t in _id:
                return False
    return True


user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.2151.97',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.2151.97',
    'Mozilla/5.0 (Linux; Android 10; HD1913) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.6045.193 Mobile Safari/537.36 EdgA/119.0.2151.78',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:88.0) Gecko/20100101 Firefox/88.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Version/14.1.2 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Version/14.1 Safari/537.36',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:88.0) Gecko/20100101 Firefox/88.0'
]


def search_bing(query: str) -> list:
    """
    使用 Bing 搜索引擎根據指定查詢字串搜索信息。

    Args:
        query (str): 要搜索的查詢字串。

    Returns:
        list: 包含搜索結果的清單。每個結果是一個字典，包含 'title'（標題）, 'link'（鏈接）和 'snippet'（摘要）。

    Examples:
        >>> search_bing("site:github.com openai")
        []

        >>> search_bing("提示工程+prompt engineering")
        [{'title': '...', 'link': '...', 'snippet': '...'}, ...]

    注意:
        - 此函數使用 requests 和 BeautifulSoup 模塊來解析 Bing 的搜索結果頁面。
        - 函數捕獲網頁內容，然後從 HTML 中提取相關的標題、鏈接和摘要信息。
        - 如果搜索無結果或發生連接錯誤，將返回空清單。
    """
    query = urlencode({"q": query.replace(' ', '+')}).replace('%2B', '+')

    headers = {
        'User-Agent': random.choice(user_agents)}

    end_of_search = 'No results found for <strong>' + query + '</strong>'
    url_list = []
    search_results = []
    if_limit = True
    sw_next = True
    response = ''
    next = 1
    limit = (3 - 1) * 10 + 1
    # &qs=n&form=QBRE&sp=-1&p{query}&first={str(next)}
    session = requests.Session()
    session.headers.update(headers)
    search_url = f"https://www.bing.com/search?{query}"

    response = session.get(search_url, headers=headers)
    time.sleep(0.5)
    soup = BeautifulSoup(response.text, 'html.parser')
    results = soup.find_all('span', class_='c_tlbxTrg')
    contentinfos = soup.find_all('div', class_='b_caption')
    snippet_texts = []

    hrefs = re.findall(r"<h2><a [^>]+href=\"([^\"]+)\"", response.text)
    # hrefs = [[unquote(h.split('=')[-1]) for h in href.split(';') if h.startswith('psq=')][0] if href.startswith(
    #     'https://www.bing.com/ck') else href for href in hrefs]
    titles = re.findall(r"<h2><a [^>]*>(.*?)</a></h2>", response.text)
    clean = re.compile('<.*?>')
    titles = [re.sub(clean, '', t) for t in titles]

    for (t, h, c) in zip(titles, hrefs, contentinfos):
        try:
            txt = c.contents[0].contents[0].text
            snippet_text = c.contents[0].text[len(txt):]
            href = h
            title = t
            search_results.append({'title': title, 'link': href, 'snippet': snippet_text})
        except Exception as e:
            print('Connection Error')
            print(e)

    return search_results, session


def search_google(query: str) -> list:
    """
    使用 Google 搜索引擎根據指定查詢字串搜索信息。

    Args:
        query (str): 要搜索的查詢字串。

    Returns:
        list: 包含搜索結果的清單。每個結果是一個字典，包含 'title'（標題）, 'link'（鏈接）和 'snippet'（摘要）。

    Examples:
        >>> search_google("提示工程+prompt engineering")
        [{'title': '...', 'link': '...', 'snippet': '...'}, ...]

        >>> search_google("github.com openai")
        []

    注意:
        - 此函數使用 requests 和 BeautifulSoup 模塊來解析 Google 的搜索結果頁面。
        - 函數捕獲網頁內容，然後從 HTML 中提取相關的標題、鏈接和摘要信息。
        - 如果搜索無結果或發生連接錯誤，將返回空清單。
    """
    if 'https://www.google.com/search' in query:
        search_url = query
    else:
        query = urlencode({"q": query.replace(' ', '+')}).replace('%2B', '+')
        search_url = f"https://www.google.com/search?{query}"

    headers = {
        'User-Agent': random.choice(user_agents)}

    session = requests.Session()
    response = session.get(search_url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    search_results = {}
    total_words = len(soup.text)

    def no_div_children(tag):
        return tag.name == 'div' and 0.2 > float(len(tag.text)) / total_words > 0.02 and tag.find_all(
            'h3') and len(tag.find_all('a', href=True)) == 1

    results = soup.find_all(no_div_children)
    for r in results:
        part = BeautifulSoup(str(r), 'html.parser')
        links = part.find_all('a', href=True)
        if len(links) == 1:
            link = links[0]['href']
            title = part.find_all('h3')[0].text
            snippet_text0 = part.span.text
            part.span.extract()
            snippet_text = part.get_text(strip=True).replace(snippet_text0, '')
            if link not in search_results:
                search_results[link] = {'title': title, 'url': link, 'summary': snippet_text}
            else:
                if len(snippet_text) > len(search_results[link]['summary']):
                    search_results[link] = {'title': title, 'url': link, 'summary': snippet_text}
            # search_results.append({'title': title, 'url': link, 'summary': snippet_text})

    # links = soup.find_all('div', class_='egMi0 kCrYT')
    #
    # links = [
    #     [k.split('=')[-1] for k in t2.contents[0].attrs['href'].split('?')[-1].split('&') if k.startswith('url')][0] for
    #     t2 in links]
    # titles = [t.text for t in soup.find_all('div', class_='BNeawe vvjwJb AP7Wnd UwRFLe')]
    # snippet_texts = [t.text for t in soup.find_all('div', class_='BNeawe s3v9rd AP7Wnd')]
    # for i in range(len(links)):
    #     title = titles[i]
    #     href = links[i]
    #     snippet_text = snippet_texts[i]
    #     search_results.append({'title': title, 'link': href, 'snippet': snippet_text})
    search_results = list(search_results.values())
    search_results = {'webpage_list': search_results}
    return search_results, session


def search_answer(query: str):
    """

    Args:
        query:

    Returns:

    Examples:
        >>> search_answer("site:https://tw.stock.yahoo.com/quote/2330.TW")
        [{'title': '...', 'link': '...', 'snippet': '...'}, ...]
        >>> search_answer("prompt engineering 技巧 site:https://promptingguide.azurewebsites.net")
        []
        >>> search_answer("提示工程+prompt engineering")
        [{'title': '...', 'link': '...', 'snippet': '...'}, ...]


    """
    search_list, _session = search_bing(query)
    if len(search_list) == 0:
        search_list, _session = search_google(query)
    headers = {
        'User-Agent': random.choice(user_agents)}

    memo = OrderedDict()
    for s in search_list:
        try:
            html_text = ''
            html_text += '# {0}  \n'.format(s['title'])

            _response = _session.get(s['link'], headers=headers, allow_redirects=True)
            while _response.status_code == 204:
                time.sleep(1)
                _response = _response.head(s['link'], headers=headers, allow_redirects=True)
            html = parse_html(_response.text)
            html_text += '{0}  \n\n'.format(html)
            memo[s['link']] = html_text
        except Exception as e:
            print(e)
    return memo


def search_web(url: str):
    """

    Args:
        url:


    Returns:

    Examples:
        >>> search_web("https://www.livingplus.com.tw/shop/shop/%E5%B1%85%E5%AE%B6%E7%94%A8%E5%93%81/%E5%B1%85%E5%AE%89%E9%98%B2%E8%AD%B7/%E4%BF%9D%E9%9A%AA%E7%AE%B1/%E7%99%BC%E5%84%84%E9%87%91%E5%BA%ABst17w%E6%99%BA%E6%85%A7%E5%9E%8B%E4%BF%9D%E9%9A%AA%E7%AE%B1")
        []

    """

    headers = {
        'User-Agent': random.choice(user_agents)}
    session = requests.Session()
    session.headers.update(headers)
    response = session.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find('title').text
        [tag.decompose() for tag in soup.find_all(class_="collapse")]
        [data.decompose() for data in
         soup(['style', 'script', 'nav', 'a', 'button', 'input', 'select', 'option', 'dd', 'dt', 'dl'])]
        total_words = len(soup.text)

        def no_div_children(tag):
            return tag.name == 'div' and (
                    len(tag.text.strip()) > 10 or float(len(tag.text.strip())) / total_words > 0.05) and not [d for
                                                                                                              d in
                                                                                                              tag.find_all(
                                                                                                                  'div')
                                                                                                              if (
                                                                                                                      len(d.text.strip()) > 10 or float(
                                                                                                                  len(d.text.strip())) / total_words > 0.05)] and check_useful_html_tag(
                tag)

        divs = soup.find_all(no_div_children)

        content = '\n'.join(['\n'.join(list(div.stripped_strings)) for div in divs])
        return content, title, response.status_code
    else:
        return None, '', response.status_code


def parse_html(html: str) -> str:
    soup = BeautifulSoup(html.replace('</p>', '/n</p>')
                         .replace('<h1>', '<h1># ')
                         .replace('<h2>', '<h2>## ')
                         .replace('<h3>', '<h3>### ')
                         .replace('</h1>', '  /n</h1>')
                         .replace('</h2>', '  /n</h2>')
                         .replace('</h3>', '  /n</h3>')
                         , "html.parser")
    total_words = len(soup.text)
    main = soup.find_all('main')
    article = soup.find_all('article')
    app = soup.find('div', id='app')
    if not main and article:
        main = article

    if not main and app:
        main = app

    def no_div_children(tag):
        return tag.name == 'div' and float(len(tag.text)) / total_words > 0.5 and not [d for d in tag.find_all('div') if
                                                                                       float(
                                                                                           len(d.text)) / total_words > 0.5]

    if main:
        divs = main[0].find_all(no_div_children)
        # divs = [d for d in divs if float(len(d.text)) / total_words > 0.5]
        if len(divs) == 0:
            raw = main[0].text
        else:
            raw = '\n\n'.join([d.text for d in divs])
    else:
        divs = soup.find_all('div')
        divs = [d for d in divs if len(d.text) > 10]
        # raw = '\n\n'.join([d.text for d in divs])
        raw = md4html(html)
    raw = re.sub(r'\n\s*\n', '\n', raw)
    return raw


def search_webpilot(url: str, *args, **kwargs) -> str:
    """

    Args:
        url:
        *args:
        **kwargs:

    Returns:

    Examples:
        >>> search_webpilot("https://www.businesstoday.com.tw/article/category/80394/post/202104150009/")
        []

    """

    header = {
        "Content-Type": "application/json",
        "WebPilot-Friend-UID": str(uuid.uuid4()),
    }

    data = {
        "link": url,
        "ur": "search",
        "l": 'zh-TW',
        "lp": True,
        "rt": False
    }
    endpoint = "https://webreader.webpilotai.com/api/visit-web"
    resp = requests.post(endpoint, headers=header, json=data)

    logging.debug("webpilot resp: {}".format(resp.json()))
    # temp = resp.json()
    # if 'content' in temp:
    #     temp['content'] = cleasing_web_text(temp['content'])

    return json.dumps(resp.json(), ensure_ascii=False)


# def parse_xml(xml: str) -> etree._Element:
#     return etree.fromstring(xml)


# with ThreadPoolExecutor(max_workers=5) as executor:
#     future_to_item = {executor.submit(get_summary, item, model, search_terms): item for item in items}
#     for future in as_completed(future_to_item):
#         try:
#             summary = future.result(timeout=5)  # 设置超时时间
#             if summary is not None:
#                 all_summaries.append("【搜索结果内容摘要】：\n" + summary)
#         except concurrent.futures.TimeoutError:
#             logger.error("处理摘要任务超时")
#         except Exception as e:
#             logger.error("在提取摘要过程


def second_derivative(x):
    return np.gradient(np.gradient(x))


def cleasing_web_text(text: str):
    """

    Args:
        url:
        *args:
        **kwargs:

    Returns:

    Examples:
        >>> cleasing_web_text(eval(search_webpilot("https://www.businesstoday.com.tw/article/category/80394/post/202104150009/"))['content'])
        []

    """

    lines = []
    is_valuable = []
    for t in text.replace(' ', '\n\n').split('\n'):
        if len(t) > 300:
            ss = seg_as_sentence(t)
            lines.extend(ss)
            is_valuable.extend([1] * len(ss))
        elif len(t) > 0:
            lines.append(t)
            is_valuable.append(0)

    is_valuable = np.array(is_valuable)
    text_lens = np.array([len(t) for t in lines])
    total_words = np.array(text_lens).sum()
    freq = {}
    for t in lines:
        if len(t) > 0:
            if len(t) not in freq:
                freq[len(t)] = 0
            freq[len(t)] += 1
    sorted_freq = sorted(freq.items(), key=lambda kv: (kv[0], kv[1]), reverse=True)
    sorted_freq = {k: v for k, v in sorted_freq}
    keys = list(sorted_freq.keys())
    remain_text = total_words
    current_len = None
    need_check = True
    while remain_text > 0.05 * total_words and (current_len is None or current_len > 10) and need_check:
        this_len = keys.pop(0)
        match_cnt = len(text_lens[text_lens == this_len])
        if match_cnt == 1:
            is_valuable[text_lens == this_len] = 1
            remain_text -= this_len
        else:
            if current_len and this_len / current_len > 0.5:
                is_valuable[text_lens == this_len] = 1
                remain_text -= this_len * match_cnt
            elif current_len and this_len / current_len < 0.5:
                need_check = False
        current_len = this_len

    results = []
    in_valid_zone = False
    partial_words = ''
    for idx in range(len(lines)):
        t = lines[idx]
        check = is_valuable[idx]
        if check == 1:
            if not in_valid_zone:
                in_valid_zone = True
            if len(partial_words) == 0:
                partial_words = t
            else:
                partial_words += '\n\n' + t
            if len(partial_words) > 100:
                results.append(partial_words)
                print(green_color(partial_words), flush=True)
                partial_words = ''
        else:
            if in_valid_zone:
                if (idx > 0 and is_valuable[idx - 1] == 1) or (idx < len(lines) - 1 and is_valuable[idx + 1] == 1):
                    if len(partial_words) > 20:
                        results.append(partial_words)
                        print(green_color(partial_words), flush=True)
                    partial_words = t
            else:
                print(magenta_color(t), flush=True)
    return results

# def cleasing_web_text(text: str):
#     lines = text.replace(' ', '\n\n').split('\n')
#     text_lens = [len(t) for t in lines if len(t) > 0]
#     freq = {}
#     for t in lines:
#         if len(t) > 0:
#             if len(t) not in freq:
#                 freq[len(t)] = 0
#             freq[len(t)] += len(t)
#     sorted_freq = sorted(freq.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
#     sorted_freq = {k: v for k, v in sorted_freq}
#     total_words = np.array(text_lens).sum()
#     keys = np.array(list(sorted_freq.keys()))
#     text_lens_ratio = np.array(list(sorted_freq.values())) / total_words
#     text_lens_accu_ratio = np.array([text_lens_ratio[:i + 1].sum() for i in range(len(sorted_freq))])
#     x_array = np.array([sorted_freq[k] / k / len(text_lens) for k in keys])
#     accu_x_array = np.array([x_array[:i + 1].sum() for i in range(len(sorted_freq))])
#     slop_array = text_lens_accu_ratio / accu_x_array
#     slop_array_1 = np.array(
#         [slop_array[i] - slop_array[i - 1] if i > 0 else slop_array[i] for i in range(len(slop_array))])
#     return text_lens_accu_ratio
