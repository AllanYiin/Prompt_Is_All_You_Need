import json
import logging
import re
import random
import time
import uuid
from collections import OrderedDict
from urllib.parse import urlencode, unquote
import markdownify
import requests
from bs4 import BeautifulSoup

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
    query = urlencode({"q": query.replace(' ', '+')}).replace('%2B', '+')
    headers = {
        'User-Agent': random.choice(user_agents)}

    search_url = f"https://www.google.com/search?{query}"
    session = requests.Session()
    response = session.get(search_url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    search_results = []

    results = soup.find_all('div')
    results = [r for r in results if 'egMi0 kCrYT' in str(r)]
    for r in results:
        part = BeautifulSoup(str(r), 'html.parser')
        link = part.find_all('div', class_='egMi0 kCrYT')
        if len(link) == 1:
            link = [
                [k.split('=')[-1] for k in t2.contents[0].attrs['href'].split('?')[-1].split('&') if
                 k.startswith('url')][0]
                for t2 in link][0]
            title = part.find_all('div', class_='BNeawe vvjwJb AP7Wnd UwRFLe')[0].text
            snippet_text = part.find_all('div', class_='BNeawe s3v9rd AP7Wnd')
            if len(snippet_text) > 0:
                snippet_text = snippet_text[0].text
                search_results.append({'title': title, 'link': link, 'snippet': snippet_text})

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
    search_results = list(set([str(s) for s in search_results]))
    search_results = [eval(s) for s in search_results]
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
        >>> search_webpilot("https://zhuanlan.zhihu.com/p/626966526")
        []

    """

    header = {
        "Content-Type": "application/json",
        "WebPilot-Friend-UID": str(uuid.uuid4()),
    }

    data = {
        "link": url,
        'user_has_request': False
    }
    endpoint = "https://webreader.webpilotai.com/api/visit-web"
    resp = requests.post(endpoint, headers=header, json=data)

    logging.debug("webpilot resp: {}".format(resp.json()))

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
