import builtins
import json
import logging
import random
import re
import copy
import time
import uuid
import io
from collections import OrderedDict
from urllib.parse import urlencode
from itertools import chain
import markdownify
import numpy as np
import requests
from bs4 import BeautifulSoup, Tag, NavigableString
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from prompt4all import context
from prompt4all.context import *
from prompt4all.common import *
from prompt4all.api.memories import *
from prompt4all.utils.markdown_utils import HTML2Text, htmltable2markdown
from prompt4all.utils.text_utils import seg_as_sentence
from prompt4all.utils.regex_utils import count_words

cxt = context._context()
if cxt.memory is None:
    cxt.memory = InMemoryCache()
    cxt.memory.load()

__all__ = ["search_google", "search_bing", "user_agents"]

ignored_exceptions = (NoSuchElementException, StaleElementReferenceException,)


def prepare_chrome_options():
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('blink-settings=imagesEnabled=false')
    chrome_options.add_argument('--disable-logging')
    chrome_options.add_argument(f"--window-size=1920,1440")
    chrome_options.add_argument('--hide-scrollbars')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_argument("--proxy-server='direct://'")
    chrome_options.add_argument("--proxy-bypass-list=*")
    chrome_options.add_argument('--ignore-certificate-errors')
    chrome_options.add_argument("--password-store=basic")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--enable-automation")
    chrome_options.add_argument("--disable-browser-side-navigation")
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-setuid-sandbox")
    chrome_options.add_argument("--disable-software-rasterizer")
    return chrome_options


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
    total_words = len(soup.text)

    def h2_with_a(tag):
        return tag.name == 'h2' and tag.find_all('a')

    results = soup.find_all(h2_with_a)
    titles = [t.text for t in results]
    hrefs = [t.find_all('a')[0]['href'] for t in results]

    # results = soup.find_all('span', class_='c_tlbxTrg')
    contentinfos = soup.find_all('div', class_='b_caption')
    snippet_texts = []
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
            PrintException()
    if len(search_results) == 5:
        search_results = search_results[:5]

    return search_results, session


def search_google(query: str) -> list:
    """
    使用 Google 搜索引擎根據指定查詢字串搜索信息。

    Args:
        query (str): 要搜索的查詢字串。

    Returns:
        list: 包含搜索結果的清單。每個結果是一個字典，包含 'title'（標題）, 'link'（鏈接）和 'snippet'（摘要）。

    Examples:
        >>> search_google("https://www.google.com/search?q=%E8%8F%B1%E6%A0%BC%E7%B4%8B")
        []
        >>> search_google("提示工程+prompt engineering")
        [{'title': '...', 'link': '...', 'snippet': '...'}, ...]



    注意:
        - 此函數使用 requests 和 BeautifulSoup 模塊來解析 Google 的搜索結果頁面。
        - 函數捕獲網頁內容，然後從 HTML 中提取相關的標題、鏈接和摘要信息。
        - 如果搜索無結果或發生連接錯誤，將返回空清單。
    """
    if 'https://www.google.com/search' in query:
        # url_parts = query.strip().split('q=')
        # search_url = url_parts[0] + 'q=' + url_parts[-1].strip().replace(' ', '%2B').replace('+', '%2B').replace(':','%3A')
        search_url = query
    else:
        query = urlencode({"q": query.strip().replace(' ', '+')}).replace('%2B', '+')
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

    def div_with_media(tag):
        return tag.name == 'div' and (
                (len([t for t in tag.contents if
                      t.name == 'img' and 'alt' in t.attrs and count_words(t.attrs['alt']) >= 8]) > 0)
                or (len([t for t in tag.contents if t.name == 'a' and count_words(t.get('aria-label')) >= 10]) > 0
                    and len([t for t in tag.find_all('svg')]) > 0))

    results = soup.find_all(no_div_children)
    media_results = soup.find_all(div_with_media)
    media_references = []
    for tag in media_results:
        vedio_url = [t.attrs['data-url'] for t in tag.find_all('div') if
                     'dara-url' in t.attrs and len(t.attrs['data-url']) > 0]
        if len(vedio_url) > 0:
            cxt.citations.append(
                '<video width="148" height="83" controls><source src="{0}" type="video/mp4"></video>'.format(
                    vedio_url[0]))
    if len(cxt.citations) > 0:
        print('citations', cyan_color('\n' + '\n'.join(cxt.citations)))
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
    if len(search_results) >= 5:
        search_results = search_results[:5]
    print('google search results:', green_color(str(search_results)), flush=True)
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
            PrintException()
    return memo


def search_web(url: str) -> list:
    """

    Args:
        url:


    Returns:

    Examples:
        >>> search_web('https://en.wikipedia.org/wiki/Electric_vehicle_charging_station')

    """
    print(cyan_color("search_web:{0}".format(url)), flush=True)
    text_frags = []
    if url.startswith('http://') or url.startswith('https://'):
        chrome_options = prepare_chrome_options()
        chrome_options.add_argument('user-agent=%s' % random.choice(user_agents))

        # session = requests.Session()
        # session.headers.update(headers)
        # response = session.get(url, headers=headers, allow_redirects=True)
        # 建立Chrome瀏覽器物件
        resulttext = ''
        title = ''
        banners = []
        contents = []
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)

        driver.implicitly_wait(3)
        try:

            body_rect = copy.deepcopy(driver.find_element(By.TAG_NAME, 'body').rect)
            windows_rect = driver.get_window_rect()
            window_width = body_rect['width']
            window_height = body_rect['height']
            head_html = driver.find_element(By.TAG_NAME, 'head').get_attribute('outerHTML')
            body_html = driver.find_element(By.TAG_NAME, 'body').get_attribute('outerHTML')
            driver.refresh()
            for d in driver.find_elements(By.TAG_NAME, 'div'):
                try:
                    drect = copy.deepcopy(d.rect)
                    drect['outerHTML'] = d.get_attribute('outerHTML').strip()
                    drect['text'] = d.get_attribute("textContent").strip()
                    if not d.is_displayed():
                        pass
                    elif drect['height'] * drect['width'] < 100:
                        pass
                    elif window_height > 0 and drect['height'] / float(window_height) > 0.6 and drect['width'] / float(
                            window_width) > 0.6 and not (drect['x'] == 0 and drect['y'] == 0):
                        if len(drect['text']) > 10:
                            if len(contents) == 0 or drect['outerHTML'] not in contents[-1]['outerHTML']:
                                if len(contents) > 0 and drect['text'] in contents[-1]['text'] and len(
                                        drect['text']) > 50 and len(contents[-1]['text']) > 50:
                                    pass
                                else:
                                    contents.append(drect)
                    elif drect['height'] / drect['width'] > 5 and drect['height'] > 0.5 * window_height and (
                            drect['x'] < window_height / 4 or drect['x'] > 3 * window_height / 4):
                        if len(banners) == 0 or drect['outerHTML'] not in banners[-1]['outerHTML']:
                            banners.append(drect)
                    elif (drect['width'] / drect['height']) / (window_width / window_height) > 5 and drect[
                        'width'] > 0.5 * window_width and (
                            drect['y'] < windows_rect['height'] / 4 or drect['y'] > 3 * window_height / 4):
                        if len(banners) == 0 or drect['outerHTML'] not in banners[-1]['outerHTML']:
                            banners.append(drect)
                    elif 0.5 < drect['width'] / drect['height'] < 2 and drect['width'] > 0.5 * window_width and \
                            drect['height'] > 0.5 * window_height and (drect['y'] < window_height / 3):
                        if count_words(drect['text']) > 10:
                            if len(contents) == 0 or drect['outerHTML'] not in contents[-1]['outerHTML']:
                                if len(contents) > 0 and drect['text'] in contents[-1]['text'] and count_words(
                                        drect['text']) > 50 and count_words(contents[-1]['text']) > 50:
                                    pass
                                else:
                                    contents.append(drect)
                    else:
                        pass
                except NoSuchElementException:
                    pass
                except StaleElementReferenceException:
                    pass
                except Exception as e:
                    print(d, flush=True)
                    PrintException()

            final_content = []

            def get_banner_overlap_areas(c):
                areas = 0
                c_html = c['outerHTML']
                for b in banners:
                    b_html = b['outerHTML']
                    if b_html in c_html:
                        areas += int(b['width']) * int(b['height'])
                return areas

            if len(contents) > 0:
                areas = [get_banner_overlap_areas(c) for c in contents]
                min_area = np.array(areas).min()
                final_content = [contents[cidx] for cidx in range(len(contents)) if areas[cidx] == min_area]
            if len(final_content) > 0:
                content_html = '<html>' + head_html + '<body>' + ''.join(
                    [c['outerHTML'] for c in final_content]) + '</body></html>'
                html = content_html
            else:
                content_html = '<html>' + head_html + body_html + '</html>'
                html = content_html
        except Exception as e:
            PrintException()
            print(e)
            html = driver.page_source
        driver.quit()

        def no_div_children(tag):
            return (tag.name == 'table') or (tag.name == 'p' and tag.text and len(tag.text.strip()) > 0) or (
                    tag.name == 'div' and tag.text and len(tag.text.strip()) > 0 and len(
                tag.find_all('p')) == 0 and not [d for d in tag.find_all('div') if
                                                 len(d.text.strip()) > 20])

        if html:
            try:
                # tables = htmltable2markdown(html)
                for banner in banners:
                    html = html.replace(banner['outerHTML'], '')
                soup = BeautifulSoup(html, 'html.parser')
                if soup.find('title'):
                    title = soup.find('title').text.strip()

                [data.decompose() for data in
                 soup(['style', 'script', 'nav', 'button', 'input', 'select', 'option', 'dd', 'dt', 'dl', 'abbr'])]
                for tag in soup.find_all('div'):
                    div_children = tag.find_all('div')
                    div_children = [count_words(d.text.strip()) for d in div_children if
                                    d.text and len(d.text.strip()) > 0]
                    if len(div_children) > 0:
                        mean_len = np.array(div_children).mean()
                        if len(div_children) > 5 and mean_len < 10:
                            tag.decompose()
                for banner in banners:
                    b = BeautifulSoup(banner['outerHTML'], 'html.parser').contents[0]
                    if b.get('id'):
                        b_area = soup.find(id=b.get('id'))
                        if b_area is not None:
                            b_area.decompose()
                    elif b.get('class'):
                        if len(soup.find_all(class_=b.get('class'))) == 1:
                            b_area = soup.find(class_=b.get('class'))
                            if b_area is not None:
                                b_area.decompose()
                # ps = soup.findAll('p')
                # for _idx in range(len(ps)):
                #     p = ps[_idx]
                #     _div = Tag(soup, name='div')  # create a P element
                #     p.replaceWith(_div)  # Put it where the A element is
                #     _div.insert(0, p.text.strip())
                total_words = count_words(soup.text.strip())

                def tag2markdown(tag, idx):
                    if tag.name == 'table':
                        _parts_text = htmltable2markdown(tag.prettify(formatter=None))[0]
                        text_frags.append(
                            build_text_fragment(source=url, page_num=idx, paragraph_num=0, text=_parts_text))
                    else:
                        h = HTML2Text(baseurl=url)
                        _parts_text = h.handle(
                            '<html>' + head_html + '<body>' + tag.prettify(formatter=None) + '</body></html>')
                        text_frags.append(
                            build_text_fragment(source=url, page_num=idx, paragraph_num=0, text=_parts_text))
                    return _parts_text

                def process_long_item(p):
                    paras = []
                    for s in p.text.strip().split('\n'):
                        if len(s) == 0:
                            pass
                        elif count_words(s) <= 300:
                            paras.append(s)
                        elif count_words(s) > 300:
                            paras.extend(seg_as_sentence(s))

                    # 選單
                    if count_words(p.text.strip()) / len(paras) < 10:
                        return []
                    elif np.array([1.0 if _p.strip().startswith('^') else 0.0 for _p in paras if
                                   len(_p.strip()) > 1]).mean() > 0.9:
                        return []  # 參考文獻、引用
                    group_results = optimal_grouping([count_words(_p.strip()) for _p in paras], min_sum=200,
                                                     max_sum=300)
                    results = []
                    # slot_group = copy.deepcopy(group_results)
                    current_idx = 0

                    # 把字數分組當作插槽插入tag
                    slot_group = copy.deepcopy(group_results)
                    for n in range(len(group_results)):
                        this_group = group_results[n]
                        if len(this_group) > 1:
                            for g in range(len(this_group)):
                                slot_group[n][g] = paras[current_idx]
                                current_idx += 1
                        elif len(this_group) == 1:
                            slot_group[n][0] = paras[current_idx]
                            current_idx += 1

                    for idx in range(len(slot_group)):
                        g = slot_group[idx]
                        if len(g) > 0:
                            this_text = ''.join(g) if len(g) > 1 else g[0]
                            if len(this_text) > 0:
                                tag = Tag(soup, name="div")
                                text = NavigableString(this_text)
                                tag.insert(0, text)
                                results.append(tag)
                    return results

                parts = [soup.contents[0]] if total_words < 300 else soup.find_all(no_div_children)
                new_parts = []
                for p in parts:
                    if count_words(p.text.strip()) > 300:
                        new_parts.extend(process_long_item(p))
                    else:
                        new_parts.append(p)
                parts = new_parts
                group_results = optimal_grouping([count_words(p.text.strip()) for p in parts], min_sum=200,
                                                 max_sum=300)
                grouped_parts = []
                # slot_group = copy.deepcopy(group_results)
                current_idx = 0

                # 把字數分組當作插槽插入tag
                slot_group = copy.deepcopy(group_results)
                for n in range(len(group_results)):
                    this_group = group_results[n]
                    if len(this_group) > 1:
                        for g in range(len(this_group)):
                            slot_group[n][g] = parts[current_idx]
                            current_idx += 1
                    elif len(this_group) == 1:
                        slot_group[n][0] = parts[current_idx]
                        current_idx += 1

                # 逐一檢查插槽，將插槽內tag合併
                for n in range(len(slot_group)):
                    this_group = slot_group[n]
                    this_text = '\n'.join([t.text.strip() for t in this_group]) if len(this_group) > 1 else this_group[
                        0].text.strip() if len(this_group) == 1 else ''
                    if len(this_group) > 1:
                        tag = Tag(soup, name="div")
                        tag.insert(0, this_text)
                        grouped_parts.append(tag)

                    elif len(this_group) == 1:
                        grouped_parts.append(slot_group[n][0])
                parts_text = [tag2markdown(p, i) for i, p in enumerate(grouped_parts)]

                cxt.memory.bulk_update(url, text_frags)
                tables = htmltable2markdown(soup.prettify(formatter=None))
                _tables = soup.find_all("table")
                for idx in range(len(_tables)):
                    t = _tables[idx]
                    tag = Tag(soup, name="div")

                    text = NavigableString("@placeholder-table-{0}".format(idx))
                    tag.insert(0, text)
                    t.replaceWith(tag)

                h = HTML2Text(baseurl=url)
                resulttext = h.handle(soup.prettify(formatter=None))

                for idx in range(len(tables)):
                    t = tables[idx]
                    resulttext = resulttext.replace("@placeholder-table-{0}".format(idx), t)

                # content = '\n'.join(['\n'.join(list(div.stripped_strings)) for div in divs])
                text_len = len(resulttext)
                return resulttext, title, 200
            except:
                PrintException()
        else:
            return None, '', 200
    return None, '', 400


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
