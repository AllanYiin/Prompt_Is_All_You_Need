import json
import time
import regex
import copy
import random
from collections import OrderedDict
from prompt4all import context
from prompt4all.common import *
from prompt4all.context import *
from prompt4all.utils import regex_utils, web_utils
from prompt4all.tools.database_tools import save_knowledge_base
from prompt4all.utils.tokens_utils import estimate_used_tokens
from urllib.parse import urlencode, unquote
import pandas as pd
import threading
import time
import struct
from openai import OpenAI
import gradio as gr
import uuid
import requests
from datetime import datetime

from io import StringIO, BytesIO
import urllib

client = OpenAI()
client._custom_headers['Accept-Language'] = 'zh-TW'
cxt = context._context()

sem = threading.Semaphore(3)


def search_rag(ur, top_k=5, min_similarity=0.88, it=None):
    return_results = {}
    try:
        query_results = cxt.memory.lookup(ur, top_k, min_similarity)
        cxt.citations.append('![web](../images/rag.png) [{0}]'.format('powered by RAG', ''))
        for k, v in query_results.items():
            cxt.citations.append('![web](../images/web.png) [{0}]({1})'.format(
                v['text'][:15] + '...' if len(v['text']) > 15 else v['text'], v['source']))

        if it in ['knowledge', 'research']:
            return_results = {
                "prompt": "以下為RAG機制所提取出來符合查詢網頁動機之相關內容，請先讀取內容後請仔細思考並將所有內容整合，應該盡可能**保留細節**，包括需要引述出處，以滿足使用者所提出之需求。以markdown格式來書寫，輸出包括標題階層(開頭要有個一級標題)、內容的詳細說明"}
        elif it in ['table']:
            return_results = {
                "prompt": "以下為RAG機制從網頁中取得之表格或數據，請先讀取內容後，請仔細思考後，參考其內容來回答使用者，請盡可能保留與使用者需求相符之內容，包括需要引述出處，不要過度簡化或是刪減。輸出格式建議為markdown格式的表格再加上說明文字"}
        else:
            return_results = {
                "prompt": "以下為RAG機制所提取出來符合查詢網頁動機之相關內容，請先讀取內容後請仔細思考並將所有內容整合，應該盡可能**保留細節**，包括需要引述出處，以滿足使用者所提出之需求。以markdown格式來書寫，輸出包括標題階層(開頭要有個一級標題)、內容的詳細說明"}

        return_results['search_results'] = query_results
        print('RAG', 'Q: {0}/n'.format(ur), orange_color(json.dumps(return_results, ensure_ascii=False)), flush=True)
        return json.dumps(return_results, ensure_ascii=False)
    except:
        PrintException()
        return json.dumps(return_results, ensure_ascii=False)


def webpage_reader(link: str, ur: str, l: str, it: str, lp: bool = False, rt: bool = False, lv=0, memo=None):
    """
    Args:
        memo:
        link: The URL to search, if not provided by the user.  'https://www.google.com/search?keyword1+keyword2' is a good choice.
        ur:a clear statement of the user's request, can be used as a search query and may include search operators..
        l: the language used by the user in the request, according to the ISO 639-1 standard. For Chinese, use zh-CN for Simplified Chinese and zh-TW for Traditional Chinese.
        it: Information extraction types: research (data driven),knowledge(descriptive),news, table, profile, commodity, prices.....
        lp: Whether the link is directly provided by the user
        rt: If the last request doesn't meet user's need, set this to true when trying to retry another request
        lv: The search depth. Defaults to 0.

    Returns:
        A string containing the results retrieved from the webpage.
    """
    results = ""
    returnData = OrderedDict()
    new_results = ''

    # def process_browse(_url, _title, returnData):
    #     new_results, title, status_code = web_utils.search_web(_url)
    #     if status_code != 200:
    #         new_results = webpage_reader(link=_url, ur=ur, l=l, it=it, lp=False, rt=True, lv=lv + 1, memo=_title)
    #
    #     # if new_results and len(new_results) > 0:
    #     #     part_id = uuid.uuid4()
    #     #     save_knowledge_base(part_id=part_id, text_content=title, parent_id=None, ordinal=None, is_rewrite=0,
    #     #                         source_type=1,
    #     #                         url=_url,
    #     #                         raw=new_results)
    #     #     if len(new_results) > 200:
    #     #         parts = web_utils.cleasing_web_text(new_results)
    #     #         for r in range(len(parts)):
    #     #             this_text = parts[r]
    #     #             save_knowledge_base(part_id=uuid.uuid4(), text_content=this_text, parent_id=part_id,
    #     #                                 ordinal=r + 1, is_rewrite=0, source_type=1,
    #     #                                 url=_url, raw=None)
    #     returnData[_url] = new_results
    def process_browse(_url, returnData):
        new_results, title, status_code = web_utils.search_web(_url)
        if new_results is None or status_code != 200:
            new_results = "{}"
        returnData[_url] = new_results

    if link.endswith('.pdf'):
        try:
            pdf_doc_text = get_pdf_content(link)
            cxt.citations.append('![pdf](../images/pdf.png) [{0}]({1})'.format(link.split('/')[-1], link))
            return pdf_doc_text
        except:
            PrintException()

    header = {
        "Content-Type": "application/json",
        "WebPilot-Friend-UID": str(uuid.uuid4()),
    }
    if 'www.statista.com' in link and lv == 0:
        link = 'https://www.google.com/search?' + urlencode({"q": ur.replace(' ', '+')}).replace('%2B', '+')
    search_lists = []
    if ur and (link is None or link == 'none' or link == '' or len(
            [s for s in link.split('/') if len(s) > 0]) == 2) and not ur.startswith('site:'):
        try:

            if len([s for s in link.split('/') if len(s) > 0]) == 2:
                url = 'https://www.google.com/search?' + urlencode('q=site:{0}+{1}'.format(link, ur))
                search_lists = web_utils.search_google(url)

            search_lists = better_search(ur)
            start_time = time.time()
            if len(search_lists) == 0:
                print(magenta_color('search_lists is empty'), link, flush=True)
            threads = []
            for i in range(len(search_lists)):
                item = search_lists[i]
                if 'url' in item and item['url'] not in list(cxt.memory._cache.keys()):
                    _url = item['url']
                    _title = item['title']
                    threads.append(threading.Thread(target=process_browse, args=(_url, returnData)))

            for i in range(len(threads)):
                threads[i].start()
                threads[i].join()
                if (i > 0 and i % 5 == 0):
                    time.sleep(2)
            while len(returnData) < len(threads):
                if time.time() - start_time > 90:
                    break
                time.sleep(1)
            return search_rag(ur, 5, 0.88, it)
            # for k, v in returnData.items():
            #     new_results = new_results + '\n\n' + k + '\n\n' + v
            # return new_results
        except Exception as e:
            PrintException()
            gr.Error(e)
    elif ur.startswith('site:') or 'https://www.google.com/search' in link or 'https://www.bing.com/search' in link:
        try:
            if ur.startswith('site:'):
                link = 'https://www.google.com/search/?q=' + ur
                search_lists, _ = web_utils.search_google(link)
            elif 'https://www.bing.com/search' in link:
                search_lists, _ = web_utils.search_bing(link)
            else:
                query = unquote(link.split('?')[1].replace('q=', '')).split('+')
                if len(query) == 1 and len(query[0]) >= 7:
                    search_lists = better_search(query[0])
                else:
                    search_lists, _ = web_utils.search_google(link)
            if len(search_lists) == 0:
                print(magenta_color('search_lists is empty'), link, flush=True)
                search_lists = better_search(query[0])
            threads = []
            start_time = time.time()
            for i in range(len(search_lists)):
                item = search_lists[i]
                if 'url' in item and item['url'] not in list(cxt.memory._cache.keys()):
                    _url = item['url']
                    _title = item['title']
                    threads.append(threading.Thread(target=process_browse, args=(_url, returnData)))

            for i in range(len(threads)):
                try:
                    threads[i].start()
                except:
                    PrintException()
                    try:
                        threads[i].start()
                    except:
                        PrintException()

            for i in range(len(threads)):
                threads[i].join()

            while len(returnData) < len(threads):
                # print(len(returnData), len(threads), flush=True)
                if time.time() - start_time > 90:
                    break
                time.sleep(1)
            return search_rag(ur, 5, 0.88, it)
            # for k, v in returnData.items():
            #     new_results = new_results + '\n\n' + k + '\n\n' + v
            # return new_results

        except:
            PrintException()


    else:
        try:
            if link not in cxt.memory._cache:
                new_results, title, status_code = web_utils.search_web(link)
                if status_code != 200 or new_results is None or len(new_results) < 200:
                    part_id = uuid.uuid4()
                    data = {
                        "link": link,
                        "ur": ur,
                        "l": l,
                        "lp": lp,
                        "rt": rt
                    }

                    endpoint = "https://webreader.webpilotai.com/api/visit-web"
                    resp = requests.post(endpoint, headers=header, data=json.dumps(data))
                    if resp.status_code != 200:
                        print('ERROR', resp.status_code, link)

                    resp = eval(resp.text)
                    title = memo if memo else resp['meta']['og:title'] if 'meta' in resp else None

                    if 'content' not in resp:
                        new_results = 'No content'
                    else:
                        new_results = resp['content']
                    return new_results

                if it == 'news':
                    title = title if title else ur
                    return get_news_list(title, new_results)
                elif it == 'table':
                    return get_table_list(new_results, link)
                elif it in ['knowledge', 'research']:
                    cxt.citations.append('![web](../images/web.png) [{0}]({1})'.format(title, link))
                return search_rag(ur, 5, 0.88, it)
            else:
                _dict = {link: '\n'.join([tf.text for tf in cxt.memory._cache[link]])}
                return json.dumps(_dict, ensure_ascii=False)
        except:
            PrintException()

    return new_results


def better_search(query_intent, keywords_cnt=3):
    _prompt = """
    你是一個專業的網路搜索達人，你能夠根據使用者提供的搜索意圖中的關鍵概念，根據以下原則轉化為{0}組實際查詢的關鍵字組合，輸出格式為markdown的有序清單，關鍵字間請用加號分隔)
     - 請基於第一性原則，思考這樣的使用者意圖會在甚麼樣的網頁中出現
    -  從使用者意圖以及剛才思考的網頁特性，思考出最重要關鍵字再加上1~2個次要的關鍵詞切忌整個句子丟進去。
    - 直接輸出，無須解釋
    """.format(keywords_cnt, query_intent)

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {'role': 'system', 'content': _prompt},
            {'role': 'user', 'content': "用戶意圖:{0}".format(query_intent)}
        ],
        temperature=0.8,
        n=1,
        stream=False,

    )
    if response and response.choices and response.choices[0].message:
        response_message = response.choices[0].message.content
        results_list = [t.replace(regex_utils.extract_numbered_list_member(t), '').strip() for t in
                        response_message.split('\n') if len(t) > 0]
        print('better search:{0}'.format(results_list))

        all_search_list = None
        for item in results_list:
            query = urlencode({"q": item.replace(' ', '+')}).replace('%2B', '+')
            search_url = f"https://www.google.com/search?{query}"
            google_search_lists, _ = web_utils.search_google(search_url)
            # search_url_bing = f"https://www.bing.com/search?{query}"
            print(item, google_search_lists)
            if all_search_list is None:
                all_search_list = google_search_lists
            else:
                all_search_list.extend(
                    google_search_lists if len(google_search_lists) <= keywords_cnt else
                    google_search_lists[:3])
            # search_list.extend(search_url_bing)
        url_deup = {}
        webpage_list = []
        for item in all_search_list:
            if item['url'] not in url_deup:
                url_deup[item['url']] = 1
                webpage_list.append(item)
        all_search_list = webpage_list
        print('better search results:', green_color(str(all_search_list)), flush=True)
        return all_search_list
    else:
        query = urlencode({"q": query_intent.replace(' ', '+')}).replace('%2B', '+')
        search_url = f"https://www.google.com/search?{query}"
        google_search_lists, _ = web_utils.search_google(search_url)
        print('google search results:', green_color(str(google_search_lists)), flush=True)
        return google_search_lists


def get_search_list(ur: str, content: str):
    """
    Args:
        ur (str): The search query.
        content (str): The original content to be processed.

    Returns:
        str: The processed content.

    """

    _json_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "webpage_list": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "網頁title"
                        },
                        "url": {
                            "type": "string",
                            "format": "url",
                            "description": "網頁url"
                        },
                        "summary": {
                            "type": "string",
                            "description": "網頁內容摘要"
                        },
                    },
                    "required": ["title", "url", "summary", ]
                }
            }
        },
        "required": ["webpage_list"]
    }

    _prompt = '請將以下內容中與搜索意圖「{0}」相關搜索內容保留，然後依照{1} schema來進行整理為列表 "\n"""\n{2}\n"""\n'.format(
        ur, _json_schema, content)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {'role': 'system', 'content': '#zh-TW'},
            {'role': 'user', 'content': _prompt}
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
        n=1,
        stream=False,

    )
    if response and response.choices and response.choices[0].message:
        response_message = response.choices[0].message
        print(response_message.content)
        return response_message.content


def get_news_list(title: str, content: str):
    """
    Args:
        title (str): The title of the news.
        content (str): The content related to the news.

    Returns:
        str: The processed news list.

    """
    cxt.status_word = '整理「{0}｣搜尋結果清單中...'.format(title)
    _json_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "news_list": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "新聞標題"
                        },
                        "url": {
                            "type": "string",
                            "format": "url",
                            "description": "新聞url"
                        },
                        "summary": {
                            "type": "string",
                            "description": "新聞摘要"
                        },
                        "media": {
                            "type": "string",
                            "description": "發布媒體"
                        },
                        "date": {
                            "type": "string",
                            "format": "date",
                            "description": "發布日期"
                        }
                    },
                    "required": ["title", "summary", "media", "date"]
                }
            }
        },
        "required": ["news_list"]
    }

    _prompt = '請將以下內容中與title「{0}」相關之新聞內容保留，然後依照{1} schema來進行整理為新聞列表，日期若是相對日期，請使用今日日期({2})換算回絕對日期，若無案例則回傳空字典 "\n"""\n{3}\n"""\n'.format(
        title, _json_schema, datetime.now(), content)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {'role': 'system', 'content': '#zh-TW'},
            {'role': 'user', 'content': _prompt}
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
        n=1,
        stream=False,

    )
    if response and response.choices and response.choices[0].message:
        response_message = response.choices[0].message
        print(response_message.content)
        return response_message.content


def get_knowledge_list(ur, content: str, l: str):
    cxt.status_word = '整理「{0}｣知識點中...'.format(ur)
    _prompt = '請將以下內容中與「{0}」相關之知識點、數據、事實與觀點予以保留，並去除冗餘、無意義之部分，並改寫為{1}語系，並且適時地餘內容中插入"\n""\n以表示段落，相近主題的內容應該要置於同一段落中，每個段落字數約在100~1000之間，並請確認改寫後結果與原文相符 "\n"""\n{2}\n"""\n'.format(
        ur, l, content)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {'role': 'user', 'content': _prompt}
        ],
        temperature=0.3,
        n=1,
        stream=False,

    )
    if response and response.choices and response.choices[0].message:
        response_message = response.choices[0].message
        print('knowlege', response_message.content)
        return response_message.content
    return ''


def get_table_list(content: str, url):
    """

    Args:
        url:
        *args:
        **kwargs:

    Returns:

    Examples:
        >>> url="https://c.8891.com.tw/sales/top?fid=2023-12&tid=10000"
        >>> get_table_list(web_utils.search_web(url)[0],url)

    """
    tables = regex.findall(regex_utils.md_table_pattern, content)
    if len(tables) > 0:
        print(yellow_color('  \n\n'.join(['\n'.join(list(t)) for t in tables])))
        return '  \n\n'.join(['\n'.join(list(t)) for t in tables])
    else:

        _prompt = '請讀取以下文字內容，請思考一下該如何整理為markdown表格的形式最能夠讓人清楚這些資料想要表達的意義，要能吸引人的且確保數值正確性，重點在於清楚呈現數值與類別之間的關係，輸出形式為code block，無須解釋，直接輸出 "\n"""\n{0}\n"""\n'.format(
            content)
        model_name = 'gpt-3.5-turbo-1106'
        num_tokens = estimate_used_tokens(_prompt, model_name=model_name)
        if num_tokens > model_info[model_name]["max_token"] // 2:
            model_name = 'gpt-4-1106-preview'

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {'role': 'system', 'content': '#zh-TW'},
                {'role': 'user', 'content': _prompt}
            ],
            temperature=0.3,
            n=1,
            stream=False,

        )

        tables_md = regex_utils.extract_code(response.choices[0].message.content)
        if tables_md is None or len(tables_md) == 0:
            tables = regex.findall(regex_utils.md_table_pattern, response.choices[0].message.content)
            tables_md = '  \n\n'.join(['\n'.join(list(t)) for t in tables])
        if len(tables_md) > 10:
            print(yellow_color(tables_md))
            return tables_md
        else:
            return context


def get_pdf_content(pdf_url):
    from prompt4all.utils import pdf_utils
    _pdf = pdf_utils.PDF(pdf_url)
    _pdf.parsing_save()
    return _pdf.doc_text
