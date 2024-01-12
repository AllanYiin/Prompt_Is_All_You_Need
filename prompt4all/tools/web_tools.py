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

    def process_browse(_url, _title, returnData):
        new_results, title, status_code = web_utils.search_web(_url)
        if status_code != 200:
            new_results = webpage_reader(link=_url, ur=ur, l=l, it=it, lp=False, rt=True, lv=lv + 1, memo=_title)
        else:
            if new_results and len(new_results) > 0:
                part_id = uuid.uuid4()
                save_knowledge_base(part_id=part_id, text_content=title, parent_id=None, ordinal=None, is_rewrite=0,
                                    source_type=1,
                                    url=_url,
                                    raw=new_results)
                if len(new_results) > 200:
                    parts = web_utils.cleasing_web_text(new_results)
                    for r in range(len(parts)):
                        this_text = parts[r]
                        save_knowledge_base(part_id=uuid.uuid4(), text_content=this_text, parent_id=part_id,
                                            ordinal=r + 1, is_rewrite=0, source_type=1,
                                            url=_url, raw=None)
        returnData[_url] = new_results

    if link.endswith('.pdf'):
        pdf_doc_text = get_pdf_content(link)
        cxt.citations.append('![pdf](../images/pdf.png) [{0}]({1})'.format(link.split('/')[-1], link))
        return pdf_doc_text

    header = {
        "Content-Type": "application/json",
        "WebPilot-Friend-UID": str(uuid.uuid4()),
    }
    if 'www.statista.com' in link and lv == 0:
        link = 'https://www.google.com/search?' + urlencode({"q": ur.replace(' ', '+')}).replace('%2B', '+')

    if ur and (link is None or link == 'none' or link == '') and not ur.startswith('site:'):
        search_lists = web_utils.search_google(ur)
        threads = []
        for i in range(len(search_lists['webpage_list'])):
            item = search_lists['webpage_list'][i]
            if 'url' in item:
                _url = item['url']
                _title = item['title']
                threads.append(threading.Thread(target=process_browse, args=(_url, _title, returnData)))

        for i in range(len(threads)):
            threads[i].start()
            threads[i].join()
            if (i > 0 and i % 5 == 0):
                time.sleep(2)
        while len(returnData) < len(threads):
            time.sleep(1)
        for k, v in returnData.items():
            results = results + '\n\n' + k + '\n\n' + v
    elif ur.startswith('site:') or 'https://www.google.com/search' in link:
        if ur.startswith('site:'):
            link = 'https://www.google.com/search/?q=' + ur
            # + urlencode(  {"q": ur.replace('\n', '+').replace(' ', '+')}).replace('%2B', '+'))
        search_lists, _ = web_utils.search_google(link)
        return json.dumps(search_lists, ensure_ascii=False)
        # threads = []
        # for i in range(len(search_lists['webpage_list'])):
        #     item = search_lists['webpage_list'][i]
        #     if 'url' in item:
        #         _url = item['url']
        #         _title = item['title']
        #         threads.append(threading.Thread(target=process_browse, args=(_url, _title, returnData)))
        #         threads[i].start()
        # for i in range(len(threads)):
        #     threads[i].join()
        # while len([k for k, v in returnData.items()]) < len(threads):
        #     time.sleep(1)
        # for k, v in returnData.items():
        #     if v and len(v) > 0:
        #         results = results + '\n\n' + k + '\n\n' + v
    elif 'https://www.bing.com/search' in link:
        search_lists, _ = web_utils.search_bing(link)
        return json.dumps(search_lists, ensure_ascii=False)
    else:

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

            print(data, 'it:' + it)
            endpoint = "https://webreader.webpilotai.com/api/visit-web"
            resp = requests.post(endpoint, headers=header, data=json.dumps(data))
            if resp.status_code != 200:
                print('ERROR', resp.status_code, link)
                return ''
            resp = eval(resp.text)
            title = memo if memo else resp['meta']['og:title'] if 'meta' in resp else None

            if 'content' not in resp:
                new_results = 'No content'
            else:
                new_results = resp['content']

        if it == 'news':
            title = title if title else ur
            return get_news_list(title, new_results)
        elif it in ['knowledge', 'research']:
            cxt.citations.append('![web](../images/web.png) [{0}]({1})'.format(title, link))
            pass
        elif it == 'table':

            return get_table_list(new_results, link)
        else:
            _prompt = '請將以下網頁內容僅保留與「{0}」相關之部分。"\n"""\n{1}\n"""\n'.format(ur, new_results)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {'role': 'system', 'content': '#zh-TW'},
                    {'role': 'user', 'content': _prompt}
                ],
                temperature=0.3,
                n=1,
                stream=False,

            )

            response_message = response.choices[0].message
            new_results = response_message.content
        if new_results and len(new_results) > 0:
            part_id = uuid.uuid4()
            save_knowledge_base(part_id=part_id, text_content=title, parent_id=None, ordinal=None, is_rewrite=0,
                                source_type=1,
                                url=link,
                                raw=new_results)

        if len(new_results) > 200:
            parts = web_utils.cleasing_web_text(new_results)
            for r in range(len(parts)):
                this_text = parts[r]
                save_knowledge_base(part_id=uuid.uuid4(), text_content=this_text, parent_id=part_id,
                                    ordinal=r + 1, is_rewrite=0, source_type=1,
                                    url=link, raw=None)

    # if (lv == 0) and it in ['knowledge', 'research']:
    #     return r'以下是透過網路搜索所獲取的情報，請盡量詳實完整的輸出給使用者(你若在這階段缺漏太多篇幅過短，會對我職業生涯造成**重大傷害**!!!)\n\n"""\n#搜索到的內容    \n\n  {0}\n\n  """'.format(
    #         results)
    return new_results


def better_search(query_intent, keywords_cnt=3):
    _prompt = """
    你是一個專業的網路搜索達人，你能夠根據使用者提供的搜索意圖中的關鍵概念，根據以下原則轉化為{0}組實際查詢的關鍵字組合(以markdown Ordered Lists形式呈現，關鍵字間請用加號分隔)
     - **關鍵概念定義**：關鍵概念的細節定義釐清，若搜索意圖涉及數字，務必釐清數字所使用的單位(與金額有關皆須要確認幣別)
   - **收集背景知識**：若是數值的預估，則包括歷史與目前數值，以及各家研究機構對於未來的預測，需要仔細確認各個數值的定義與單位，背景知識越多元越好。
    - **重大影響的具體事件**：近期對關鍵概念會有重大影響的具體事件，不是概念性的，通常是新法律的頒布或修訂、經濟狀態的急速變化 、地域政治的影響。
    直接輸出，無須說明。
    使用者搜索意圖:{1}
    """.format(keywords_cnt, query_intent)

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {'role': 'system', 'content': '#zh-TW'},
            {'role': 'user', 'content': _prompt}
        ],
        temperature=0.3,
        n=1,
        stream=False,

    )
    if response and response.choices and response.choices[0].message:
        response_message = response.choices[0].message.content

        results_list = [t.replace(regex_utils.extract_numbered_list_member(t), '').strip() for t in
                        response_message.split('\n') if len(t) > 0]

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
                all_search_list['webpage_list'].extend(google_search_lists['webpage_list'])
            # search_list.extend(search_url_bing)
        url_deup = {}
        webpage_list = []
        for item in all_search_list['webpage_list']:
            if item['url'] not in url_deup:
                url_deup[item['url']] = 1
                webpage_list.append(item)
        all_search_list['webpage_list'] = webpage_list

        return all_search_list
    else:
        query = urlencode({"q": query_intent.replace(' ', '+')}).replace('%2B', '+')
        search_url = f"https://www.google.com/search?{query}"
        google_search_lists, _ = web_utils.search_google(search_url)
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
    tables = regex.findall(regex_utils.md_table_pattern, content)

    # _json_schema = {
    #     "$schema": "http://json-schema.org/draft-07/schema#",
    #     "type": "object",
    #     "properties": {
    #         "tables": {
    #             "type": "array",
    #             "items": {
    #                 "type": "object",
    #                 "properties": {
    #                     "title": {
    #                         "type": "string",
    #                         "description": "The title of the table"
    #                     },
    #                     "description": {
    #                         "type": "string",
    #                         "description": "A brief description of the table"
    #                     },
    #                     "source": {
    #                         "type": "string",
    #                         "format": "url",
    #                         "description": "The URL of the web page where the table is extracted from"
    #                     },
    #                     "table": {
    #                         "type": "string",
    #                         "description": "The table formatted as markdown"
    #                     }
    #                 },
    #                 "required": ["title", "table", "url"]
    #             }
    #         }
    #     },
    #     "required": ["tables"]
    # }
    #
    # _prompt = '請將以下source為{0}內容中表格形式的數據，然後依照{1} schema來進行整理為表格列表，若無案例則回傳空字典 "\n"""\n{2}\n"""\n'.format(
    #     url, _json_schema, content)
    # response = client.chat.completions.create(
    #     model="gpt-3.5-turbo-1106",
    #     messages=[
    #         {'role': 'system', 'content': '#zh-TW'},
    #         {'role': 'user', 'content': _prompt}
    #     ],
    #     temperature=0.3,
    #     response_format={"type": "json_object"},
    #     n=1,
    #     stream=False,
    #
    # )
    # response_message = response.choices[0].message
    # print(response_message.content)
    # for item in json.loads(response_message.content)['tables']:
    #     cxt.citations.append('![table](../images/table.png) [{0}]({1})'.format(item['title'], url))
    print(yellow_color('  \n\n'.join(['\n'.join(list(t)) for t in tables])))
    return '  \n\n'.join(['\n'.join(list(t)) for t in tables])


def get_pdf_content(pdf_url):
    from prompt4all.utils import pdf_utils
    _pdf = pdf_utils.PDF(pdf_url)
    _pdf.parsing_save()
    return _pdf.doc_text
