import json
import time
from prompt4all import context
from prompt4all.context import *
from prompt4all.utils import regex_utils, web_utils
import pandas as pd
import struct
from openai import OpenAI
import gradio as gr
import uuid
import requests
from datetime import datetime

client = OpenAI()
client._custom_headers['Accept-Language'] = 'zh-TW'
cxt = context._context()


def webpage_reader(link: str, ur: str, l: str, lp: bool, it: str, rt: bool = False):
    header = {
        "Content-Type": "application/json",
        "WebPilot-Friend-UID": str(uuid.uuid4()),
    }
    data = {
        "link": link,
        "ur": ur,
        "l": l,
        "lp": lp,
        "rt": rt
    }

    if it == 'table':
        results = web_utils.search_web(link)
        results = get_table_list(results)
        return results

    endpoint = "https://webreader.webpilotai.com/api/visit-web"
    resp = requests.post(endpoint, headers=header, data=json.dumps(data))
    resp = eval(resp.text)

    title = resp['meta']['og:title']
    results = None
    if it == 'news':
        results = get_news_list(title, resp['content'])

    else:
        _prompt = '請將以下網頁內容僅保留與title「{0}」相關之部分，如果是新聞請將標題、內容、發布媒體、日期等信息視為一個群組，在群組間加入"<br/>"。"\n"""\n{1}\n"""\n'.format(
            title, resp['content'])

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
        results = response_message.content
    return results


def get_news_list(title: str, content: str):
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
    response_message = response.choices[0].message
    print(response_message.content)
    return response_message.content


def get_table_list(content: str):
    _json_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "tables": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "The title of the table"
                        },
                        "description": {
                            "type": "string",
                            "description": "A brief description of the table"
                        },
                        "source": {
                            "type": "string",
                            "format": "uri",
                            "description": "The URL of the web page where the table is extracted from"
                        },
                        "table": {
                            "type": "string",
                            "description": "The table formatted as markdown"
                        }
                    },
                    "required": ["title", "table"]
                }
            }
        },
        "required": ["tables"]
    }

    _prompt = '請將以下內容中表格形式的數據，然後依照{0} schema來進行整理為表格列表，若無案例則回傳空字典 "\n"""\n{1}\n"""\n'.format(
        _json_schema, content)
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
    response_message = response.choices[0].message
    print(response_message.content)
    return response_message.content
