# -*- coding: utf-8-sig -*-
import time
import gradio as gr
import os
import json
import requests
import openai
import copy
import regex
import requests
from bs4 import BeautifulSoup
import PyPDF2
from PIL import Image
import io
import base64

__all__ = ['process_chat','process_url','process_context','build_message']
def process_chat(conversation_dict: dict):
    if conversation_dict['role'] == 'user':
        return '😲:\n' + conversation_dict['content'] + "\n"
    elif conversation_dict['role'] == 'assistant':
        return '🤖:\n' + conversation_dict['content'] + "\n"

def extract_urls_text(text):
    url_pattern = regex.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    urls = regex.findall(url_pattern, text)
    if len(urls)==0:
        return text
    else:
        text_type,url_text=process_url(urls[0])
        all_text=text_type+'內容: '+url_text+'\n'+text.replace(urls[0],text_type+'內容')

        return all_text


def process_url(url):
    text_type = 'None'

    response = requests.get(url)

    content_type = response.headers.get('Content-Type', '')

    if 'text/html' in content_type:
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        text_type = '網頁文字'
    elif 'application/pdf' in content_type:
        with io.BytesIO(response.content) as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ''

            for page_num in range(len(pdf_reader.pages)):
                _pdf_text = pdf_reader.pages[page_num].extract_text()
                print(_pdf_text)
                text += _pdf_text

            text = text.replace('\x03', '').replace('\x02', '').replace('-\n', '').replace(' \n', ' ')
            text = regex.sub(r'(?<=[a-z\u4e00-\u9fff])\n(?=[a-z\u4e00-\u9fff])', ' ', text)
            text_type = 'pdf'
    elif 'image/' in content_type:
        image = Image.open(io.BytesIO(response.content))
        buffered = io.BytesIO()
        image.save(buffered, format=image.format)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        text = f'data:{content_type};base64,{img_base64}'
        text_type = '圖檔'
    else:
        text = ''
    return text_type, text


def build_message(role,content):
    return {"role": str(role), "content": str(content)}


def process_context(prompt, context_type,history: list):
    message_context = [build_message(message['role'],message['summary'] if message['role'] == 'assistant' and 'summary' in message else message['content']) for message in history]
    message_context.append({"role": "user", "content": extract_urls_text(prompt)})
    return message_context