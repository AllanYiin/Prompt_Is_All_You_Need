# -*- coding: utf-8-sig -*-
import base64
import io
import regex
import requests
from PIL import Image
from bs4 import BeautifulSoup

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
        import PyPDF2
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
    """
    Build a chat message with the given role and content.

    Args:
    role (str): The role of the message sender, e.g., "system", "user", or "assistant".
    content (str): The content of the message.

    Returns:
    dict: A dictionary containing the role and content of the message.
    """
    return {"role": str(role), "content": str(content)}


def process_context(prompt, context_type,history: list):
    message_context = [build_message(message['role'],message['summary'] if message['role'] == 'assistant' and 'summary' in message else message['content']) for message in history]
    message_context.append({"role": "user", "content": extract_urls_text(prompt)})
    return message_context


def parse_codeblock(text):
    if "```" in text:
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if "```" in line:
                if line != "```":
                    lines[i] = f'<pre><code class="{lines[i][3:]}">'
                else:
                    lines[i] = '</code></pre>'
            else:
                if i > 0:
                    lines[i] = "<br/>" + line.replace("<", "&lt;").replace(">", "&gt;")
        return "".join(lines)
    else:
        return text