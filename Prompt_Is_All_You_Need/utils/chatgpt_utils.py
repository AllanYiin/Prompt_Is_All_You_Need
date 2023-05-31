# -*- coding: utf-8-sig -*-
import base64
import io
import regex
import requests
from pypdf import PdfReader, PdfWriter
from PIL import Image
from bs4 import BeautifulSoup

__all__ = ['process_chat','process_url','process_context','build_message','regular_txt_to_markdown']

def regular_txt_to_markdown(text):
    text = text.replace('\n', '\n\n')
    text = text.replace('\n\n\n', '\n\n')
    text = text.replace('\n\n\n', '\n\n')
    return text

def process_chat(conversation_dict: dict):
    if conversation_dict['role'] == 'user':
        return '😲:\n' + regular_txt_to_markdown(conversation_dict['content']) + "\n"
    elif conversation_dict['role'] == 'assistant':
        return '🤖:\n' + regular_txt_to_markdown(conversation_dict['content'])+ "\n"
    elif conversation_dict['role'] == 'system':
        return '💡:\n' + conversation_dict['content'] + "\n"


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


MAX_SECTION_LENGTH = 1000
SENTENCE_SEARCH_LIMIT = 100
SECTION_OVERLAP = 100

def get_document_text(filename):
    offset = 0
    page_map = []

    reader = PdfReader(filename)
    pages = reader.pages
    for page_num, p in enumerate(pages):
        page_text = p.extract_text()
        page_map.append((page_num, offset, page_text))
        offset += len(page_text)
    return page_map


def split_text(page_map):
    SENTENCE_ENDINGS = [".", "!", "?"]
    WORDS_BREAKS = [",", ";", ":", " ", "(", ")", "[", "]", "{", "}", "\t", "\n"]
    #if args.verbose: print(f"Splitting '{filename}' into sections")

    def find_page(offset):
        l = len(page_map)
        for i in range(l - 1):
            if offset >= page_map[i][1] and offset < page_map[i + 1][1]:
                return i
        return l - 1

    all_text = "".join(p[2] for p in page_map)
    length = len(all_text)
    start = 0
    end = length
    while start + SECTION_OVERLAP < length:
        last_word = -1
        end = start + MAX_SECTION_LENGTH

        if end > length:
            end = length
        else:
            # Try to find the end of the sentence
            while end < length and (end - start - MAX_SECTION_LENGTH) < SENTENCE_SEARCH_LIMIT and all_text[
                end] not in SENTENCE_ENDINGS:
                if all_text[end] in WORDS_BREAKS:
                    last_word = end
                end += 1
            if end < length and all_text[end] not in SENTENCE_ENDINGS and last_word > 0:
                end = last_word  # Fall back to at least keeping a whole word
        if end < length:
            end += 1

        # Try to find the start of the sentence or at least a whole word boundary
        last_word = -1
        while start > 0 and start > end - MAX_SECTION_LENGTH - 2 * SENTENCE_SEARCH_LIMIT and all_text[
            start] not in SENTENCE_ENDINGS:
            if all_text[start] in WORDS_BREAKS:
                last_word = start
            start -= 1
        if all_text[start] not in SENTENCE_ENDINGS and last_word > 0:
            start = last_word
        if start > 0:
            start += 1

        section_text = all_text[start:end]
        yield (section_text, find_page(start))

        last_table_start = section_text.rfind("<table")
        if (last_table_start > 2 * SENTENCE_SEARCH_LIMIT and last_table_start > section_text.rfind("</table")):
            # If the section ends with an unclosed table, we need to start the next section with the table.
            # If table starts inside SENTENCE_SEARCH_LIMIT, we ignore it, as that will cause an infinite loop for tables longer than MAX_SECTION_LENGTH
            # If last table starts inside SECTION_OVERLAP, keep overlapping
            if args.verbose: print(
                f"Section ends with unclosed table, starting next section with the table at page {find_page(start)} offset {start} table start {last_table_start}")
            start = min(end - SECTION_OVERLAP, start + last_table_start)
        else:
            start = end - SECTION_OVERLAP

    if start + SECTION_OVERLAP < end:
        yield (all_text[start:end], find_page(start))

def blob_name_from_file_page(filename, page = 0):
    if os.path.splitext(filename)[1].lower() == ".pdf":
        return os.path.splitext(os.path.basename(filename))[0] + f"-{page}" + ".pdf"
    else:
        return os.path.basename(filename)

def create_sections(filename, page_map):
    for i, (section, pagenum) in enumerate(split_text(page_map)):
        yield {
            "id": re.sub("[^0-9a-zA-Z_-]", "_", f"{filename}-{i}"),
            "content": section,
            "sourcepage": blob_name_from_file_page(filename, pagenum),
            "sourcefile": filename
        }