import uuid
from prompt4all.utils import regex_utils
from prompt4all import context
from openai import OpenAI
import requests
import base64
import zlib
from PIL import Image
import io
import regex
import json
import time
import subprocess
import tempfile
import uuid
import os
from sys import stderr, stdout

client = OpenAI()
client._custom_headers['Accept-Language'] = 'zh-TW'
cxt = context._context()


def js_string_to_byte(data: str) -> bytes:
    """Convert a string to bytes using ascii encoding."""
    return bytes(data, 'ascii')


def js_bytes_to_string(data: bytes) -> str:
    """Decode bytes to a string using ascii decoding."""
    return data.decode('ascii')


def js_btoa(data: bytes) -> bytes:
    """Encode bytes to base64."""
    return base64.b64encode(data)


def pako_deflate(data: bytes) -> bytes:
    """Compress the given bytes using zlib."""
    compress = zlib.compressobj(9, zlib.DEFLATED, 15, 8, zlib.Z_DEFAULT_STRATEGY)
    compressed_data = compress.compress(data)
    compressed_data += compress.flush()
    return compressed_data


def encode_to_pako(graphMarkdown: str) -> str:
    """Encode the graph markdown to a pako format."""
    jGraph = {
        "code": graphMarkdown,
        "mermaid": {"theme": "default"}
    }
    byteStr = js_string_to_byte(json.dumps(jGraph))
    deflated = pako_deflate(byteStr)
    dEncode = js_btoa(deflated)
    return js_bytes_to_string(dEncode)


def generate_mermaid_diagram(graph):
    jGraph = {
        "code": graph,
        "mermaid": {"theme": "default"}
    }
    graphbytes = json.dumps(jGraph).encode("ascii")
    deflated = pako_deflate(graphbytes)
    base64_bytes = base64.b64encode(deflated)
    base64_string = base64_bytes.decode("utf-8")
    new_url = ''
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36'}
    response = requests.get('https://mermaid.live/edit#pako:' + base64_string, allow_redirects=True, headers=headers)
    time.sleep(2)
    encode_pecko = response.url.split('pako:')[-1]
    return encode_pecko


mermaid_charts = ['flowchart', 'sequenceDiagram', 'classDiagram', 'stateDiagram', 'erDiagram',
                  'journey', 'gantt', 'gitGraph', 'pie', 'mindmap', 'quadrantChart', 'xychart']


def extract_code(text):
    """
    從給定的文本中提取代碼區塊。

    :param text: 包含代碼區塊的字符串。
    :return: 包含所有代碼區塊的列表。
    """

    code_blocks = regex.findall(r'```(.*?)```', text, regex.S)
    if code_blocks:
        return '\n'.join(code_blocks[0].split('\n')[1:])
    else:
        return ''


def generate_diagram(di, dt, ss=None):
    cxt.status_word = "生成{0}圖表中...".format(dt)
    response_content = ''
    if ss:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {'role': 'system', 'content': '#zh-TW 你是一個有多年經驗、熟悉資料視覺化的數據科學家'},
                {'role': 'user',
                 'content': '請協助檢查以下內容是否符合mermaid {0} 語法規範，尤其需要確認標點符號以及特殊符號出現時需要處理逸出字元、過長的文字則可使用<br/>換行，最終結果請以代碼區塊的格式輸出\n"""\n{1}\n"""\n'.format(
                     dt, ss)}
            ],
            temperature=0.3,
            n=1,
            presence_penalty=0,
            stream=False
        )
        response_content = response.choices[0].message.content
    else:
        if dt == "flowchart":
            response = client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {'role': 'system', 'content': '#zh-TW 你是一個有多年經驗、熟悉資料視覺化的數據科學家'},
                    {'role': 'user',
                     'content': '#zh-TW請將以下內容轉換為正確之Mermaid {0} 語法，尤其需要確認標點符號以及特殊符號出現時需要處理逸出字元、過長的文字則可使用<br/>換行，最終結果請以代碼區塊的格式輸出\n"""\n{1}\n"""\n'.format(
                         dt, di)}
                ],
                temperature=0.3,
                n=1,
                presence_penalty=0,
                stream=False
            )
            response_content = response.choices[0].message.content
            print(response_content)
    graph_syntax = extract_code(response_content)
    print(graph_syntax)
    encode_pecko = generate_mermaid_diagram(graph_syntax)
    # print('https://mermaid.ink/img/pako:' + encode_pecko)
    print('https://mermaid.live/view#pako:' + encode_pecko)
    print('https://mermaid.live/edit#pako:' + encode_pecko)
    # img = Image.open(io.BytesIO(requests.get(
    #     'https://mermaid.ink/img/pako:' + encode_pecko).content))
    #
    # filepath = 'generate_images/{0}_{1}.png'.format(dt, uuid.uuid4().node)
    # img.save(filepath)
    return str({"圖表類型": dt, "瀏覽圖表路徑": 'https://mermaid.live/view#pako:' + encode_pecko,
                "編輯圖表路徑": 'https://mermaid.live/edit#pako:' + encode_pecko, "產出圖表語法": graph_syntax})


def exec_mermaid_cli(mermaid_markdown, diagram_type):
    if 'graph TB' in mermaid_markdown:
        mermaid_markdown = mermaid_markdown.replace('graph LR')
    temfolder = tempfile.gettempdir()
    input_path = os.path.join(temfolder, str(uuid.uuid4().node) + '.mmd')
    with open(input_path, "w") as t:
        t.write(mermaid_markdown)
    output_path = './generate_images/{0}_{1}.png'.format(diagram_type, uuid.uuid4().node)

    mm_args = ["mmdc", "-i", input_path, "-o", output_path]
    p = subprocess.Popen(
        mm_args, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = p.communicate(input=None, timeout=60)
    if p.returncode == 0:
        return output_path
    else:

        return "[output]\n%s\n[error]\n%s" % (output, error) + mermaid_markdown


def get_diagram_guidelines(dt):
    if dt == 'sequenceDiagram':
        return open("prompts/sequenceDiagram.md", encoding="utf-8").read()
    elif dt == 'flowchart':
        return open("prompts/flowchart.md", encoding="utf-8").read()
    elif dt == 'mindmap':
        return open("prompts/mindmap.md", encoding="utf-8").read()
    elif dt == 'xychart':
        return open("prompts/xychart.md", encoding="utf-8").read()
    elif dt == 'pie':
        return open("prompts/pie.md", encoding="utf-8").read()
    elif dt == 'gantt':
        return open("prompts/gantt.md", encoding="utf-8").read()
    elif dt == 'quadrantChart':
        return open("prompts/quadrantChart.md", encoding="utf-8").read()
    elif dt == 'stateDiagram':
        return open("prompts/stateDiagram.md", encoding="utf-8").read()


def get_diagram(dt, mc, ra=1):
    diagram_path = exec_mermaid_cli(mc, dt)
    if not diagram_path or not os.path.isfile(diagram_path) or not os.path.exists(diagram_path):
        return "產生圖表失敗，請檢查圖表語法是否正確。{0}".format(diagram_path if diagram_path else '')
    encode_pecko = generate_mermaid_diagram(mc)
    return_markdown = '![{0}]({1}  "{2}")'.format(
        '由mermaid生成', diagram_path, dt) + "\n  " + \
                      "[查看全屏圖表]({0})\n  [在線編輯圖表]({1})\n  ".format(
                          'https://mermaid.live/view#pako:' + encode_pecko,
                          'https://mermaid.live/edit#pako:' + encode_pecko)
    _place_holder_id = 'get_diagram_results_{0}'.format(str(uuid.uuid4().node))
    _place_holder = '@Placeholder({0})'.format(_place_holder_id)
    cxt.placeholder_lookup[_place_holder_id] = return_markdown
    return "Mermaid圖表已經繪製完成，請在你的回覆內容需要呈現{0}之處改放置此佔位符字串作為代表:{1}".format(dt,
                                                                                                          _place_holder)
