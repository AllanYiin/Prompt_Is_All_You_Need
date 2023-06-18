# -*- coding: utf-8-sig -*-
import json
import os
import regex
import gradio as gr
import openai
import copy
import requests
import asyncio
import time
import nest_asyncio
import openai_async
nest_asyncio.apply()
from collections import OrderedDict
from datetime import datetime
from prompt4all.utils.chatgpt_utils import *
from prompt4all.utils.regex_utils import *
import prompt4all.api.context_type as ContextType
from prompt4all.api.base_api import *
from prompt4all.utils.tokens_utils import *
#from gradio_chatbot_patch import Chatbot as grChatbot
# from gradio_css import code_highlight_css
from prompt4all.theme import adjust_theme, advanced_css

os.environ['no_proxy'] = '*'

# 設置您的OpenAI API金鑰
# 請將您的金鑰值寫入至環境變數"OPENAI_API_KEY"中
# os.environ['OPENAI_API_KEY']=#'你的金鑰值'
if "OPENAI_API_KEY" not in os.environ:
    print("OPENAI_API_KEY  is not exists!")
openai.api_key = os.getenv("OPENAI_API_KEY")
URL = "https://api.openai.com/v1/chat/completions"

pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')


def index2context(idx: int):
    if idx is None or idx == 0:
        return ContextType.prompt
    elif idx == 1:
        return ContextType.globals
    elif idx == 2:
        return ContextType.skip
    elif idx == 3:
        return ContextType.sandbox
    elif idx == 4:
        return ContextType.explain
    elif idx == 5:
        return ContextType.override
    else:
        return '[@PROMPT]'


def prompt_api(inputs, context_type, top_p, temperature, top_k, frequency_penalty, full_history=[]):
    _context_type = index2context(context_type)
    baseChatGpt.API_PARAMETERS['temperature'] = temperature
    baseChatGpt.API_PARAMETERS['top_p'] = top_p
    baseChatGpt.API_PARAMETERS['top_k'] = top_k
    baseChatGpt.API_PARAMETERS['frequency_penalty'] = frequency_penalty
    streaming_chat = baseChatGpt.post_a_streaming_chat(inputs, _context_type, baseChatGpt.API_PARAMETERS, full_history)
    while True:
        try:
            chat, answer, full_history = next(streaming_chat)
            yield chat, full_history, full_history
        except StopIteration:
            break
        except Exception as e:
            raise ValueError(e)


def nlu_api(text_input):
    # 創建與API的對話

    text_inputs = text_input.split('\n')
    _parameters = copy.deepcopy(baseChatGpt.API_PARAMETERS)
    _parameters['temperature'] = 0.1
    results = []
    for txt in text_inputs:
        conversation = [
            {
                "role": "system",
                "content": "請逐一讀取下列句子，每個句子先理解語意後再進行分詞、情感偵測、命名實體偵測以及意圖偵測。分詞結果是指將輸入的文字，先進行語意理解，然後基於語意理解合理性的前提下，將輸入文字進行分詞(tokenize)，若非必要，盡量不要出現超過3個字的詞，然後使用「|」插入至分詞詞彙即構成分詞結果。\n需要偵測的情感類型\n正面情緒(positive_emotions):[自信,快樂,體貼,幸福,信任,喜愛,尊榮,期待,感動,感謝,熱門,獨特,稱讚]\n負面情緒(negative_emotions):[失望,危險,後悔,冷漠,懷疑,恐懼,悲傷,憤怒,擔心,無奈,煩悶,虛假,討厭,貶責,輕視]\n當句子中有符合以上任何情感種類時，請盡可能的將符合的「情感種類」及句子中的那些「觸及到情感種類的內容」成對的列舉出來，一個句子可以觸及不只一種情感。\n需要偵測的實體類型(entities)[中文人名,中文翻譯人名,外語人名,地名/地點,時間,公司機構名/品牌名,商品名,商品規格,化合物名/成分名,其他專有名詞,金額,其他數值]\n此外，若是句子中有偵測到符合上述實體類型時，也請盡可能的將符合的「實體類型」及句子中的那些「觸及到實體類型內容｣成對的列舉出來，一個句子可以觸及不只一種實體類型。當你偵測到句子中有要求你代為執行某個任務或是查詢某資訊的意圖(intents)時，根據以英文「名詞+動詞-ing」的駝峰式命名形式來組成意圖類別(例如使用者說「請幫我訂今天下午5點去高雄的火車票」其意圖類別為TicketOrdering)，及句子中的那些「觸及到意圖類別的內容」成對的列舉出來，一個句子可以觸及不只一種意圖。以下為「張大帥的人生是一張茶几，上面放滿了杯具。而本身就是杯具」的範例解析結果\n"
                           "{\nsentence:  \"張大帥的人生是一張茶几，上面放滿了杯具。而本身就是杯具\",\nsegmented_sentence:  \"張大帥|的|人生|是|一|張|茶几|，|上面|放滿了|杯具|。|而|本身|就是|杯具\",\npositive_emotions:  [\n0:  {\ntype:  \"煩悶\",\ncontent:  \"放滿了杯具\"\n} ,\n1:  {\ntype:  \"無奈\",\ncontent:  \"本身就是杯具\"\n}\n],\nnegative_emotions:  [\n0:  {\ntype:  \"失望\",\ncontent:  \"上面放滿了杯具\"\n} \n],\nentities:  [\n0:  {\ntype:  \"中文人名\",\ncontent:\"張大帥\"\n}\n]\n}\n\r最後將每個句子的解析結果整合成單一json格式，縮進量為1。"
            },
            {
                "role": "user",
                "content": txt
            }
        ]
        jstrs = pattern.findall(baseChatGpt.post_and_get_answer(conversation, _parameters))
        jstrs = jstrs[0] if len(jstrs) == 1 else '[' + ', '.join(jstrs) + ']'
        output_json = json.loads(jstrs)
        results.append(json.dumps(output_json, ensure_ascii=False, indent=3))

        yield '[' + ', '.join(results) + ']'


def image_api(text_input, image_size, temperature=1.2):
    # 創建與API的對話

    _parameters = copy.deepcopy(baseChatGpt.API_PARAMETERS)
    _parameters['temperature'] = temperature
    _parameters['max_tokens'] = 100
    results = []
    conversation = [
        {
            "role": "system",
            "content": "你是一個才華洋溢的視覺藝術設計師以及DALLE-2提示專家，你會根據輸入的視覺需求以及風格設計出吸引人的視覺構圖，並懂得適時的運用視覺風格專有名詞、視覺風格形容詞、畫家或視覺藝術家名字以及各種渲染效果名稱來指導生成圖片的效果，並擅長將構圖轉化成為DALLE-2影像生成模型能理解的prompt，如果只是將輸入需求直接翻譯成英文，或是用打招呼、自我介紹或是與視覺無關的內容來充數，這是你職業道德不允許的行為。請確保產生的圖像具備高解析度以及高質感以及包含構圖中的視覺細節，只需要回覆我prompt的本體即可，不需要解釋輸入需求文字的意義，prompt內只會保留會出現在影像中的物體以及其他視覺有關的文字描述，prompt本體以\"An image\"開始，你生成的prompt長度絕對不要超過800 characters, 請用英語撰寫"
        },
        {
            "role": "user",
            "content": text_input
        }
    ]
    image_prompt = baseChatGpt.post_and_get_answer(conversation, _parameters)
    if ':' in image_prompt:
        image_prompt = ' '.join(image_prompt.split(':')[1:])
    images_urls = baseChatGpt.generate_images(image_prompt, text_input, image_size)
    return image_prompt, images_urls


def rewrite_api(text_input, style_name):
    # 創建與API的對話

    style_name = style_name.split('(')[0].strip()
    _parameters = copy.deepcopy(baseChatGpt.API_PARAMETERS)
    _parameters['temperature'] = 1.2
    _parameters['frequency_penalty'] = 1.5
    _parameters['presence_penalty'] = 0.5
    results = []
    conversation = [
        {
            "role": "system",
            "content": "你是一個寫作風格專家，使用繁體中文書寫"
        },
        {
            "role": "user",
            "content": "套用{0}的風格來改寫以下文字:\n{1}".format(style_name, text_input)
        }
    ]
    streaming_answer = baseChatGpt.post_and_get_streaming_answer(conversation, _parameters, conversation)
    while True:
        try:
            answer, full_history = next(streaming_answer)
            yield answer
        except StopIteration:
            break

async def summarize_text(text_input, system_prompt):
    """post 串流形式的對話
    :param system_prompt:
    :param text_input:
    :return:
    """
    partial_words = ''
    token_counter = 0
    context_type = ContextType.skip
    conversation = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": "#zh-TW \n輸入文字內容:\"\"\"\n{0}\n\"\"\"".format(text_input)
        }
    ]
    payload = baseChatGpt.parameters2payload(baseChatGpt.API_MODEL, conversation, baseChatGpt.API_PARAMETERS,stream=False)

    response =await asyncio.to_thread(
        requests.post,
        baseChatGpt.BASE_URL, headers=baseChatGpt.API_HEADERS, json=payload,stream=False
    )



    try:
        # 解析返回的JSON結果
        this_choice = json.loads(response.content.decode())['choices'][0]
        print(this_choice)
        summary =this_choice["message"]
        total_tokens = response.json()["usage"]['completion_tokens']
        summary['total_tokens'] = total_tokens
        return summary
    except Exception as e:
        return str(response.json()) + "\n" + str(e)


async def rolling_summary(large_inputs, full_history, is_parallel):
    _parameters = copy.deepcopy(baseChatGpt.API_PARAMETERS)
    _parameters['temperature'] = 0.001
    large_inputs = large_inputs.split('\n') if isinstance(large_inputs, str) else large_inputs
    large_inputs_bk = copy.deepcopy(large_inputs)
    st = datetime.now()
    if is_parallel:
        _system_prompt = "#你是一個萬能文字助手，你擅長使用繁體中文以條列式的格式來整理逐字稿、會議記錄以及長文本文件，你懂得如何將「輸入文字內容」視狀況修正同音錯字，去除口語贅字後，尤其是涉及[人名、公司機構名稱、事物名稱、地點、時間、數量、知識點、事實、數據集、url]這些資訊時，在保持原意不變的前提下進行摘要，請基於縮排或標號(\"-\")來表達摘要資訊間的階層性與相關性，輸出結果應該是「摘要清單」，不解釋原因。請以繁體中文書寫。 #zh-tw"
        this_system_tokens = estimate_used_tokens(_system_prompt) + estimate_used_tokens('system',
                                                                                         model_name=baseChatGpt.API_MODEL) + 4
        available_tokens = (baseChatGpt.MAX_TOKENS - this_system_tokens - 4 - 2) * 0.66
        keep_summary = True
        text_dict = OrderedDict()
        tasks = []
        n = 0
        while keep_summary:
            partial_words, large_inputs = get_next_paragraph(large_inputs, available_tokens)

            if len(large_inputs) == 0 or len(partial_words) == 0:
                keep_summary = False


            text_dict[n] = OrderedDict()
            text_dict[n]['text'] = '\n'.join(partial_words)
            tasks.append(summarize_text('\n'.join(partial_words), _system_prompt))
            time.sleep(0.5)
        print('預計切成{0}塊'.format(len(tasks)))
        return_values = await asyncio.gather(*tasks)
        print(datetime.now() - st)
        print(return_values)

        yield aggregate_summary(return_values), full_history

    else:

        _system_prompt = "你是一個萬能文字助手，你擅長使用繁體中文以條列式的格式來整理逐字稿以及會議記錄，你懂得如何基於滾動式摘要，將「輸入文字內容」視狀況修正同音錯字，去除口語贅字後，比對之前的「累積摘要清單」，若是有新增資訊，尤其是涉及[人名、公司機構名稱、事物名稱、地點、時間、數量、知識點、事實、數據集、url]這些資訊時，在保持原意不變的前提下，提煉為新的摘要內容並將其append至「累積摘要清單」中,請基於縮排或標號來表達摘要資訊間的階層性與相關性，所有已存在於「累積摘要清單」內的資訊在新的內容加入後可以視狀況作微調的二次摘要，但是原有的資訊不應該因此而丟失。輸出結果應該是「累積摘要清單」，不解釋原因。請以繁體中文書寫。"
        this_system_tokens = estimate_used_tokens(_system_prompt) + estimate_used_tokens('system',
                                                                                         model_name=baseChatGpt.API_MODEL) + 4
        available_tokens = baseChatGpt.MAX_TOKENS * 0.75 - this_system_tokens - 4 - 2
        summary_history = '空的清單'
        partial_words = ''

        keep_summary = True
        cnt = 0

        while keep_summary:
            this_system_tokens = estimate_used_tokens(summary_history)
            this_available_tokens = (available_tokens - this_system_tokens) - 100
            # get tokens
            partial_words, large_inputs = get_next_paragraph(large_inputs, this_available_tokens)

            if len(large_inputs) == 0 or len(partial_words) == 0:
                keep_summary = False
                break

            conversation = [
                {
                    "role": "system",
                    "content": _system_prompt
                },
                {
                    "role": "user",
                    "content": "#zh-tw \n累積摘要清單: \"\"\"\n{0}\n\"\"\"\n\n輸入文字內容:\"\"\"\n{1}\n\"\"\"".format(
                        summary_history, '\n'.join(partial_words))
                }
            ]
            print(conversation)
            streaming_answer = baseChatGpt.post_and_get_streaming_answer(conversation, _parameters, full_history)
            answer = ''
            answer_head = '[第{0}部分摘要]\n\r'.format(cnt + 1)
            while True:
                try:
                    answer, full_history = next(streaming_answer)
                    yield answer_head + answer, full_history
                except StopIteration:
                    break
            print(answer_head)
            print(answer)
            print('\n\n')
            summary_history = answer
            available_tokens = int(
                (baseChatGpt.MAX_TOKENS - 200 - estimate_used_tokens(answer) - this_system_tokens - 4 - 2) * 0.75)
            cnt += 1
            yield summary_history, full_history



def estimate_tokens(text,text2,state):
    t1= '輸入文本長度為{0},預計耗用tokens數為:{1}'.format(len(text),estimate_used_tokens(text,baseChatGpt.API_MODEL)+4)
    if len(text2)==0:
        return t1, state
    else:
        t2='輸出文本長度為{0},預計耗用tokens數為:{1}'.format(len(text2),estimate_used_tokens(text2,baseChatGpt.API_MODEL)+4)
        return t1+'\t\t'+t2, state


def process_file(file,state):
    if file is None:
        return '', state
    elif file.name.lower().endswith('.pdf'):
        doc_map=get_document_text(file.name)
        return_text=''
        for pg,offset,text in doc_map:
            return_text+=text+'\n'
            return_text += 'page {0}'.format(pg+1) + '\n''\n'
        return  return_text ,state
    else:
        with open(file.name, encoding="utf-8") as f:
          content = f.read()
          print(content)
        return content,state



def clear_history():
    FULL_HISTORY = [{"role": "system", "content": baseChatGpt.SYSTEM_MESSAGE,
                     "estimate_tokens": estimate_used_tokens(baseChatGpt.SYSTEM_MESSAGE,
                                                             model_name=baseChatGpt.API_MODEL)}]
    return [], FULL_HISTORY, FULL_HISTORY


def reset_textbox():
    return gr.update(value='')


def reset_context():
    return gr.update(value="[@PROMPT] 一般指令")


def pause_message():
    is_pause = True


if __name__ == '__main__':
    PORT = 7860
    title = """<h1 align="center">🔥🤖Prompt is All You Need! 🚀</h1>"""
    description = ""
    cancel_handles = []
    with gr.Blocks(title="Prompt is what you need!", css=advanced_css, analytics_enabled=False,
                   theme=adjust_theme()) as demo:
        baseChatGpt = GptBaseApi(model="gpt-3.5-turbo")
        state = gr.State([{"role": "system", "content": '所有內容以繁體中文書寫',
                           "estimate_tokens": estimate_used_tokens('所有內容以繁體中文書寫',
                                                                   model_name=baseChatGpt.API_MODEL)}])  # s

        baseChatGpt.FULL_HISTORY = state
        gr.HTML(title)

        with gr.Tabs():
            with gr.TabItem("對話"):
                with gr.Row():
                    with gr.Column(scale=3.5,elem_id="col_container"):
                        chatbot = gr.Chatbot(elem_id='chatbot',container=True,scale=1,height=550)
                    with gr.Column(scale=1):
                        with gr.Row():
                            inputs = gr.Textbox(placeholder="你與語言模型Bert有何不同?", label="輸入文字後按enter",lines=10,max_lines=2000)  # t
                            context_type = gr.Dropdown(
                                ["[@PROMPT] 一般指令", "[@GLOBAL] 全局指令", "[@SKIP] 跳脫上文", "[@SANDBOX] 沙箱隔絕",
                                 "[@EXPLAIN] 解釋上文", "[@OVERRIDE] 覆寫全局"],
                                value="[@PROMPT] 一般指令", type='index', label="context處理", elem_id='context_type',
                                interactive=True)
                        with gr.Row():
                            b1 = gr.Button(value='送出')
                            with gr.Row():
                                b3 = gr.Button(value='🧹')
                                b2 = gr.Button(value='⏹️')
                        with gr.Accordion("超參數", open=False):
                            top_p = gr.Slider(minimum=-0, maximum=1.0, value=1, step=0.05, interactive=True,
                                              label="限制取樣範圍(Top-p)", )
                            temperature = gr.Slider(minimum=-0, maximum=2.0, value=0.9, step=0.1, interactive=True,
                                                    label="溫度 (Temperature)", )
                            top_k = gr.Slider(minimum=1, maximum=50, value=1, step=1, interactive=True,
                                              label="候選結果個數(Top-k)", )
                            frequency_penalty = gr.Slider(minimum=-2, maximum=2, value=0, step=0.01, interactive=True,
                                                          label="重複性處罰(Frequency Penalty)",
                                                          info='值域為-2~+2，數值越大，對於重複用字會給予懲罰，數值越負，則鼓勵重複用字')
            with gr.TabItem("歷史"):
                with gr.Column(elem_id="col_container"):
                    history_viewer = gr.JSON(elem_id='history_viewer')
            with gr.TabItem("NLU"):
                with gr.Column(elem_id="col_container"):
                    gr.Markdown(
                        "將文本輸入到下面的方塊中，按下「送出」按鈕將文本連同上述的prompt發送至OpenAI ChatGPT API，然後將返回的JSON顯示在視覺化界面上。")
                    with gr.Row():
                        with gr.Column(scale=1):
                            nlu_inputs = gr.Textbox(lines=6, placeholder="輸入句子...")
                        with gr.Column(scale=2):
                            nlu_output = gr.Text(label="回傳的JSON視覺化", interactive=True, max_lines=40 ,show_copy_button=True)
                    nlu_button = gr.Button("送出")
            with gr.TabItem("Dall.E2"):
                with gr.Column(variant="panel"):
                    with gr.Row(variant="compact"):
                        image_text = gr.Textbox(
                            label="請輸入中文的描述",
                            show_label=False,
                            max_lines=1,
                            placeholder="請輸入中文的描述",
                            container=False
                        )
                    image_btn = gr.Button("設計與生成圖片" ,scale=1)
                    image_prompt = gr.Markdown("")
                    image_gallery = gr.Gallery(value=None, show_label=False,columns=[4], object_fit="contain",height="auto")
                with gr.Accordion("超參數", open=False):
                    temperature2 = gr.Slider(minimum=-0, maximum=2.0, value=0.7, step=0.1, interactive=True,
                                             label="溫度 (Temperature)", )
                    image_size = gr.Radio([256, 512, 1024], label="圖片尺寸", value=512)
            with gr.TabItem("風格改寫"):
                with gr.Column(elem_id="col_container"):
                    rewrite_dropdown = gr.Dropdown(
                        ["抽象 (Abstract)",
                         "冒險 (Adventurous)",
                         "比喻體 (Allegorical)",
                         "曖昧 (Ambiguous)",
                         "擬人化 (Anthropomorphic)",
                         "對比 (Antithetical)",
                         "領悟 (Aphoristic)",
                         "思辯 (Argumentative)",
                         "聲音式 (Auditory)",
                         "喚醒 (Awakening)",
                         "無邊際 (Boundless)",
                         "突破 (Breakthrough)",
                         "古典 (Classical)",
                         "口語 (Colloquial)",
                         "逆袭 (Comeback)",
                         "喜劇 (Comedic)",
                         "舒適 (Comforting)",
                         "簡潔 (Concise)",
                         "自信 (Confident)",
                         "體悟 (Contemplative)",
                         "反向思考 (Counterintuitive)",
                         "勇敢 (Courageous)",
                         "創意無限 (Creative)",
                         "深奧 (Cryptic)",
                         "可愛 (Cute)",
                         "飛舞 (Dancing)",
                         "燦爛 (Dazzling)",
                         "細緻 (Delicate)",
                         "描繪 (Descriptive)",
                         "冷漠 (Detached)",
                         "保持距離 (Distant)",
                         "夢幻 (Dreamy)",
                         "優雅 (Elegant)",
                         "感性 (Emotional)",
                         "迷人 (Enchanting)",
                         "無盡 (Endless)",
                         "隱喻 (Euphemistic)",
                         "精緻 (Exquisite)",
                         "充滿信念 (Faithful)",
                         "無畏 (Fearless)",
                         "無懈可擊 (Flawless)",
                         "靈活 (Flexible)",
                         "正式 (Formal)",
                         "自由 (Free Verse)",
                         "未來主義 (Futuristic)",
                         "天賦異禀 (Gifted)",
                         "壯麗 (Grandiose)",
                         "溫馨 (Heartwarming)",
                         "豪邁 (Heroic)",
                         "幽默 (Humorous)",
                         "誇張 (Hyperbolic)",
                         "個性化 (Idiomatic)",
                         "獨立 (Independent)",
                         "強烈 (Intense)",
                         "問答 (Interrogative)",
                         "疑問 (Interrogative)",
                         "道出内心 (Introspective)",
                         "反諷 (Ironic)",
                         "歡樂 (Joyful)",
                         "傳奇 (Legendary)",
                         "人生哲理 (Life Wisdom)",
                         "抒情 (Lyric)",
                         "魔幻 (Magical)",
                         "隱喻 (Metonymic)",
                         "現代 (Modern)",
                         "神秘 (Mysterious)",
                         "敘事 (Narrative)",
                         "自然主義 (Naturalistic)",
                         "高貴 (Noble)",
                         "懷舊 (Nostalgic)",
                         "客觀 (Objective)",
                         "原聲 (Onomatopoeic)",
                         "充滿激情 (Passionate)",
                         "激情 (Passionate)",
                         "個人 (Personal)",
                         "哲學 (Philosophical)",
                         "淺白 (Plain)",
                         "俏皮 (Playful)",
                         "詩意 (Poetic)",
                         "正能量 (Positive)",
                         "實用主義 (Pragmatic)",
                         "頌揚 (Praising)",
                         "亮麗 (Radiant)",
                         "叛逆 (Rebellious)",
                         "高雅 (Refined)",
                         "文藝復興 (Renaissance)",
                         "復古 (Retro)",
                         "啟示 (Revelatory)",
                         "革命 (Revolutionary)",
                         "修辭 (Rhetorical)",
                         "諷刺 (Satirical)",
                         "科幻 (Science Fiction)",
                         "魅惑 (Seductive)",
                         "聳人聽聞 (Sensational)",
                         "感傷 (Sentimental)",
                         "銳利 (Sharp)",
                         "疑問 (Skeptical)",
                         "社會評論 (Social Commentary)",
                         "嚴肅 (Solemn)",
                         "心靈 (Soulful)",
                         "靈性 (Spiritual)",
                         "主觀 (Subjective)",
                         "奇幻 (Surreal)",
                         "懸疑 (Suspenseful)",
                         "象徵 (Symbolic)",
                         "道家 (Taoist)",
                         "格調 (Tone)",
                         "傳統 (Traditional)",
                         "超凡脫俗 (Transcendent)",
                         "過渡 (Transitional)",
                         "流行 (Trendy)",
                         "從容 (Unhurried)",
                         "奔放 (Unrestrained)",
                         "充滿活力 (Vibrant)",
                         "漫遊式 (Wandering)",
                         "溫暖 (Warm)",
                         "充滿智慧 (Wise)",
                         "俏皮 (Witty)",
                         "瑜珈式 (Yogic)",
                         "青春 (Youthful)"], value="正式 (Formal)", multiselect=False, label="改寫文字風格形容詞",
                        interactive=True)
                    gr.Markdown("將文本輸入到下面的方塊中，選取改寫風格後，點選改寫後即可將文字基於選取風格進行改寫")
                    with gr.Row():
                        with gr.Column(scale=1):
                            rewrite_inputs = gr.Textbox(lines=30, placeholder="輸入句子...")
                        with gr.Column(scale=1):
                            rewrite_output = gr.Text(label="改寫", interactive=True, lines=30,show_copy_button=True)
                    rewrite_button = gr.Button("送出")
            with gr.TabItem("長文本摘要"):
                rolling_state = gr.State([])
                with gr.Column(elem_id="col_container"):
                    text_statistics=gr.Markdown()
                    with gr.Row():
                        rolliing_source_file=gr.File(file_count="single", file_types=["text", ".json", ".csv", ".pdf"])
                        rolling_parallel_checkbox=gr.Checkbox(label="平行計算",value=True)
                        rolling_button = gr.Button("送出")
                    with gr.Row():
                        with gr.Column(scale=1):
                            large_inputs = gr.Textbox(label="來源文字", lines=30, max_lines=5000,
                                                      placeholder="大量輸入...")
                        with gr.Column(scale=1):
                            summary_output = gr.Text(label="摘要", interactive=True, lines=30,
                                                     max_lines=500,show_copy_button=True)


        inputs_event = inputs.submit(prompt_api,
                                     [inputs, context_type, top_p, temperature, top_k, frequency_penalty, state],
                                     [chatbot, state, history_viewer])
        cancel_handles.append(inputs_event)
        inputs_event.then(reset_context, [], [context_type]).then(reset_textbox, [], [inputs])
        b1_event = b1.click(prompt_api, [inputs, context_type, top_p, temperature, top_k, frequency_penalty, state],
                            [chatbot, state, history_viewer])
        cancel_handles.append(b1_event)
        b1_event.then(reset_context, [], [context_type]).then(reset_textbox, [], [inputs])
        b3.click(clear_history, [], [chatbot, state, history_viewer]).then(reset_textbox, [], [inputs])
        b2.click(fn=None, inputs=None, outputs=None, cancels=cancel_handles)

        nlu_inputs.submit(nlu_api, nlu_inputs, nlu_output)
        nlu_button.click(nlu_api, nlu_inputs, nlu_output)

        image_text.submit(image_api, [image_text, image_size, temperature2], [image_prompt, image_gallery])
        image_btn.click(image_api, [image_text, image_size, temperature2], [image_prompt, image_gallery])

        rewrite_inputs.submit(rewrite_api, [rewrite_inputs, rewrite_dropdown], rewrite_output)
        rewrite_button.click(rewrite_api, [rewrite_inputs, rewrite_dropdown], rewrite_output)

        rolling_button.click(rolling_summary, [large_inputs, rolling_state,rolling_parallel_checkbox], [summary_output, rolling_state]).then(estimate_tokens, [large_inputs,summary_output, rolling_state],[text_statistics,rolling_state])
        large_inputs.submit(rolling_summary, [large_inputs, rolling_state,rolling_parallel_checkbox], [summary_output, rolling_state]).then(estimate_tokens, [large_inputs,summary_output, rolling_state],[text_statistics,rolling_state])
        large_inputs.change(estimate_tokens, [large_inputs,summary_output, rolling_state],[text_statistics,rolling_state])
        rolliing_source_file.change(process_file,[rolliing_source_file,rolling_state],[large_inputs,rolling_state])

        gr.Markdown(description)


        # gradio的inbrowser触发不太稳定，回滚代码到原始的浏览器打开函数
        def auto_opentab_delay():
            import threading, webbrowser, time
            print(f"若是瀏覽器未自動開啟，請直接點選以下連結：")
            print(f"\t（暗黑模式）: http://localhost:{PORT}/?__theme=dark")
            print(f"\t（光明模式）: http://localhost:{PORT}")

            def open():
                time.sleep(2)  # 打开浏览器
                DARK_MODE = True
                if DARK_MODE:
                    webbrowser.open_new_tab(f"http://localhost:{PORT}/?__theme=dark")
                else:
                    webbrowser.open_new_tab(f"http://localhost:{PORT}")

            threading.Thread(target=open, name="open-browser", daemon=True).start()


        auto_opentab_delay()
        demo.queue(concurrency_count=3, api_open=False).launch(show_error=True, max_threads=200, share=True)
