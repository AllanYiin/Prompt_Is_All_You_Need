# -*- coding: utf-8-sig -*-
import json
import os
import regex
import gradio as gr
import openai
import requests
from utils.chatgpt_utils import *
from utils.regex_utils import *
import api.context_type as ContextType
from api.base_api import *
from utils.chatgpt_utils import process_context,process_chat


# 設置您的OpenAI API金鑰
#請將您的金鑰值寫入至環境變數"OPENAI_API_KEY"中
#os.environ['OPENAI_API_KEY']=#'你的金鑰值'
if "OPENAI_API_KEY" not in os.environ:
    print("OPENAI_API_KEY  is not exists!")
openai.api_key = os.getenv("OPENAI_API_KEY")
URL = "https://api.openai.com/v1/chat/completions"

baseChatGpt=GptBaseApi(model="gpt-3.5-turbo")

pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')
def index2context(idx:int):
    if idx is None or idx==0:
        return ContextType.prompt
    elif idx==1:
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
    _context_type=index2context(context_type)
    baseChatGpt.API_PARAMETERS['temperature']=temperature
    baseChatGpt.API_PARAMETERS['top_p'] = top_p
    baseChatGpt.API_PARAMETERS['top_k'] = top_k
    baseChatGpt.API_PARAMETERS['frequency_penalty'] = frequency_penalty
    streaming_chat = baseChatGpt.post_a_streaming_chat(inputs, _context_type, baseChatGpt.API_PARAMETERS,baseChatGpt.FULL_HISTORY)
    while True:
        try:
            chat, answer, full_history= next(streaming_chat)
            yield chat,full_history,full_history
        except StopIteration:
            break



def nlu_api(text_input):
    # 創建與API的對話

    text_inputs=text_input.split('\n')
    _parameters = copy.deepcopy(baseChatGpt.API_PARAMETERS)
    _parameters['temperature'] = 0.1
    results=[]
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
        jstrs=pattern.findall(baseChatGpt.post_and_get_answer(conversation, _parameters))
        jstrs = jstrs[0] if len(jstrs) == 1 else '[' + ', '.join(jstrs) + ']'
        output_json = json.loads(jstrs)
        results.append(json.dumps(output_json, ensure_ascii=False, indent=3))

        yield '[' + ', '.join(results) + ']'


def image_api(text_input):
    # 創建與API的對話

    _parameters = copy.deepcopy(baseChatGpt.API_PARAMETERS)
    _parameters['temperature'] = 0.9
    _parameters['max_tokens'] = 100
    results=[]
    conversation = [
        {
            "role": "system",
            "content": "你是一個才華洋溢的DALLE-2 提示生成工具，你會根據接下來提供的視覺需求以及風格設計出對應的圖像配置，並且生成出能讓DALLE-2畫出符合需求的圖片之簡短prompt，為了節省字數，不一定要完整的句子，可以是一連串關鍵詞，無須打招互以及自我介紹，請確保圖像具備高解析度以及良好畫質，不要有文字出現在圖中以及要能夠吸引人類的目光，請用英語撰寫，絕對不要超過800 characters"
        },
        {
            "role": "user",
            "content": text_input
        }
    ]
    image_prompt=baseChatGpt.post_and_get_answer(conversation, _parameters)
    images_urls=baseChatGpt.generate_images(image_prompt)
    # images_urls=["https://www.digitaltrends.com/wp-content/uploads/2022/10/Variation-on-DALL-E-2-Prompt.jpg?fit=720%2C720&p=1",
    #             "https://www.digitaltrends.com/wp-content/uploads/2022/10/DALL-E-2-Image-on-OpenAI.jpg?p=1",
    #             "https://www.google.com/url?sa=i&url=https%3A%2F%2Ffuturism.com%2Fthe-byte%2Fopenai-dall-e2-realistic-images-descriptions&psig=AOvVaw0gIVytXCfu7fj-5LCPA3dM&ust=1681743982221000&source=images&cd=vfe&ved=0CBEQjRxqFwoTCJiQl9fWrv4CFQAAAAAdAAAAABAQ",
    #             "https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.reddit.com%2Fr%2FPokemonart%2Fcomments%2Fu05cci%2Fthis_was_generated_by_a_text_to_image_ai_dalle2%2F&psig=AOvVaw0gIVytXCfu7fj-5LCPA3dM&ust=1681743982221000&source=images&cd=vfe&ved=0CBEQjRxqFwoTCJiQl9fWrv4CFQAAAAAdAAAAABAY"]
    return image_prompt,images_urls


def clear_history():
    baseChatGpt.FULL_HISTORY=[baseChatGpt.FULL_HISTORY[0]]
    return [],baseChatGpt.FULL_HISTORY

def reset_textbox():
    return gr.update(value='')

def reset_context():
    return gr.update(value="[@PROMPT] 一般指令")

def pause_message():
    is_pause=True


if __name__ == '__main__':

    title = """<h1 align="center">🔥🤖ChatGPT Streaming 🚀</h1>"""
    description = ""

    with gr.Blocks(css="""#col_container {width: 80%; margin-left: auto; margin-right: auto;}
                    #chatbot {height: 50%; overflow: auto;}
                    #history_viewer {height: 50%; overflow: auto;}""",theme=gr.themes.Soft(spacing_size="sm", radius_size="none",font=["Microsoft JhengHei UI", "Arial", "sans-serif"])) as demo:
        gr.HTML(title)
        with gr.Tabs():
            with gr.TabItem("對話"):
                with gr.Column(elem_id="col_container"):
                    chatbot = gr.Chatbot(color_map=("orange", "dark gray"),elem_id='chatbot')  # c
                    with gr.Row():
                        radio = gr.Radio(["創意", "平衡", "精確"], show_label=False,interactive=True)
                        context_type = gr.Dropdown(
                            ["[@PROMPT] 一般指令", "[@GLOBAL] 全局指令", "[@SKIP] 跳脫上文", "[@SANDBOX] 沙箱隔絕", "[@EXPLAIN] 解釋上文", "[@OVERRIDE] 覆寫全局"],
                            value="[@PROMPT] 一般指令",type='index',label="context處理", elem_id='context_type',interactive=True)
                        scenario_type = gr.Dropdown(
                            ["(無)","電子郵件", "部落格文章","法律文件"],value="(無)", label="應用場景", elem_id='scenario',interactive=True)
                    with gr.Row():
                        inputs = gr.Textbox(placeholder="你與語言模型Bert有何不同?", label="輸入文字後按enter")  # t
                    with gr.Row():
                        b1 = gr.Button(value='送出')
                    with gr.Row():
                        b3 = gr.Button(value='🗑️')
                        b2 = gr.Button(value='⏸️')
                    state = gr.State([{"role": "system", "content": '所有內容以繁體中文書寫'}])  # s
            with gr.TabItem("歷史"):
                with gr.Column(elem_id="col_container"):
                    history_viewer =gr.JSON(elem_id='history_viewer')
            with gr.TabItem("NLU"):
                with gr.Column(elem_id="col_container"):
                    gr.Markdown("將文本輸入到下面的方塊中，按下「送出」按鈕將文本連同上述的prompt發送至OpenAI ChatGPT API，然後將返回的JSON顯示在視覺化界面上。")
                    with gr.Row():
                        with gr.Column(scale=1):
                            nlu_inputs = gr.Textbox(lines=6, placeholder="輸入句子...")
                        with gr.Column(scale=2):
                            nlu_output =gr.Text(label="回傳的JSON視覺化",interactive=True,max_lines=40).style(show_copy_button=True)
                    nlu_button = gr.Button("送出")
            with gr.TabItem("Dall.E2"):
                with gr.Column(variant="panel"):
                    with gr.Row(variant="compact"):
                        image_text = gr.Textbox(
                            label="請輸入中文的描述",
                            show_label=False,
                            max_lines=1,
                            placeholder="請輸入中文的描述",
                        ).style(
                            container=False,
                        )
                    image_btn = gr.Button("設計與生成圖片").style(full_width=False)
                    image_prompt=gr.Markdown("")
                    image_gallery = gr.Gallery(value=None,show_label=False).style(grid=4)


            # with gr.TabItem("翻譯"):
            #     with gr.Column(elem_id="col_container"):
            #         history_viewer = gr.Text(elem_id='history_viewer',max_lines=30)
        # inputs, top_p, temperature, top_k, repetition_penalty
        with gr.Accordion("超參數", open=False):
            top_p = gr.Slider(minimum=-0, maximum=1.0, value=0.95, step=0.05, interactive=True,
                              label="限制取樣範圍(Top-p)", )
            temperature = gr.Slider(minimum=-0, maximum=5.0, value=1, step=0.1, interactive=True,
                                    label="溫度 (Temperature)", )
            top_k = gr.Slider(minimum=1, maximum=50, value=1, step=1, interactive=True, label="候選結果個數(Top-k)", )
            frequency_penalty = gr.Slider(minimum=-2, maximum=2, value=0, step=0.01, interactive=True,
                                          label="重複性處罰(Frequency Penalty)",
                                          info='值域為-2~+2，數值越大，對於重複用字會給予懲罰，數值越負，則鼓勵重複用字')



        inputs.submit(prompt_api, [inputs, context_type, top_p, temperature, top_k, frequency_penalty, state],
                      [chatbot, state,history_viewer]).then(reset_context, [], [context_type]).then(reset_textbox, [], [inputs])
        b1.click(prompt_api, [inputs, context_type, top_p, temperature, top_k, frequency_penalty, state], [chatbot, state,history_viewer], )
        b3.click(clear_history, [], [chatbot,history_viewer]).then(reset_textbox, [], [inputs])
        b2.click(fn=pause_message, inputs=[], outputs=None)


        nlu_inputs.submit(nlu_api, nlu_inputs,nlu_output)
        nlu_button.click(nlu_api, nlu_inputs, nlu_output)

        image_text.submit(image_api, image_text,[image_prompt,image_gallery])
        image_btn.click(image_api, image_text, [image_prompt,image_gallery])

        gr.Markdown(description)
        demo.queue(concurrency_count=3,api_open=True).launch(show_error=True)
