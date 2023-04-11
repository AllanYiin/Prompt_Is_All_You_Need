# -*- coding: utf-8-sig -*-
import json
import os

import gradio as gr
import openai
import requests
from utils.chatgpt_utils import *

# 設置您的OpenAI API金鑰
openai.api_key = os.getenv("OPENAI_API_KEY")
URL = "https://api.openai.com/v1/chat/completions"

def index2context(idx:int):
    if idx is None or idx==0:
        return '[@PROMPT]'
    elif idx==1:
        return '[@GLOBAL]'
    elif idx == 2:
        return '[@SKIP]'
    elif idx == 3:
        return '[@SANDBOX]'
    elif idx == 4:
        return '[@EXPLAIN]'
    elif idx == 5:
        return '[@OVERRIDE]'
    else:
        return '[@PROMPT]'

def prompt_api(inputs, context_type, top_p, temperature, top_k, frequency_penalty, history=[]):
    _context_type=index2context(context_type)
    headers = {
        'Accept': 'text/event-stream',
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }
    is_pause=False
    if _context_type == "[@GLOBAL]":
        history[0]["content"]= history[0]["content"]+'/n'+inputs

        chat = [(process_chat(history[i]), process_chat(history[i + 1])) for i in
                range(1, len(history) - 1, 2) if history[i]['role'] != 'system']

        yield chat, history,history
    elif _context_type == "[@OVERRIDE]":
        history[0]["content"]= inputs
        chat = [(process_chat(history[i]), process_chat(history[i + 1])) for i in
                range(1, len(history) - 1, 2) if history[i]['role'] != 'system']
        yield chat, history,history


    elif inputs:
        # 調用openai.ChatCompletion.create來生成機器人的回答
        message_context = process_context(inputs, _context_type,history)
        partial_words = ''
        token_counter = 0
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": message_context,
            "temperature": temperature,
            "top_p": top_p,
            "n": top_k,
            "stream": True,
            "frequency_penalty": frequency_penalty
        }


        history.append({"role": "user", "content": inputs,"context_type":_context_type})
        request = requests.post(URL, headers=headers, json=payload, stream=True)

        finish_reason = 'None'
        # client = sseclient.SSEClient(request)
        for chunk in request.iter_content(chunk_size=512, decode_unicode=False):

            try:
                if is_pause:
                    finish_reason = '[DONE]'
                    request.close()
                if chunk.decode('utf-8-sig').endswith('data: [DONE]\n\n'):
                    finish_reason = '[DONE]'
                    # sys.stdout.write('[DONE]')
                    break
                jstrs = chunk.decode('utf-8-sig').replace(':null', ':\"None\"')[5:]
                this_chunk = json.loads(jstrs)
                this_choice = this_chunk['choices'][0]['delta']
                finish_reason = this_chunk['choices'][0]['finish_reason']

                if 'content' in this_choice:
                    if partial_words == '' and this_choice['content'] == '\n\n':
                        pass
                    elif this_choice['content'] == '\n\n':
                        partial_words += '\n  '
                    else:
                        partial_words += this_choice['content']
                        # sys.stdout.write(this_choice['content'])
                    if token_counter == 0:
                        history.append({"role": "assistant", "content": partial_words,"context_type":_context_type})
                    else:
                        history[-1]['content'] = partial_words
                    chat = [(process_chat(history[i]), process_chat(history[i + 1])) for i in
                            range(1, len(history) - 1, 2) if history[i]['role'] != 'system']
                    token_counter += 1
                    yield chat, history,history



            except Exception as e:
                if len(partial_words) == 0:
                    pass
                else:
                    print(e)
                    print(pattern.findall(chunk.decode('utf-8-sig').replace(':null', ':\"None\"')))


        # 檢查finish_reason是否為length
        while finish_reason != '[DONE]':
            # 自動以user角色發出「繼續寫下去」的PROMPT
            # print("[繼續寫下去]\n")
            prompt = "繼續"
            # 調用openai.ChatCompletion.create來生成機器人的回答
            message_context = process_context(prompt,context_type, history)
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": message_context,
                "temperature": temperature,
                "top_p": top_p,
                "n": top_k,
                "stream": True,
                "frequency_penalty": frequency_penalty
            }
            request = requests.post(URL, headers=headers, json=payload, stream=True)
            partial_words = ''
            finish_reason = 'None'
            # client = sseclient.SSEClient(request)
            for chunk in request.iter_content(chunk_size=512):
                # tt=chunk.decode('utf-8')[6:].rstrip('\n)
                try:

                    jstrs = chunk.decode('utf-8-sig').replace(':null', ':\"None\"')[5:]
                    this_chunk = json.loads(jstrs)
                    this_choice = this_chunk['choices'][0]['delta']
                    finish_reason = this_chunk['choices'][0]['finish_reason']
                    if chunk.decode('utf-8').endswith('data: [DONE]\n\n'):
                        finish_reason = '[DONE]'
                        sys.stdout.write('[DONE]')
                        break
                    if 'content' in this_choice:
                        if partial_words == '' and this_choice['content'] == '\n\n':
                            pass
                        elif this_choice['content'] == '\n\n':
                            partial_words += '\n'
                        else:
                            partial_words += this_choice['content']
                            # sys.stdout.write(this_choice['content'])
                        if token_counter == 0:
                            history.append({"role": "assistant", "content": partial_words,"context_type":_context_type})
                        else:
                            history[-1]['content'] = partial_words
                        chat = [(process_chat(history[i]), process_chat(history[i + 1])) for i in
                                range(1, len(history) - 1, 2) if history[i]['role'] != 'system']

                        token_counter += 1
                        yield chat, history,history

                except Exception as e:
                    if len(partial_words) == 0:
                        pass
                    else:
                        print(e)
                        print(chunk.decode('utf-8'))

        # 檢查接續後的完整回覆是否過長
        # print('bot_output: ',len(bot_output))

        if len(partial_words) > 500:
            # 自動以user角色要求「將此內容進行簡短摘要」的prompt需
            print(len(partial_words))
            print('\n[簡短摘要]\n')
            prompt = "使用繁體中文直接以下內容進行簡短摘要:" + partial_words
            # 調用openai.ChatCompletion.create來生成機器人的回答

            payload = {
                "model": "gpt-3.5-turbo",
                "messages": prompt,
                "temperature": temperature,
                "top_p": top_p,
                "n": top_k,
                "stream": True,
                "frequency_penalty": frequency_penalty
            }
            request = requests.post(URL, headers=headers, json=payload, stream=True)
            summary = ''
            finish_reason = 'None'
            # client = sseclient.SSEClient(request)
            for chunk in request.iter_content(chunk_size=512):
                # tt=chunk.decode('utf-8')[6:].rstrip('\n)
                try:

                    if chunk.decode('utf-8').endswith('data: [DONE]\n\n'):
                        finish_reason = '[DONE]'
                        sys.stdout.write('[DONE]')
                        break

                    jstrs = chunk.decode('utf-8-sig').replace(':null', ':\"None\"')[5:]
                    this_chunk = json.loads(jstrs)
                    this_choice = this_chunk['choices'][0]['delta']
                    finish_reason = this_chunk['choices'][0]['finish_reason']

                    if 'content' in this_choice:
                        if summary == '' and this_choice['content'] == '\n\n':
                            pass
                        elif this_choice['content'] == '\n\n':
                            summary += '\n'
                        else:
                            summary += this_choice['content']
                            # sys.stdout.write(this_choice['content'])
                        history[-1]['summary'] = summary
                        chat = [(process_chat(history[i]), process_chat(history[i + 1])) for i in
                                range(1, len(history) - 1, 2) if history[i]['role'] != 'system']

                        token_counter += 1
                        yield chat, history,history
                except Exception as e:
                    if len(summary) == 0:
                        pass
                    else:
                        print(e)
                        print(chunk.decode('utf-8'))
            # messages_history.append({"role": "assistant", "content": summary})
            print('summary: ', len(summary),summary)
        else:
            chat = [(process_chat(history[i]), process_chat(history[i + 1])) for i in
                    range(1, len(history) - 1, 2) if history[i]['role'] != 'system']
            yield chat, history,history

        # 顯示機器人的回答
        # window["bot_output"].update(bot_output)
        # 將用戶的輸入和機器人的回答加入歷史對話變數中

    else:
        # 如果用戶沒有輸入任何內容，則提示用戶
        yield "請輸入您想要說的話。", history,history

def nlu_api(text_input):
    # 創建與API的對話
    conversation = [
        {
            "role": "system",
            "content": "請逐一讀取下列句子，每個句子先理解語意後再進行分詞、情感偵測、命名實體偵測以及意圖偵測。分詞結果是指將輸入的文字，先進行語意理解，然後基於語意理解合理性的前提下，將輸入文字進行分詞(tokenize)，若非必要，盡量不要出現超過3個字的詞，然後使用「|」插入至分詞詞彙即構成分詞結果。\n需要偵測的情感類型\n正面情緒(positive_emotions):[自信,快樂,體貼,幸福,信任,喜愛,尊榮,期待,感動,感謝,熱門,獨特,稱讚]\n負面情緒(negative_emotions):[失望,危險,後悔,冷漠,懷疑,恐懼,悲傷,憤怒,擔心,無奈,煩悶,虛假,討厭,貶責,輕視]\n當句子中有符合以上任何情感種類時，請盡可能的將符合的「情感種類」及句子中的那些「觸及到情感種類的內容」成對的列舉出來，一個句子可以觸及不只一種情感。\n需要偵測的實體類型(entities)[中文人名,中文翻譯人名,外語人名,地名/地點,時間,公司機構名/品牌名,商品名,商品規格,化合物名/成分名,其他專有名詞,金額,其他數值]\n此外，若是句子中有偵測到符合上述實體類型時，也請盡可能的將符合的「實體類型」及句子中的那些「觸及到實體類型內容｣成對的列舉出來，一個句子可以觸及不只一種實體類型。當你偵測到句子中有要求你代為執行某個任務或是查詢某資訊的意圖(intents)時，根據以英文「名詞+動詞-ing」的駝峰式命名形式來組成意圖類別(例如使用者說「請幫我訂今天下午5點去高雄的火車票」其意圖類別為TicketOrdering)，及句子中的那些「觸及到意圖類別的內容」成對的列舉出來，一個句子可以觸及不只一種意圖。以下為「張大帥的人生是一張茶几，上面放滿了杯具。而本身就是杯具」的範例解析結果\n"
            "{\nsentence:  \"張大帥的人生是一張茶几，上面放滿了杯具。而本身就是杯具\",\nsegmented_sentence:  \"張大帥|的|人生|是|一|張|茶几|，|上面|放滿了|杯具|。|而|本身|就是|杯具\",\npositive_emotions:  [\n0:  {\ntype:  \"煩悶\",\ncontent:  \"放滿了杯具\"\n} ,\n1:  {\ntype:  \"無奈\",\ncontent:  \"本身就是杯具\"\n}\n],\nnegative_emotions:  [\n0:  {\ntype:  \"失望\",\ncontent:  \"上面放滿了杯具\"\n} \n],\nentities:  [\n0:  {\ntype:  \"中文人名\",\ncontent:\"張大帥\"\n}\n]\n}\n\r最後將每個句子的解析結果整合成單一json格式，縮進量為1。"
        },
        {
            "role": "user",
            "content": text_input
        }
    ]
    # 向OpenAI ChatGPT API發送請求
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation,
        temperature=0.1
    )
    try:
        # 解析返回的JSON結果
        jstrs=pattern.findall(response.choices[0].message['content'].strip())
        jstrs=jstrs[0] if len(jstrs)==1 else '['+', '.join(jstrs)+']'
        output_json = json.loads(jstrs)
        return json.dumps(output_json, ensure_ascii=False, indent=4)
    except Exception as e:
        return response.choices[0].message['content'].strip()+"\n"+str(e)

def reset_textbox():
    return gr.update(value='')

def reset_context():
    return gr.update(value="[@PROMPT] 一般指令")

def pause_message():
    is_pause=True



title = """<h1 align="center">🔥🤖ChatGPT Streaming 🚀</h1>"""
description = ""

with gr.Blocks(css="""#col_container {width: 70%; margin-left: auto; margin-right: auto;}
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
                    nlu_inputs = gr.Textbox(lines=6, placeholder="輸入句子...")
                    nlu_output =gr.Text(label="回傳的JSON視覺化",interactive=True,max_lines=40)
                nlu_button = gr.Button("送出")



        # with gr.TabItem("翻譯"):
        #     with gr.Column(elem_id="col_container"):
        #         history_viewer = gr.Text(elem_id='history_viewer',max_lines=30)
    # inputs, top_p, temperature, top_k, repetition_penalty
    with gr.Accordion("超參數", open=False):
        top_p = gr.Slider(minimum=-0, maximum=1.0, value=0.95, step=0.05, interactive=True,
                          label="限制取樣範圍(Top-p)", )
        temperature = gr.Slider(minimum=-0, maximum=5.0, value=0.5, step=0.1, interactive=True,
                                label="溫度 (Temperature)", )
        top_k = gr.Slider(minimum=1, maximum=50, value=1, step=1, interactive=True, label="候選結果個數(Top-k)", )
        frequency_penalty = gr.Slider(minimum=-2, maximum=2, value=0, step=0.01, interactive=True,
                                      label="重複性處罰(Frequency Penalty)",
                                      info='值域為-2~+2，數值越大，對於重複用字會給予懲罰，數值越負，則鼓勵重複用字')



    inputs.submit(prompt_api, [inputs, context_type, top_p, temperature, top_k, frequency_penalty, state],
                  [chatbot, state,history_viewer]).then(reset_context, [], [context_type]).then(reset_textbox, [], [inputs])
    b1.click(prompt_api, [inputs, context_type, top_p, temperature, top_k, frequency_penalty, state], [chatbot, state,history_viewer], )
    b3.click(reset_textbox, [], [inputs])
    b2.click(fn=pause_message, inputs=[], outputs=None)


    nlu_inputs.submit(nlu_api, [nlu_inputs],[nlu_output])
    nlu_button.click(nlu_api, [nlu_inputs], [nlu_output])

    gr.Markdown(description)
    demo.queue(concurrency_count=3,api_open=False).launch(debug=True)
