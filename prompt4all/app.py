# -*- coding: utf-8-sig -*-
import json
import os
import datetime
import time
import string
import builtins
import uuid
import regex
from pathlib import Path
import gradio as gr
import openai
import inspect
import copy
import requests
import asyncio
import nest_asyncio
import openai_async
import whisper
import math
import numpy as np
from pydub import AudioSegment
from datetime import datetime, timedelta
from queue import Queue
from openai.types import beta
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI
from gradio.routes import safe_join

nest_asyncio.apply()
from collections import OrderedDict
from datetime import datetime
from prompt4all.utils.chatgpt_utils import *
from prompt4all.utils.regex_utils import *
import prompt4all.api.context_type as ContextType
from prompt4all.api.base_api import *
from prompt4all.api.message import *
from prompt4all.utils.tokens_utils import *
from prompt4all.utils.summary_utils import *
from prompt4all.utils.pdf_utils import *

from prompt4all.theme import adjust_theme, advanced_css
from prompt4all.utils.whisper_utils import *
from prompt4all.common import *

from prompt4all import context
from prompt4all.context import *
from prompt4all.common import find_available_port
from prompt4all.ui import settings_ui, rewrite_ui, image_ui, nlu_ui, assistant_ui, chatbot_ui, summarization_ui
from openai import OpenAI, AsyncOpenAI, AzureOpenAI
import uvicorn

# app = FastAPI()

os.chdir(os.path.dirname(__file__))
cxt = context._context()
os.environ['no_proxy'] = '*'

# 設置您的OpenAI API金鑰
# 請將您的金鑰值寫入至環境變數"OPENAI_API_KEY"中
# os.environ['OPENAI_API_KEY']=#'你的金鑰值'
if "OPENAI_API_KEY" not in os.environ:
    print("OPENAI_API_KEY  is not exists!")

openai.api_key = os.getenv("OPENAI_API_KEY")
model_lists = openai.models.list()

##
new_assistants = []
for a in cxt.assistants:
    if isinstance(a, str):
        _assistant = beta.Assistant(**json.loads(a))

        new_assistants.append(_assistant)
cxt.__setattr__('assistants', new_assistants)
initialize_conversation_history()


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


def message2chat():
    _state = cxt.state.value
    chat = []
    # chat.append((None,
    #              '![流程圖](./generate_images/1701555707-日本櫻前線數據為主題的立體infographic-1.png  "流程圖")'))
    # chat.append(
    #     (None,
    #      '<div style="height:350px"><img title="流程圖" alt="流程圖"  src="http://localhost:7860/generate_images/1701555707-日本櫻前線數據為主題的立體infographic-1.png"></div>'))
    if _state:
        for i in range(len(_state)):
            current_message = process_chat(_state[i])
            if _state[i]['role'] == 'user':
                next_message = process_chat(_state[i + 1]) if i < len(_state) - 1 else None
                chat.append((current_message, next_message))
            elif _state[i]['role'] == 'assistant':
                if current_message is None or len(current_message.strip()) == 0:
                    pass
                elif i > 0 and _state[i - 1]['role'] == 'user':
                    pass
                else:
                    chat.append((None, current_message))
    if cxt.baseChatGpt and cxt.baseChatGpt.temp_state and len(cxt.baseChatGpt.temp_state) > 0:
        for i in range(len(cxt.baseChatGpt.temp_state)):
            current_message = process_chat(cxt.baseChatGpt.temp_state[i])
            if current_message is None or len(current_message.strip()) == 0:
                pass
            else:
                chat.append((None, current_message))

    if len(chat) > 0:
        return chat
    return None


def clear_history():
    FULL_HISTORY = [{"role": "system", "content": cxt.baseChatGpt.SYSTEM_MESSAGE,
                     "estimate_tokens": estimate_used_tokens(cxt.baseChatGpt.SYSTEM_MESSAGE,
                                                             model_name=cxt.baseChatGpt.API_MODEL)}]
    return [], FULL_HISTORY, FULL_HISTORY


def reset_textbox():
    return gr.Textbox(placeholder="什麼是LLM?", value="",
                      label="輸入文字後按enter", lines=10, max_lines=2000)


def reset_context():
    return gr.Dropdown(
        ["[@PROMPT] 一般指令", "[@GLOBAL] 全局指令", "[@SKIP] 跳脫上文",
         "[@SANDBOX] 沙箱隔絕",
         "[@EXPLAIN] 解釋上文", "[@OVERRIDE] 覆寫全局"],
        value="[@PROMPT] 一般指令", type='index', label="context處理",
        elem_id='context_type',
        interactive=True)


def pause_message():
    is_pause = True


if __name__ == '__main__':
    PORT = find_available_port(7860)
    title = """<h1 align="center">🔥🤖Prompt is All You Need! 🚀</h1>"""
    if "OPENAI_API_KEY" not in os.environ:
        title = """<h1 align="center">🔥🤖Prompt is All You Need! 🚀</h1><br><h2 align="center"><span style='color:red'>你尚未設置api key</span></h2>"""
    description = ""
    cancel_handles = []

    this_blocks = gr.Blocks(title="Prompt is what you need!", css=advanced_css, analytics_enabled=False,
                            theme=adjust_theme())

    with this_blocks:

        cxt.baseChatGpt = GptBaseApi(cxt.baseChatGpt) if cxt.baseChatGpt else GptBaseApi(model="gpt-4-1106-preview")
        cxt.baseChatGpt.enable_database_query(cxt.is_db_enable)
        cxt.summaryChatGpt = GptBaseApi(cxt.summaryChatGpt) if cxt.summaryChatGpt else GptBaseApi(
            model="gpt-3.5-turbo-1106")
        cxt.imageChatGpt = GptBaseApi(cxt.imageChatGpt) if cxt.imageChatGpt else GptBaseApi(model="gpt-4-1106-preview")
        cxt.otherChatGpt = GptBaseApi(cxt.otherChatGpt) if cxt.otherChatGpt else GptBaseApi(model="gpt-4-1106-preview")
        state = gr.State(
            [{"role": "system", "content": '所有內容以繁體中文書寫',
              "estimate_tokens": estimate_used_tokens('所有內容以繁體中文書寫',
                                                      model_name=cxt.baseChatGpt.API_MODEL)}])
        cxt.state = state
        cxt.counter = 0


        def get_status_word():
            cxt.counter += 1
            if cxt.counter == 3:
                cxt.counter = 0
            if len(cxt.status_word) > 0 and cxt.status_word.endswith('...'):
                return cxt.status_word.replace("...", "".join(["."] * (cxt.counter + 1)))
            return cxt.status_word


        # cxt.baseChatGpt.FULL_HISTORY = cxt.state.value
        gr.HTML(title)

        with gr.Tabs():
            with gr.TabItem("對話"):
                with gr.Row():
                    with gr.Tabs():
                        with gr.TabItem("聊天"):
                            with gr.Column(scale=3):
                                inputs = gr.Textbox(placeholder="什麼是LLM?",
                                                    label="輸入文字後按enter", lines=10, max_lines=2000)  # t
                                context_type = gr.Dropdown(
                                    ["[@PROMPT] 一般指令", "[@GLOBAL] 全局指令", "[@SKIP] 跳脫上文",
                                     "[@SANDBOX] 沙箱隔絕",
                                     "[@EXPLAIN] 解釋上文", "[@OVERRIDE] 覆寫全局"],
                                    value="[@PROMPT] 一般指令", type='index', label="context處理",
                                    elem_id='context_type',
                                    interactive=True)
                                with gr.Row(variant="panel"):
                                    b1 = gr.Button(value='送出', interactive=True, size='sm', scale=1)
                                    b2 = gr.Button(value='中止', interactive=True, size='sm', scale=1)

                                with gr.Row(variant="panel"):
                                    b4 = gr.Button(value='new chat', interactive=True, size='sm', scale=1)
                                    b3 = gr.ClearButton([inputs], interactive=True, size='sm', scale=1)

                                with gr.Accordion("超參數", open=False):
                                    top_p = gr.Slider(minimum=-0, maximum=1.0, value=1, step=0.05, interactive=True,
                                                      label="限制取樣範圍(Top-p)", )
                                    temperature = gr.Slider(minimum=-0, maximum=2.0, value=0.9, step=0.1,
                                                            interactive=True,
                                                            label="溫度 (Temperature)", )
                                    top_k = gr.Slider(minimum=1, maximum=50, value=1, step=1, interactive=True,
                                                      label="候選結果個數(Top-k)", )
                                    frequency_penalty = gr.Slider(minimum=-2, maximum=2, value=0, step=0.01,
                                                                  interactive=True,
                                                                  label="重複性處罰(Frequency Penalty)",
                                                                  info='值域為-2~+2，數值越大，對於重複用字會給予懲罰，數值越負，則鼓勵重複用字')
                        with gr.TabItem("對話紀錄"):
                            with gr.Column(scale=3):
                                conversation_history_share_btm = gr.Button('更名', scale=1, size='sm')
                                conversation_history_delete_btm = gr.Button('刪除', scale=1, size='sm')
                            with gr.Row():
                                history_list = gr.templates.List(value=cxt.conversation_history.titles, height=550,
                                                                 headers=["歷史"], datatype=["str"],
                                                                 col_count=(1, 'fixed'), interactive=True)
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(value=message2chat, elem_classes='chatbot', container=True, height=550,
                                             render_markdown=True,
                                             avatar_images=["images/avatar/human.png", "images/avatar/assistant.png"],
                                             show_copy_button=True, bubble_full_width=True, show_share_button=True,
                                             likeable=True, every=1,
                                             layout="panel")
                        b3.add(chatbot)
            with gr.TabItem("歷史"):
                with gr.Column(elem_id="col_container"):
                    history_viewer = gr.JSON(elem_id='history_viewer')
            with gr.TabItem("GPTs助理"):
                assistant_ui.assistant_panel()
            with gr.TabItem("NLU"):
                nlu_ui.nlu_panel()
            with gr.TabItem("Dall.E3"):
                image_ui.image_panel()
            with gr.TabItem("風格改寫"):
                rewrite_ui.rewrite_panel()
            with gr.TabItem("長文本摘要"):
                summarization_ui.summerization_panel()
            with gr.TabItem("設定"):
                with gr.Column():
                    settings_ui.service_type_panel()
                    dropdown_api1 = gr.Dropdown(choices=[k for k in model_info.keys()], value="gpt-4-1106-preview",
                                                label="對話使用之api", interactive=True)
                    dropdown_api4 = gr.Dropdown(choices=[k for k in model_info.keys()], value="gpt-4-1106-preview",
                                                label="以文生圖使用之api", interactive=True)
                    dropdown_api2 = gr.Dropdown(choices=[k for k in model_info.keys()], value="gpt-3.5-turbo-16k-0613",
                                                label="長文本摘要使用之api", interactive=True)
                    dropdown_api3 = gr.Dropdown(choices=[k for k in model_info.keys()], value="gpt-4-1106-preview",
                                                label="其他功能使用之api", interactive=True)
                    gr.Group(dropdown_api1, dropdown_api4, dropdown_api2, dropdown_api3)
                    settings_ui.database_query_panel()


        def prompt_api(inputs, context_type, top_p, temperature, top_k, frequency_penalty, full_history):
            cxt.baseChatGpt.temp_state.append({"role": "status", "content": '執行中...'})
            _context_type = index2context(context_type)
            cxt.baseChatGpt.API_PARAMETERS['temperature'] = temperature
            cxt.baseChatGpt.API_PARAMETERS['top_p'] = top_p
            cxt.baseChatGpt.API_PARAMETERS['top_k'] = top_k
            cxt.baseChatGpt.API_PARAMETERS['frequency_penalty'] = frequency_penalty

            # if isinstance(full_history, gr.State):
            #     full_history = full_history.value
            streaming_chat = cxt.baseChatGpt.post_a_streaming_chat(inputs, _context_type,
                                                                   cxt.baseChatGpt.API_PARAMETERS,
                                                                   full_history)
            while True:
                if _context_type == ContextType.override:
                    if len(cxt.conversation_history.selected_item.mapping) > 2:
                        keys = list(cxt.conversation_history.selected_item.mapping.keys())
                        cxt.conversation_history.selected_item.mapping[keys[1]].message.content.parts[0] = inputs
                elif _context_type == ContextType.globals:
                    if len(cxt.conversation_history.selected_item.mapping) > 2:
                        keys = list(cxt.conversation_history.selected_item.mapping.keys())
                        cxt.conversation_history.selected_item.mapping[keys[1]].message.content.parts.append(inputs)
                else:
                    _current_item = cxt.conversation_history.selected_item.current_item
                    mid = str(uuid.uuid4())
                    if _current_item.children is None:
                        _current_item.children = []
                    _current_item.children.append(mid)
                    cxt.conversation_history.selected_item.mapping[mid] = Mapping(mapping_id=mid,
                                                                                  parent=_current_item.id).new_user_message_mapping(
                        inputs)
                    cxt.conversation_history.selected_item.current_node = mid

                    pass
                try:
                    full_history = next(streaming_chat)
                    chat = message2chat()
                    # if len(full_history) > 2:
                    #     hisory = [item for item in full_history if item['role'] not in ['system', 'tool']]
                    #     if len(hisory) % 2 == 0:
                    #         chat = [(process_chat(hisory[i]), process_chat(hisory[i + 1])) for i in
                    #                 range(0, len(hisory), 2)]
                    #         if len(chat) > 0 and len(chat[-1]) == 2:
                    #             _last_tuple = chat[-1]
                    #             chat[-1] = (
                    #                 _last_tuple[0],
                    #                 _last_tuple[1] + "\n" + status_word.value if _last_tuple[
                    #                                                                  1] is not None else status_word.value)
                    yield chat, full_history
                except StopIteration:
                    break
                except Exception as e:
                    raise gr.Error(str(e))


        inputs_event = inputs.submit(prompt_api,
                                     [inputs, context_type, top_p, temperature, top_k, frequency_penalty, state],
                                     [chatbot, state])
        cancel_handles.append(inputs_event)
        inputs_event.then(reset_context, [], [context_type]).then(reset_textbox, [], [inputs])
        b1_event = b1.click(prompt_api, [inputs, context_type, top_p, temperature, top_k, frequency_penalty, state],
                            [chatbot, state])
        cancel_handles.append(b1_event)
        b1_event.then(reset_context, [], [context_type]).then(reset_textbox, [], [inputs])
        b3.click(clear_history, [], [state, history_viewer]).then(reset_textbox, [], [inputs])
        b2.click(fn=None, inputs=None, outputs=None, cancels=cancel_handles)
        chatbot.change(scroll_to_output=True)


        def new_chat(state):
            cxt.conversation_history.new_chat()
            history_list.value = cxt.conversation_history.titles
            chat = cxt.conversation_history.selected_item.get_gradio_chat()
            return state


        b4.click(fn=new_chat, inputs=[state], outputs=[state])


        def select_conversation(evt: gr.SelectData):
            conversations = cxt.conversation_history.conversations[evt.index[0]].get_prompt_messages(only_final=True)
            cxt.conversation_history.selected_index = evt.index[0]
            chat = cxt.conversation_history.selected_item.get_gradio_chat()
            return cxt.state


        history_list.select(select_conversation, None, [cxt.state])

        dropdown_api1.change(lambda x: cxt.baseChatGpt.change_model(x), [dropdown_api1], [])
        dropdown_api2.change(lambda x: cxt.summaryChatGpt.change_model(x), [dropdown_api2], [])
        dropdown_api3.change(lambda x: cxt.otherChatGpt.change_model(x), [dropdown_api3], [])
        dropdown_api4.change(lambda x: cxt.imageChatGpt.change_model(x), [dropdown_api4], [])

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

        # async def get_image(self, filename: str):
        #     return FileResponse(f"{generate_images}/{filename}")
        #
        # app = gr.routes.App.create_app(this_blocks, timeout=5)
        #
        this_blocks.queue(default_concurrency_limit=100, api_open=False, max_size=100)
        app = gr.routes.App.create_app(this_blocks)


        @app.get("/generate_images/{filename:path}")
        async def get_generate_images(filename: str):
            return FileResponse(f"generate_images/{filename}")


        @app.get("/images/{filename:path}")
        async def get_images(filename: str):
            return FileResponse(f"images/{filename}")


        @app.get("/download_pdfs/{filename:path}")
        async def get_download_pdfs(filename: str):
            return FileResponse(f"download_pdfs/{filename}")


        # gradio_

        # app.mount("/", gradio_app)

        # app = gr.mount_gradio_app(app, this_blocks, path="/")
        # # app.mount("/", StaticFiles(directory="generate", html=True), name="static")
        uvicorn.run(app, host='localhost', port=PORT)

        # app.add_route('/images/{filename}', get_image, methods=['GET'])

        # @app.get("/images/{path:path}")
        # def build_resource(path: str):
        #     build_file = safe_join(BUILD_PATH_LIB, path)
        #     return FileResponse(build_file)

        # @app.get("/file{file_path:path}")
        # async def file(file_path: str, request: Request):
        #     # redirect to the "api." host and gradio_tts path
        #     current_url = request.url
        #     if "localhost" in current_url.hostname or "127." in current_url.hostname:
        #         return RedirectResponse(url="http://localhost:8000/gradio_tts/file" + str(file_path))
        #     return RedirectResponse(url="https://api.text-generator.io/gradio_tts/file" + str(file_path))
        #
        # this_blocks.launch(show_error=True, prevent_thread_lock=True, allowed_paths=['generate_images'], show_api=True,
        #                    max_threads=200, share=True,
        #                    server_port=PORT)
