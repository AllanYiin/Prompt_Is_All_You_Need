import glob
import time
import threading
import gradio as gr
from openai import OpenAI
from prompt4all import context
from prompt4all.context import *
from prompt4all.api.assistant import Assistant
from prompt4all.tools.web_tools import webpage_reader
from prompt4all.utils.chatgpt_utils import *

client = OpenAI()
client._custom_headers['Accept-Language'] = 'zh-TW'

cxt = context._context()


def message2chat():
    _state = cxt.assistant_state.value
    chat = []
    if _state:
        for i in range(len(_state)):
            current_message = process_chat(_state[i])
            if _state[i]['role'] == 'user':
                next_message = process_chat(_state[i + 1]) if i < len(_state) - 1 else None
                chat.append((current_message, next_message))
            elif _state[i]['role'] == 'assistant':
                current_message = process_chat(_state[i])
                if i > 0 and _state[i - 1]['role'] == 'user':
                    pass
                else:
                    chat.append((None, current_message))
    if cxt.current_assistant and cxt.current_assistant.temp_state and len(cxt.current_assistant.temp_state) > 0:
        for i in range(len(cxt.current_assistant.temp_state)):
            current_message = process_chat(cxt.current_assistant.temp_state[i])
            chat.append((None, current_message))
    if len(chat) > 0:
        return chat
    return None


def get_assistant_dropdown_list():
    return_list = ['➕建立新助理', '🔄根據id載入助理']
    return_list.extend([a.name for a in cxt.assistants])
    return return_list


def get_model_dropdown_list():
    """

    Returns: Available models

    """
    return [m.id for m in client.models.list() if 'gpt-4' in m.id or 'gpt-3.5' in m.id]


def upload_file(file_path):
    file = client.files.create(
        file=open("file_path", "rb"),
        purpose='assistants'
    )
    return file


def add_message_and_run(state, user_input):
    thread, run = create_thread_and_run(
        "Generate the first 20 fibbonaci numbers with code."
    )
    run = wait_on_run(run, thread)
    pretty_print(get_response(thread))


def get_assistant_files(assistant_id, assistant_state):
    if 'assistant_files' not in assistant_state:
        assistant_files = client.beta.assistants.files.list(assistant_id=assistant_id)
        assistant_files = [client.files.retrieve(file.id) for file in assistant_files]
        assistant_state['assistant_files'] = assistant_files
    return [f.filename for f in assistant_state['assistant_files']]


def retrieve_assistant(assistant_id):
    my_assistant = client.beta.assistants.retrieve(assistant_id)

    idx = -1
    for i in range(len(cxt.assistants)):
        if cxt.assistants[i].id == assistant_id:
            idx = i
    if idx == -1:
        cxt.assistants.append(my_assistant)
        cxt.write_session()
        idx = len(cxt.assistants) + 2 - 1
    else:
        cxt.assistants[idx] = my_assistant
        cxt.write_session()
        idx += 2
    assistants_dropdown = gr.Dropdown(choices=get_assistant_dropdown_list(), type='index',
                                      multiselect=False, show_label=False,
                                      value=get_assistant_dropdown_list()[idx],
                                      interactive=True)
    return assistants_dropdown


def clear_history():
    cxt.assistant_state.value = []
    cxt.current_assistant.temp_state = []
    return gr.Textbox(value=None, placeholder="什麼是LLM?", label="輸入文字後按enter", lines=5, max_lines=2000, scale=4)


def reset_textbox():
    return gr.Textbox(value=None, placeholder="什麼是LLM?", label="輸入文字後按enter", lines=5, max_lines=2000, scale=4)


def assistant_panel():
    state = gr.State([])
    cxt.assistant_state = state
    assistants = glob.glob('./images/assistants/*.png')

    with gr.Column(elem_id="col_container") as _panel:
        assistant_state = gr.State({})
        this_thread = None
        with gr.Row(variant="panel", elem_classes="screen_container", equal_height=True):
            with gr.Group():
                with gr.Column():
                    assistants_dropdown = gr.Dropdown(choices=get_assistant_dropdown_list(), type='index',
                                                      multiselect=False, show_label=False, interactive=True)
                    assistants_name = gr.Textbox(interactive=True, autoscroll=False, lines=1, max_lines=5,
                                                 min_width=240,
                                                 container=True, show_label=True, label='名稱')

                    with gr.Row():
                        assistants_id = gr.Textbox(interactive=False, autoscroll=False, lines=1, max_lines=5,
                                                   min_width=208,
                                                   container=True, scale=5, show_label=False)
                        retreive_assistant = gr.Button(size='sm', min_width=32, value='🔄', scale=1, visible=False)
                    assistants_instructions = gr.TextArea(elem_classes="instructions_text", interactive=True, lines=10,
                                                          min_width=240, container=True, show_label=True,
                                                          show_copy_button=True,
                                                          label='instructions')
                    assistants_model = gr.Dropdown(choices=get_model_dropdown_list(), type='value', interactive=True,
                                                   min_width=240, show_label=True, label='Models', visible=True)

                    tool_checkboxgroup = gr.Checkboxgroup(choices=["函數", "代碼解釋器", "知識庫"], label='工具')
                    assistants_files = gr.Dropdown(elem_classes="assistants_files", choices=[], type='value',
                                                   interactive=True, multiselect=True,
                                                   min_width=240, show_label=False, visible=True)
            with gr.Column(scale=3):
                with gr.Column():
                    chatbot = gr.Chatbot(value=message2chat, elem_id='chatbot', container=True, scale=4,
                                         render_markdown=True, min_width=550,
                                         avatar_images=["./images/avatar/human.png",
                                                        "./images/assistants/assistant01.png"],
                                         likeable=True,
                                         show_copy_button=True, bubble_full_width=True, show_share_button=True, every=1,
                                         layout="panel")
                    with gr.Group():
                        with gr.Row():
                            user_inputs = gr.Textbox(placeholder="什麼是LLM?",
                                                     label="輸入文字後按enter", lines=5, max_lines=2000, scale=4)  # t

                            with gr.Column():
                                with gr.Row(variant='panel'):
                                    b1 = gr.Button(value='▶️', interactive=True, size='sm', scale=1, min_width=64)
                                    b2 = gr.Button(value='⏹️', interactive=True, size='sm', scale=1, min_width=64)
                                with gr.Row(variant='compact'):
                                    b4 = gr.Button(value="💬", interactive=True, size='sm', scale=1, min_width=6, )
                                    b3 = gr.Button(value="🗑️", interactive=True, size='sm', scale=1, min_width=64)
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

    def dropdown_change(value, assistant_state):
        is_code = False
        is_retrieval = False
        is_function = False
        if value:
            if value == 0:
                assistants_name = gr.Textbox(interactive=True, autoscroll=False, lines=1, max_lines=5,
                                             min_width=240,
                                             container=True, show_label=True, label='名稱')
                retreive_assistant = gr.Button(size='sm', min_width=32, value='🔄', scale=1, visible=False)
                assistants_id = gr.Textbox(interactive=False, autoscroll=False, lines=1, max_lines=5,
                                           min_width=208,
                                           container=True, scale=5, show_label=False, visible=False)
                assistants_instructions = gr.TextArea(elem_classes="instructions_text", interactive=True, lines=5,
                                                      min_width=240,
                                                      container=True, show_label=True, label='instructions',
                                                      show_copy_button=True,
                                                      visible=True)
                assistants_model = gr.Dropdown(choices=get_model_dropdown_list(), type='value', interactive=True,
                                               min_width=240, show_label=True, label='Models', visible=True)
                tool_checkboxgroup = gr.Checkboxgroup(choices=["函數", "代碼解釋器", "知識庫"], label='工具',
                                                      visible=True)
                assistants_files = gr.Dropdown(choices=[], type='value', interactive=True, multiselect=True,
                                               min_width=240, show_label=False, visible=True)

            elif value == 1:
                assistants_name = gr.Textbox(interactive=False, autoscroll=False, lines=1, max_lines=5,
                                             min_width=240,
                                             container=True, show_label=True, label='名稱', value='', visible=False)
                retreive_assistant = gr.Button(size='sm', min_width=32, value='🔄', scale=1, visible=True)
                assistants_id = gr.Textbox(interactive=True, autoscroll=False, lines=1, max_lines=5,
                                           min_width=208,
                                           container=True, scale=5, show_label=False, value="")
                assistants_instructions = gr.TextArea(elem_classes="instructions_text", interactive=True, lines=10,
                                                      min_width=240,
                                                      container=True, show_label=True, label='instructions',
                                                      visible=False)
                assistants_model = gr.Dropdown(choices=get_model_dropdown_list(), type='value', interactive=True,
                                               min_width=240, show_label=True, label='Models', visible=False)

                tool_checkboxgroup = gr.Checkboxgroup(choices=["函數", "代碼解釋器", "知識庫"], label='工具',
                                                      visible=False)
                assistants_files = gr.Dropdown(choices=[], type='value', interactive=True, multiselect=True,
                                               min_width=240, show_label=False, visible=False)
            else:
                myassistant = cxt.assistants[value - 2]
                cxt.current_assistant = Assistant(assistant_id=myassistant.id)
                assistants_name = gr.Textbox(interactive=True, autoscroll=False, lines=1, max_lines=5,
                                             min_width=240,
                                             container=True, show_label=True, label='名稱', value=myassistant.name,
                                             visible=True)

                retreive_assistant = gr.Button(size='sm', min_width=32, value='🔄', scale=1, visible=False)
                assistants_id = gr.Textbox(interactive=False, autoscroll=False, lines=1, max_lines=5,
                                           min_width=208,
                                           container=True, scale=5, show_label=False, value=myassistant.id)
                assistants_instructions = gr.TextArea(elem_classes="instructions_text", interactive=True,
                                                      lines=10,
                                                      min_width=240,
                                                      container=True, show_label=True, label='instructions',
                                                      show_copy_button=True,
                                                      value=myassistant.instructions, visible=True)
                assistants_model = gr.Dropdown(choices=get_model_dropdown_list(), type='value', interactive=True,
                                               min_width=240, show_label=True, label='Models', visible=True,
                                               value=myassistant.model)
                tool_values = []
                for tool in myassistant.tools:
                    if tool.type == "retrieval":
                        tool_values.append("知識庫")
                    elif tool.type == "code_interpreter":
                        tool_values.append("代碼解釋器")
                    elif tool.type == "function":
                        tool_values.append("函數")

                tool_checkboxgroup = gr.Checkboxgroup(choices=["函數", "代碼解釋器", "知識庫"], value=tool_values,
                                                      label='工具', visible=True)
                assistants_files = gr.Dropdown(
                    choices=get_assistant_files(myassistant.id, assistant_state) if "知識庫" in tool_values else [],
                    value=get_assistant_files(myassistant.id, assistant_state) if "知識庫" in tool_values else [],
                    type='value',
                    interactive=True, multiselect=True,
                    min_width=240, show_label=False, visible=True)
        return assistant_state, assistants_name, retreive_assistant, assistants_id, assistants_instructions, assistants_model, tool_checkboxgroup, assistants_files

    def run_assistant(user_input, state):
        _, run = cxt.current_assistant.create_thread_and_run(user_input)
        cxt.assistant_state.value.append({"role": "user", "content": user_input})
        t = threading.Thread(target=cxt.current_assistant.wait_on_run, args=(run,))
        t.start()
        return state

    assistants_dropdown.change(dropdown_change, inputs=[assistants_dropdown, assistant_state],
                               outputs=[assistant_state, assistants_name, retreive_assistant, assistants_id,
                                        assistants_instructions, assistants_model, tool_checkboxgroup,
                                        assistants_files])
    retreive_assistant.click(fn=retrieve_assistant, inputs=[assistants_id],
                             outputs=[assistants_dropdown])
    chatbot.change(show_progress=False, scroll_to_output=True)
    # b2.click(fn=None, inputs=None, outputs=None, cancels=cancel_handles)
    b1.click(fn=run_assistant, inputs=[user_inputs, state], outputs=[state]).then(reset_textbox, [], [user_inputs])
    b3.click(fn=clear_history, inputs=[], outputs=[user_inputs])

    return _panel
