from prompt4all.api.base_api import GptBaseApi
import time
import os
import uuid
import json
import threading
from prompt4all import context
from prompt4all.context import *
from prompt4all.common import *
import gradio as gr
from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI, RequestOptions
from openai._types import NotGiven, NOT_GIVEN
from openai.types.beta.assistant import Assistant as openai_Assistant

client = OpenAI()
client._custom_headers['Accept-Language'] = 'zh-TW'
cxt = context._context()


class Assistant(GptBaseApi):
    def __init__(self, *args, **kwargs):
        self.API_MODEL = None
        if len(args) == 1:
            if isinstance(args[0], openai_Assistant):
                self.base = args[0]
            elif isinstance(args[0], str) and args[0].startswith('asst_'):
                self.assistant_id = args[0]
                self.base = client.beta.assistants.retrieve(self.assistant_id)
        else:
            raise RuntimeError('Assistant init error')
        self.assistant_id = self.base.id
        self.name = self.base.name
        super().__init__(self.base.model)
        self.change_model(self.base.model)

        self.description = self.base.description
        self.instructions = self.base.instructions
        self.tools = self.base.tools
        self.files = None
        t = threading.Thread(target=self.retrive_files)
        t.start()

        self.API_KEY = os.getenv("OPENAI_API_KEY")
        self.temp_state = []
        self.FULL_HISTORY = []

        self.BASE_IDENTITY = uuid.uuid4()
        self.current_thread = None
        self.current_run = None
        self.current_runsteps = None

    def retrive_files(self):
        assistant_files = client.beta.assistants.files.list(assistant_id=self.assistant_id)
        assistant_files = [client.files.retrieve(file.id) for file in assistant_files]
        self.files = assistant_files

    def create_thread(self):
        if not self.current_thread:
            _thread = client.beta.threads.create()
            self.current_thread = _thread
        return self.current_thread

    def create_message(self, client, user_message):
        return client.beta.threads.messages.create(
            thread_id=self.current_thread.id,
            role="user",
            content=user_message
        )

    def submit_message(self, user_message):
        client.beta.threads.messages.create(
            thread_id=self.current_thread.id, role="user", content=user_message
        )
        _runs = client.beta.threads.runs.create(
            thread_id=self.current_thread.id,
            assistant_id=self.assistant_id,
        )
        return _runs

    def get_message_text(self, message):
        if isinstance(message, str):
            message_id = message
            message = client.beta.threads.messages.retrieve(
                message_id=message_id,
                thread_id=self.current_thread.id,
            )
        message_content = message.content[0].text
        annotations = message_content.annotations
        citations = []

        # Iterate over the annotations and add footnotes
        for index, annotation in enumerate(annotations):
            # Replace the text with a footnote
            message_content.value = message_content.value.replace(annotation.text, f' [{index}]')

            # Gather citations based on annotation attributes
            if (file_citation := getattr(annotation, 'file_citation', None)):
                cited_file = client.files.retrieve(file_citation.file_id)
                citations.append(f'[{index}] {file_citation.quote} from {cited_file.filename}')
            elif (file_path := getattr(annotation, 'file_path', None)):
                cited_file = client.files.retrieve(file_path.file_id)
                citations.append(f'[{index}] 點選 <here> 下載 {cited_file.filename}')
                # Note: File download functionality not implemented above for brevity

        # Add footnotes to the end of the message before displaying to user
        return message_content.value + '\n' + '\n'.join(citations)

    def create_thread_and_run(self, user_input):
        if self.current_thread is None:
            _thread = client.beta.threads.create()
            self.current_thread = _thread

        run = self.submit_message(user_input)
        self.current_run = run
        return self.current_thread, self.current_run

    def wait_on_run(self, state):
        is_not_finished = True
        while is_not_finished:
            try:
                # state = self.wait_on_run(state)
                run = client.beta.threads.runs.retrieve(
                    run_id=self.current_run.id,
                    thread_id=self.current_thread.id,
                )

                self.current_run = run
                print('{0} {1}'.format(self.current_run.status, time.time()), flush=True)
                while self.current_run.status in ["queued", "in_progress", "requires_action"]:

                    self.temp_state = []
                    if self.current_run.status == "requires_action":
                        tool_outputs = []
                        for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                            if tool_call:
                                if tool_call.type == "function":
                                    name = tool_call.function.name
                                    self.temp_state.append(
                                        {"role": "status", "content": '使用工具{0}中...'.format(name)})
                                    arguments = json.loads(tool_call.function.arguments)
                                    tool_function = get_tool(tool_call.function.name)
                                    if tool_function:
                                        results = tool_function(**arguments)
                                        print(tool_call.function.name, arguments, yellow_color(results), flush=True)
                                        tool_outputs.append({
                                            "tool_call_id": tool_call.id,
                                            "output": results,
                                        })
                                    else:
                                        self.temp_state.append(
                                            {"role": "status", "content": '找不到對應工具:{0}'.format(name)})
                        self.current_run = client.beta.threads.runs.submit_tool_outputs(
                            run_id=run.id,
                            thread_id=self.current_thread.id,
                            tool_outputs=tool_outputs,
                        )
                    else:
                        run_steps = client.beta.threads.runs.steps.list(
                            thread_id=self.current_thread.id,
                            run_id=self.current_run.id, order="asc"
                        )
                        for step in run_steps.data:
                            step_details = step.step_details
                            if step.type == 'tool_calls':
                                for i in range(len(step_details.tool_calls)):
                                    tool_call = step_details.tool_calls[i]
                                    if not isinstance(tool_call, dict):
                                        tool_call = tool_call.__dict__
                                    if tool_call['type'] == 'code_interpreter':
                                        if step.status == 'completed':
                                            self.temp_state.append(
                                                {"role": "status",
                                                 "content": '撰寫代碼完成，撰寫回覆中...'})

                                            yield cxt.assistant_state.value
                                        else:
                                            self.temp_state.append(
                                                {"role": "status",
                                                 "content": '撰寫代碼中...'})

                                        yield cxt.assistant_state.value
                                    elif tool_call['type'] == 'retrieval':
                                        if step.status == 'completed':
                                            self.temp_state.append(
                                                {"role": "status",
                                                 "content": '知識庫查詢完成，撰寫回覆中...'})
                                            yield cxt.assistant_state.value
                                        else:
                                            self.temp_state.append(
                                                {"role": "status",
                                                 "content": '知識庫查詢中...'})
                                        # print(tool_call['type'], tool_call['retrieval'], step.status, flush=True)

                                        yield cxt.assistant_state.value
                                    elif tool_call['type'] == 'function':
                                        _tool_function = tool_call['function'].__dict__ if not isinstance(
                                            tool_call['function'],
                                            dict) else tool_call[
                                            'function']
                                        self.temp_state.append(
                                            {"role": "status",
                                             "content": '使用工具{0}中...'.format(_tool_function['name'])})
                                        yield cxt.assistant_state.value
                            elif step.type == 'message_creation' and step.status == 'completed':
                                self.temp_state.append(
                                    {"role": "status",
                                     "content": self.get_message_text(
                                         step_details.message_creation.message_id)})
                                yield cxt.assistant_state.value
                    time.sleep(1)
                    self.current_run = client.beta.threads.runs.retrieve(
                        run_id=self.current_run.id,
                        thread_id=self.current_thread.id,
                    )
                    print('{0} {1}'.format(self.current_run.status, time.time()), flush=True)

                if self.current_run.status == "completed":
                    self.temp_state = []
                    messages = client.beta.threads.messages.list(thread_id=self.current_thread.id,
                                                                 order="asc")
                    for message in messages.data:
                        if message.role == "assistant" and message.run_id == run.id:
                            cxt.assistant_state.value.append(
                                {"role": "assistant", "content": self.get_message_text(message)})
                    is_not_finished = False
                else:
                    is_not_finished = False
                    self.temp_state = []

            except StopIteration:
                break
            except Exception as e:
                response = client.beta.threads.delete(self.current_thread.id)
                PrintException()
                raise gr.Error(str(e))

        yield cxt.assistant_state.value
