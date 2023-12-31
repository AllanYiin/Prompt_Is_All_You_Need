from prompt4all.api.base_api import GptBaseApi
import time
import os
import uuid
import json
from prompt4all import context
from prompt4all.context import *
from prompt4all.common import *
from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI, RequestOptions
from openai._types import NotGiven, NOT_GIVEN

client = OpenAI()
client._custom_headers['Accept-Language'] = 'zh-TW'
cxt = context._context()


class Assistant(GptBaseApi):
    def __init__(self, assistant_id, name='MyGPTs', model="gpt-4-1106-preview", instruction=''):
        super().__init__(model)
        self.name = name
        self.assistant_id = assistant_id
        self.instruction = instruction
        self.API_MODEL = None
        self.API_KEY = os.getenv("OPENAI_API_KEY")
        self.temp_state = []
        self.FULL_HISTORY = []

        self.change_model(model)

        self.BASE_IDENTITY = uuid.uuid4()
        self.functions = NOT_GIVEN
        self.tools = NOT_GIVEN

        self.current_thread = None
        self.current_run = None
        self.current_runsteps = None

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

    def wait_on_run(self, run):

        self.current_run = run
        while run.status == "queued" or run.status == "in_progress" or run.status == "requires_action":
            self.temp_state = []
            if run.status == "requires_action":
                tool_outputs = []
                for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                    if tool_call.type == "function":
                        name = tool_call.function.name
                        self.temp_state.append({"role": "assistant", "content": '使用工具{0}中...'.format(name)})
                        arguments = json.loads(tool_call.function.arguments)
                        tool_function = get_tool(tool_call.function.name)

                        if tool_function:
                            results = tool_function(**arguments)
                            print(tool_call.function.name, arguments, yellow_color(results))
                            tool_outputs.append({
                                "tool_call_id": tool_call.id,
                                "output": results,
                            })
                        else:
                            self.temp_state.append({"role": "assistant", "content": '找不到對應工具:{0}'.format(name)})
                run = client.beta.threads.runs.submit_tool_outputs(
                    thread_id=self.current_thread.id,
                    run_id=run.id,
                    tool_outputs=tool_outputs,
                )
            else:
                run = client.beta.threads.runs.retrieve(
                    thread_id=self.current_thread.id,
                    run_id=run.id,
                )
                run_steps = client.beta.threads.runs.steps.list(
                    thread_id=self.current_thread.id, run_id=run.id, order="asc"
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
                                        {"role": "assistant",
                                         "content": '撰寫代碼完成...'})
                                else:
                                    self.temp_state.append(
                                        {"role": "assistant",
                                         "content": '撰寫代碼中...'})
                                print(tool_call['type'], tool_call['code_interpreter'], step.status, flush=True)
                            elif tool_call['type'] == 'retrieval':
                                if step.status == 'completed':
                                    self.temp_state.append(
                                        {"role": "assistant",
                                         "content": '知識庫查詢完成...'})
                                else:
                                    self.temp_state.append(
                                        {"role": "assistant",
                                         "content": '知識庫查詢中...'})
                                print(tool_call['type'], tool_call['retrieval'], step.status, flush=True)
                            elif tool_call['type'] == 'function':
                                _tool_function = tool_call['function'].__dict__ if not isinstance(tool_call['function'],
                                                                                                  dict) else tool_call[
                                    'function']
                                self.temp_state.append(
                                    {"role": "assistant",
                                     "content": '使用工具{0}中...'.format(_tool_function['name'])})
                                print(tool_call['type'], tool_call['function'], step.status, flush=True)
                            time.sleep(0.5)
                    elif step.type == 'message_creation' and step.status == 'completed':
                        self.temp_state.append(
                            {"role": "assistant",
                             "content": self.get_message_text(step_details.message_creation.message_id)})
            time.sleep(0.5)

        messages = client.beta.threads.messages.list(thread_id=self.current_thread.id, order="asc")
        self.temp_state = []
        for message in messages.data:
            if message.role == "assistant" and message.run_id == run.id:
                cxt.assistant_state.value.append({"role": "assistant", "content": self.get_message_text(message)})
