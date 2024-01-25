import glob
import json
import os
import uuid
import openai
from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI, RequestOptions
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, \
    ChatCompletionAssistantMessageParam, ChatCompletionToolMessageParam, ChatCompletionFunctionMessageParam, \
    ChatCompletionMessageToolCallParam
from openai._types import NotGiven, NOT_GIVEN
import asyncio
import requests
import copy
import threading
import prompt4all.api.context_type as ContextType
from prompt4all.utils.regex_utils import *
from prompt4all.common import *
from prompt4all.tools import database_tools, web_tools, diagram_tools
from prompt4all.utils.chatgpt_utils import process_chat
from prompt4all.utils.tokens_utils import estimate_used_tokens
from prompt4all import context
from prompt4all.context import *

cxt = context._context()

client = AzureOpenAI()
client._custom_headers['Accept-Language'] = 'zh-TW'

__all__ = ["GptBaseApi"]


class GptBaseApi:
    def __init__(self, model="gpt-4-1106-preview", temperature=0.5,
                 system_message='#zh-TW 請以繁體中文回答', enable_db=False):
        self.tools = []
        js = glob.glob("./tools/*.json")
        js.remove('./tools\\query_sql.json')
        js.remove('./tools\\code_interpreter.json')
        js.remove('./tools\\image_generation.json')
        self.temp_state = []
        self.tools = []
        for j in js:
            _tool = eval(open(j, encoding="utf-8").read())
            if isinstance(_tool, dict):
                self.tools.append(_tool)
            elif isinstance(_tool, list):
                self.tools.extend(_tool)

        self.API_MODEL = None
        self.API_TYPE = 'openai'
        self.BASE_URL = None
        self.client = None
        self.MAX_TOKENS = NOT_GIVEN
        self.API_KEY = os.getenv("OPENAI_API_KEY") if not 'azure' in model else os.getenv("AZURE_OPENAI_KEY")

        self.change_model(model)

        self.BASE_IDENTITY = uuid.uuid4()
        self.functions = NOT_GIVEN

        self.API_HEADERS = {
            'Accept': 'text/event-stream',
            'Accept-Language': 'zh-TW',
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.API_KEY}"
        }
        self.SYSTEM_MESSAGE = system_message
        self.API_PARAMETERS = {'top_p': 1, 'temperature': temperature, 'top_k': 1, 'presence_penalty': 0,
                               'frequency_penalty': 0}
        self.FULL_HISTORY = [{"role": "system", "content": self.SYSTEM_MESSAGE,
                              "estimate_tokens": estimate_used_tokens(self.SYSTEM_MESSAGE, model_name=self.api_model)}]

    @property
    def api_model(self):
        return self.API_MODEL

    @api_model.setter
    def api_model(self, value):
        self.change_model(value)

    @property
    def api_type(self):
        return openai.api_type

    @api_type.setter
    def api_type(self, value):
        openai.api_type = value

    def change_model(self, model="gpt-3.5-turbo-0613"):
        need_change = True
        if model.startswith('azure '):
            if self.API_TYPE and self.API_TYPE == 'azure' and model.replace('azure ', '') == self.API_MODEL:
                need_change = False
            else:
                self.API_MODEL = model.replace('azure ', '')
                self.API_TYPE = 'azure'
                openai.api_type = 'azure'
        else:
            if self.API_TYPE and self.API_TYPE == 'openai' and model == self.API_MODEL:
                need_change = False
            else:
                self.API_MODEL = model
                self.API_TYPE = 'openai'
                openai.api_type = 'openai'
        if need_change or not self.client:
            self.API_MODEL = model
            self.MAX_TOKENS = model_info[model]["max_token"]
            if "azure" in model:
                self.client = AzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_KEY"),
                    api_version="2023-10-01-preview",
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
                )
                self.API_KEY = os.getenv("AZURE_OPENAI_KEY"),
                self.BASE_URL = os.getenv("AZURE_OPENAI_ENDPOINT")
            else:
                self.API_KEY = os.getenv("OPENAI_API_KEY")
                self.BASE_URL = model_info[model]["endpoint"]
                self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY']
                                     )
            self.client._custom_headers['Accept-Language'] = 'zh-TW'
        self.enable_database_query(cxt.is_db_enable)

    def enable_database_query(self, is_enable: bool):

        if is_enable:
            # self.functions = [open("./tools/query_sql.json", encoding="utf-8").read()]
            self.tools.append({
                "type": "function",
                "function": {
                    "name": "query_sql",
                    "description": "將使用者查詢資料庫或者是取得某個彙總數據的需求轉成T-SQL後直接執行並回傳結果",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_intent": {
                                "type": "string",
                                "description": "使用者查詢資料庫或者是取得某個彙總數據的需求"
                            }
                        },
                        "required": ["query_intent"]
                    }
                },
            })

    def history2message_context(self, history):
        message_context = []
        for message in history:
            if message['role'] == 'system':
                _message = ChatCompletionSystemMessageParam(**{"content": message['content'], "role": "system"})
                message_context.append(_message)
            if message['role'] == 'assistant':
                args = {"content": message['content'], "role": "assistant"}
                if 'tool_calls' in message:
                    args['tool_calls'] = []
                    for toolcall in message['tool_calls']:
                        args['tool_calls'].append(ChatCompletionMessageToolCallParam(
                            **{"id": toolcall['id'], "type": toolcall['type'], "function": toolcall['function']}))
                _message = ChatCompletionAssistantMessageParam(**args)
                message_context.append(_message)
            elif message['role'] == 'user':
                _message = ChatCompletionUserMessageParam(**{"content": message['content'], "role": "user"})
                message_context.append(_message)
            elif message['role'] == 'tool':
                _message = ChatCompletionToolMessageParam(
                    **{"content": message['content'], "tool_call_id": message['tool_call_id'], 'name': message['name'],
                       "role": "tool"})
                message_context.append(_message)
        return message_context

    def build_message(self, role, content):
        """
        Build a chat message with the given role and content.

        Args:
        role (str): The role of the message sender, e.g., "system", "user", or "assistant".
        content (str): The content of the message.

        Returns:
        dict: A dictionary containing the role and content of the message.
        """
        return {"role": str(role), "content": str(content)}

    def process_context(self, prompt, context_type):
        # 確認每筆對話紀錄都有estimate_tokens，若無則計算寫入
        for i in range(len(cxt.state.value)):
            if 'estimate_tokens' not in cxt.state.value[i]:
                if cxt.state.value[i]['content'] is None:
                    cxt.state.value[i]['estimate_tokens'] = 4
                else:
                    cxt.state.value[i]['estimate_tokens'] = estimate_used_tokens(
                        cxt.state.value[i]['content']) + estimate_used_tokens(cxt.state.value[i]['role'],
                                                                              model_name=self.API_MODEL) + 4

        # 避免重複送出或是查詢時網路中斷
        if cxt.state.value[-1]['role'] == 'user' and cxt.state.value[-1]['content'] == prompt:
            cxt.state.value.pop(-1)
        # 最低需求量等於 本次prompt tokens+系統prompt tokens(除非 ContextType.sandbox)+200  預留輸出用
        this_input_tokens = estimate_used_tokens(prompt) + estimate_used_tokens('user', model_name=self.api_model) + 4
        if this_input_tokens + (
                cxt.state.value[0][
                    'estimate_tokens'] if context_type != ContextType.sandbox else 0) + 200 > self.MAX_TOKENS:
            raise ValueError('輸入prompt加上預留回覆總耗用tokens數為{0},超過模型上限{1}'.format(
                this_input_tokens + cxt.state.value[0]['estimate_tokens'] + 200, self.MAX_TOKENS))

        if context_type == ContextType.skip:
            message_context = [m for m in self.history2message_context(cxt.state.value) if m['role'] != 'system']

        elif context_type == ContextType.sandbox:
            message_context = []
        else:

            cxt.state.value.append({"role": "user", "content": prompt})
            remain_tokens = self.MAX_TOKENS - this_input_tokens - 200

            # estimate_tokens = sum([message['estimate_tokens'] for message in full_history]) + 2

            message_context = self.history2message_context(cxt.state.value)
            # message_context = [self.build_message(message['role'], message['content']) for message in cxt.state.value]
            # if estimate_tokens > remain_tokens:
            #     message_context = [self.build_message(message['role'], message['summary'] if message[
            #                                                                                      'role'] == 'assistant' and 'summary' in message and 'auto_continue' in message else
            #     message['content']) for message in full_history]
            #     estimate_tokens = sum([message['summary_tokens'] if message[
            #                                                             'role'] == 'assistant' and 'summary' in message and 'auto_continue' in message else
            #                            message['estimate_tokens'] for message in full_history]) + 2
            #     if estimate_tokens > remain_tokens:
            #         message_context = [self.build_message(message['role'], message['summary'] if message[
            #                                                                                          'role'] == 'assistant' and 'summary' in message else
            #         message['content']) for message in full_history]
            #         estimate_tokens = sum([message['summary_tokens'] if message[
            #                                                                 'role'] == 'assistant' and 'summary' in message else
            #                                message['estimate_tokens'] for message in full_history]) + 2
            #         if estimate_tokens > remain_tokens:
            #             message_context_tokens = [
            #                 message['summary_tokens'] if message['role'] == 'assistant' and 'summary' in message else
            #                 message['estimate_tokens'] for message in full_history]
            #             if len(message_context) >= 5 and sum(message_context_tokens[:3]) < remain_tokens:
            #                 while (sum(message_context_tokens) + 2 > remain_tokens):
            #                     remove_index = -1
            #                     for i in range(message_context):
            #                         if message_context[i]['role'] == 'assistant':
            #                             remove_index = i
            #                             break
            #                     if remove_index == -1:
            #                         for i in range(message_context):
            #                             if i > 1 and message_context[i]['role'] == 'user':
            #                                 remove_index = i
            #                                 break
            #                         if remove_index == -1:
            #                             break
            #                     message_context.pop(remove_index)
            #                     message_context_tokens.pop(remove_index)

        context_tokens = sum(
            [estimate_used_tokens(message['content']) + estimate_used_tokens(message['role']) + 4 for message in
             message_context]) + 2
        # with open(
        #         os.path.join('context_log', "{0}.json".format(int(datetime.now().timestamp()))),
        #         'w') as f:
        #     f.write(json.dumps({
        #         "message_context": message_context,
        #         "tokens": context_tokens
        #     }, ensure_ascii=False, indent=3))

        return message_context, context_tokens

    def parameters2payload(self, model, message_with_context, parameters, stream=True):
        payload = {
            "model": model,
            "messages": message_with_context,
            "temperature": parameters.get('temperature'),
            "top_p": parameters.get('top_p'),
            "n": parameters.get('top_k'),
            "stream": stream,
            "presence_penalty": parameters.get('presence_penalty'),
            "frequency_penalty": parameters.get('frequency_penalty')
        }

        return payload

    def make_response(self, model, message_with_context, parameters, stream=True):
        return self.client.chat.completions.create(
            model=model,
            messages=message_with_context,
            temperature=parameters.get('temperature'),
            top_p=parameters.get('top_p'),
            n=parameters.get('top_k', 1),
            max_tokens=parameters.get('max_tokens', NOT_GIVEN),
            presence_penalty=parameters.get('presence_penalty'),
            frequency_penalty=parameters.get('frequency_penalty'),
            stream=stream,
            tools=self.tools,
            tool_choice=NOT_GIVEN if self.tools == [] else "auto"

        )

    async def make_async_response(self, model, message_with_context, parameters, stream=False):
        self.functions = functions
        self.aclient = AsyncOpenAI()
        return await self.aclient.chat.completions.acreate(
            model=model,
            messages=message_with_context,
            temperature=parameters.get('temperature'),
            top_p=parameters.get('top_p'),
            n=parameters.get('top_k', 1),
            max_tokens=int(parameters.get('max_tokens'), NotGiven()),
            presence_penalty=parameters.get('presence_penalty'),
            frequency_penalty=parameters.get('frequency_penalty'),
            stream=stream,
            tools=self.tools,
            tool_choice=NOT_GIVEN if self.tools == [] else "auto"
        )

    def post_a_streaming_chat(self, input_prompt, context_type, parameters, state):
        """post 串流形式的對話

        Args:
            input_prompt:
            context_type:
            parameters:
            full_history:

        Returns:

        """
        full_history = cxt.state.value
        if context_type == ContextType.globals:
            full_history[0]["content"] = full_history[0]["content"] + '/n' + input_prompt
            full_history[0]["estimate_tokens"] = estimate_used_tokens(full_history[0]["content"],
                                                                      model_name=self.API_MODEL) + estimate_used_tokens(
                'system', model_name=self.API_MODEL) + 4

        elif context_type == ContextType.override:
            full_history[0]["content"] = input_prompt
            full_history[0]["estimate_tokens"] = estimate_used_tokens(input_prompt,
                                                                      model_name=self.API_MODEL) + estimate_used_tokens(
                'system', model_name=self.API_MODEL) + 4

        elif input_prompt and len(full_history) >= 3 and full_history[-1]['role'] == 'assistant' and full_history[-2][
            'role'] == 'user' and full_history[-2]['content'] == 'input_prompt':
            pass
        elif input_prompt:

            # 調用openai.ChatCompletion.create來生成機器人的回答
            estimate_tokens = estimate_used_tokens(input_prompt) + estimate_used_tokens('user',
                                                                                        model_name=self.API_MODEL) + 4
            message_context, context_tokens = self.process_context(input_prompt, context_type)
            partial_words = ''
            token_counter = 0

            cxt.citations = []
            self.temp_state.append({"role": "assistant", "content": partial_words, "context_type": context_type})
            completion = self.make_response(self.api_model, message_context, parameters, stream=True)

            tool_calls = []
            start = True
            finish_reason = 'None'
            try:
                self.temp_state = [s for s in self.temp_state if s['role'] != 'status']
                for chunk in completion:
                    try:
                        this_choice = chunk_message = chunk.choices[0]
                        this_delta = this_choice.delta
                        finish_reason = this_choice.finish_reason
                        if not this_delta:
                            break
                        elif this_delta and this_delta.content:
                            partial_words += this_delta.content
                            for i in range(len(self.temp_state)):
                                if self.temp_state[-i]['role'] == 'assistant':
                                    self.temp_state[-i]['content'] = partial_words
                                    break
                                yield full_history

                        if this_delta.tool_calls:
                            self.temp_state = [s for s in self.temp_state if s['role'] != 'status']
                            self.temp_state.append({"role": "status", "content": '解析使用工具需求...'})
                            for tool_call in this_delta.tool_calls:
                                index = tool_call.index
                                if index == len(tool_calls):
                                    tool_calls.append({})
                                if tool_call.id:
                                    tool_calls[index]['id'] = tool_call.id
                                    tool_calls[index]['type'] = 'function'
                                if tool_call.function:
                                    if 'function' not in tool_calls[index]:
                                        tool_calls[index]['function'] = {}
                                    if tool_call.function.name:
                                        tool_calls[index]['function']['name'] = tool_call.function.name
                                        tool_calls[index]['function']['arguments'] = ''
                                    if tool_call.function.arguments:
                                        tool_calls[index]['function']['arguments'] += (
                                            tool_call.function.arguments)
                                yield full_history

                        if finish_reason == 'stop':
                            break

                    except Exception as e:
                        finish_reason = '[EXCEPTION]'
                        if len(partial_words) == 0:
                            pass
                        else:
                            full_history[-1]['exception'] = str(e)
                        PrintException()
                        gr.Error(str(e))


            except Exception as e:
                finish_reason = '[EXCEPTION]'
                print(e)
                PrintException()
            # 檢查finish_reason是否為length
            print('finish_reason:', finish_reason, flush=True)
            while finish_reason == 'length':
                # 自動以user角色發出「繼續寫下去」的PROMPT
                prompt = "繼續"
                # 調用openai.ChatCompletion.create來生成機器人的回答
                message_context, context_tokens = self.process_context(prompt, context_type)
                # payload = self.parameters2payload(self.API_MODEL, message_context, self.API_PARAMETERS)
                completion2 = self.make_response(self.api_model, message_context, parameters, stream=True)
                # full_history[-1]['auto_continue'] = 1 if 'auto_continue' not in full_history[-1] else full_history[-1][
                #                                                                                           'auto_continue'] + 1
                finish_reason = 'None'

                for chunk in completion2:
                    try:
                        this_choice = chunk.choices[0]
                        this_delta = this_choice.delta
                        finish_reason = this_choice.finish_reason
                        # if (
                        #         'data: [DONE]' in this_choice):  # or (len(json.loads(chunk_decoded[6:])['choices'][0]["delta"]) == 0):
                        #     finish_reason = '[DONE]'
                        #     break

                        if this_choice.delta.content is not None:
                            partial_words += this_delta.content
                            for i in range(len(self.temp_state)):
                                if self.temp_state[-i]['role'] == 'assistant':
                                    self.temp_state[-i]['content'] = partial_words
                                    break
                            token_counter += 1
                        yield full_history
                    except Exception as e:
                        finish_reason = '[EXCEPTION]'
                        if len(partial_words) == 0:
                            pass
                        else:
                            full_history[-1]['exception'] = str(e)
                    yield full_history

            # 檢查接續後的完整回覆是否過長
            # print('bot_output: ',len(bot_output))

            while len(tool_calls) > 0:
                # Step 3: call the function
                # Note: the JSON response may not always be valid; be sure to handle errors

                cxt.state.value.append({
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': tool_calls
                })

                for tool_call in tool_calls:
                    function_name = tool_call['function']['name']
                    self.temp_state = [s for s in self.temp_state if s['role'] != 'status']
                    self.temp_state.append({"role": "status", "content": '使用工具:{0}中...'.format(function_name)})
                    try:
                        function_to_call = get_tool(function_name)

                        function_args = json.loads(tool_call['function']['arguments'])
                        yield full_history
                        print(blue_color('tool_call:{0}  {1}'.format(function_name, function_args)), flush=True)
                        function_response = function_to_call(**function_args)
                    except Exception as e:
                        function_response = str(e)
                    print('function_response', function_name, function_response, flush=True)
                    cxt.state.value.append(
                        {
                            "tool_call_id": tool_call['id'],
                            "role": "tool",
                            "name": function_name,
                            "content": function_response
                        }
                    )
                tool_calls = []
                second_response = self.client.chat.completions.create(
                    model=self.API_MODEL,
                    messages=self.history2message_context(cxt.state.value),
                    stream=True,
                    temperature=0.1,
                    n=1,
                    tools=self.tools,
                    tool_choice="auto"
                )
                is_placeholder = False
                placeholder_start_index = None
                self.temp_state = [s for s in self.temp_state if s['role'] != 'status']
                for second_chunk in second_response:
                    this_second_choice = second_chunk.choices[0]
                    this_second_delta = this_second_choice.delta
                    finish_reason = this_second_choice.finish_reason
                    if not this_second_delta:
                        break
                    elif this_second_delta and this_second_delta.content:
                        partial_words += this_second_delta.content
                        partial_words_without_placeholder = partial_words
                        if not is_placeholder:
                            if '@' in this_second_delta.content:
                                is_placeholder = True
                                placeholder_start_index = len(partial_words) - 1

                        else:
                            placeholder_candidate = partial_words[placeholder_start_index:]
                            if len(placeholder_candidate) <= len('@Placeholder(') and not '@Placeholder('.startswith(
                                    placeholder_candidate):
                                is_placeholder = False
                                placeholder_start_index = None
                            else:
                                if len(placeholder_candidate) < len('@Placeholder(') and '@Placeholder('.startswith(
                                        placeholder_candidate):
                                    partial_words_without_placeholder = partial_words[:placeholder_start_index]
                                else:
                                    maybe_placeholder = False
                                    for k in list(cxt.placeholder_lookup.keys()):
                                        lookup_key = '@Placeholder({0})'.format(k)
                                        if lookup_key == placeholder_candidate or lookup_key in placeholder_candidate:
                                            partial_words = partial_words.replace(lookup_key, cxt.placeholder_lookup[k])
                                            partial_words_without_placeholder = partial_words
                                            del cxt.placeholder_lookup[k]
                                            is_placeholder = False
                                            placeholder_start_index = None
                                            break
                                        elif lookup_key.startswith(placeholder_candidate):
                                            maybe_placeholder = True
                                            break
                                    if not maybe_placeholder:
                                        is_placeholder = False
                                        placeholder_start_index = None
                                    else:
                                        partial_words_without_placeholder = partial_words[:placeholder_start_index]

                        for i in range(len(self.temp_state)):
                            if self.temp_state[-i]['role'] == 'assistant':
                                self.temp_state[-i]['content'] = partial_words_without_placeholder
                                break
                        yield full_history
                        token_counter += 1

                    if this_second_delta.tool_calls:
                        self.temp_state = [s for s in self.temp_state if s['role'] != 'status']
                        self.temp_state.append({"role": "status", "content": '解析使用工具需求...'})
                        for tool_call in this_second_delta.tool_calls:
                            index = tool_call.index
                            if index == len(tool_calls):
                                tool_calls.append({})
                            if tool_call.id:
                                tool_calls[index]['id'] = tool_call.id
                                tool_calls[index]['type'] = 'function'
                            if tool_call.function:
                                if 'function' not in tool_calls[index]:
                                    tool_calls[index]['function'] = {}
                                if tool_call.function.name:
                                    tool_calls[index]['function']['name'] = tool_call.function.name
                                    tool_calls[index]['function']['arguments'] = ''
                                if tool_call.function.arguments:
                                    tool_calls[index]['function']['arguments'] += (
                                        tool_call.function.arguments)
                        yield full_history
                _placeholders = find_all_placeholders(partial_words)
                # print('找到{0}個佔位符'.format(len(_placeholders)), _placeholders)
                if len(_placeholders) > 0:
                    for _placeholder_id in _placeholders:
                        if _placeholder_id in cxt.placeholder_lookup:
                            partial_words = partial_words.replace('@Placeholder({0})'.format(_placeholder_id),
                                                                  cxt.placeholder_lookup[_placeholder_id])

                if len(cxt.citations) > 0:
                    partial_words = partial_words + '\n' + '\n'.join(cxt.citations)
                    print('citations', cyan_color('\n' + '\n'.join(cxt.citations)))
                cxt.citations = []

                if len(_placeholders) > 0:
                    for _placeholder_id in _placeholders:
                        if _placeholder_id in cxt.placeholder_lookup:
                            del cxt.placeholder_lookup[_placeholder_id]
                cxt.state.value.append(
                    {"role": "assistant", "content": partial_words,
                     "estimate_tokens": estimate_used_tokens(partial_words,
                                                             model_name=self.API_MODEL) + estimate_used_tokens(
                         'assistant', model_name=self.API_MODEL) + 4})
                if len(tool_calls) > 0:
                    partial_words = ''
                    self.temp_state.append(
                        {"role": "assistant", "content": partial_words, "context_type": context_type})
                else:
                    self.temp_state = []
                yield full_history

            if len(partial_words) > 200:
                def summerize_it(partial_words, **kwargs):
                    summarization_text = self.summarize_text(partial_words, 60)
                    _session = context._context()
                    for i in range(len(_session.state.value)):
                        if _session.state.value[-i]['role'] == 'assistant':
                            _session.state.value[-i]['summary'] = summarization_text
                            _session.state.value[-i]['summary_tokens'] = estimate_used_tokens(summarization_text)
                            break

                threading.Thread(target=summerize_it, args=(partial_words,)).start()
            yield full_history

    def post_and_get_streaming_answer(self, message_context, parameters, full_history=[]):
        """post 串流形式的對話

        :param message_context:
        :param parameters:
        :param full_history:
        :return:
        """

        partial_words = ''
        token_counter = 0
        context_type = ContextType.prompt
        # payload = self.parameters2payload(self.API_MODEL, message_context, parameters)
        try:
            if len(full_history) == 0:
                full_history = message_context
            completion = self.make_response(self.API_MODEL, message_context, parameters, stream=True)

            finish_reason = 'None'
            full_history.append({"role": "assistant", "content": partial_words, "context_type": context_type})
            for chunk in completion:
                try:
                    this_choice = chunk_message = chunk.choices[0]
                    this_delta = this_choice.delta
                    finish_reason = this_choice.finish_reason

                    if this_delta.content is not None:
                        partial_words += this_delta.content
                        full_history[-1]['content'] = partial_words
                        token_counter += 1
                except Exception as e:
                    if len(partial_words) == 0:
                        pass
                    else:
                        print('Exception', e)
                        finish_reason = '[EXCEPTION]'
                        full_history[-1]['exception'] = str(e)
                        break
                answer = full_history[-1]['content']
                yield answer, full_history
            while finish_reason != 'stop' and finish_reason != '[EXCEPTION]':
                # 自動以user角色發出「繼續寫下去」的PROMPT
                prompt = "繼續"
                # 調用openai.ChatCompletion.create來生成機器人的回答
                message_context, context_tokens = self.process_context(prompt, context_type)
                completion2 = self.make_response(self.API_MODEL, message_context, parameters, stream=True)

                for chunk in completion2:
                    try:
                        this_choice = chunk_message = chunk.choices[0]
                        this_delta = this_choice.delta
                        finish_reason = this_choice.finish_reason

                        if this_delta.content is not None:
                            partial_words += this_delta.content
                            full_history[-1]['content'] = partial_words
                            token_counter += 1
                    except Exception as e:
                        if len(partial_words) == 0:
                            pass
                        else:
                            finish_reason = '[EXCEPTION]'
                            full_history[-1]['exception'] = str(e)
                            break
                    answer = full_history[-1]['content']
                    yield answer, full_history

            full_history[-1]["estimate_tokens"] = estimate_used_tokens(partial_words, model_name=self.API_MODEL)
            answer = full_history[-1]['content']
            yield answer, full_history
        except Exception as e:
            print(e)
            PrintException()

    def summarize_text(self, text_input, timeout=120):
        """

        Args:
            text_input:
            timeout:

        Returns:

        """

        partial_words = ''
        token_counter = 0
        context_type = ContextType.skip
        conversation = [
            {
                "role": "system",
                "content": "你是萬能的文字助手，你擅長將任何輸入文字在保持原意不變，但必須保留[人名,公司機構名稱,事物名稱,地點,時間,數值,程式碼,數據集,陳述事實,知識點]前提下，作最精簡的摘要。"
            },
            {
                "role": "user",
                "content": text_input
            }
        ]
        paras = copy.deepcopy(self.API_PARAMETERS)
        paras['temperature'] = 1e-5
        aclient = AsyncOpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        async def make_async_response() -> None:
            chat_completion = await aclient.chat.completions.create(
                model=self.API_MODEL,
                messages=conversation,
                temperature=1e-5,
                stream=False,
            )
            return chat_completion

        completion = asyncio.run(make_async_response())
        return completion.choices[0].message.content

    def post_and_get_answer(self, message_context, parameters, full_history=None):
        """ 發問並獲取答案

        Args:
            message_context: 包含上下文以及本次問題之對話記錄
            parameters:
            full_history: 若為None，表示此次對話無須紀錄於對話歷史中

        Returns:

        """

        partial_words = ''
        token_counter = 0
        finish_reason = 'None'
        if full_history is not None:
            last_message = copy.deepcopy(message_context[-1])
            last_message["context_type"] = ContextType.prompt
            full_history.append(last_message)
        estimate_tokens = sum(
            [estimate_used_tokens(message['content']) + estimate_used_tokens(message['role']) + 4 for message in
             message_context]) + 2
        try:
            completion = self.make_response(self.API_MODEL, message_context, parameters, stream=False)
            return completion.choices[0].message.content
        except Exception as e:
            PrintException()

    def generate_images(self, input_prompt, shorter_prompt=None, image_size="1792x1024"):
        """

        Args:
            input_prompt:
            shorter_prompt:
            image_size:

        Returns:

        """
        response = openai.images.generate(
            model="dall-e-3",
            prompt=input_prompt,
            size=image_size,
            quality="standard",
            n=1,
        )

        response2 = openai.images.generate(
            model="dall-e-3",
            prompt=input_prompt,
            size=image_size,
            quality="standard",
            n=1,
        )

        images = []
        make_dir_if_need("./generate_images")
        image_file = "./generate_images/{0}-{1}.png".format(response.created, 0)
        if shorter_prompt is not None:
            image_file = "./generate_images/{0}-{1}-{2}.png".format(response.created,
                                                                    replace_special_chars(shorter_prompt), 0)
        images.append(image_file)
        img_data = requests.get(response.data[0].url).content
        with open(image_file, 'wb') as handler:
            handler.write(img_data)

        image_file = "./generate_images/{0}-{1}.png".format(response.created, 1)
        if shorter_prompt is not None:
            image_file = "./generate_images/{0}-{1}-{2}.png".format(response.created,
                                                                    replace_special_chars(shorter_prompt), 1)
        images.append(image_file)
        img_data = requests.get(response2.data[0].url).content
        with open(image_file, 'wb') as handler:
            handler.write(img_data)

            # image_data=cv2.imdecode(np.fromstring(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)

        return images

    def get_embedding(self, text):
        """

        Args:
            text:

        Returns:

        """
        text = text.replace("\n", " ")
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def save_history(self, filename=None):
        """

        Args:
            filename:

        Returns:

        """
        history_json = json.dumps(self.FULL_HISTORY, ensure_ascii=False, indent=4)

    def load_history(self, filename=None):
        """

        Args:
            filename:

        Returns:

        """
        history_json = json.dumps(self.FULL_HISTORY, ensure_ascii=False, indent=4)
