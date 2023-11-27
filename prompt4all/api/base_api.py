import json
import uuid
import os
import openai
from openai import OpenAI
from openai._types import NotGiven,NOT_GIVEN
import openai_async
from datetime import datetime
import asyncio
import nest_asyncio
import threading
from collections import defaultdict
nest_asyncio.apply()

import regex
import requests
import math
import sys
import copy
import numpy as np
import cv2
import PIL.Image as pil_image
from base64 import b64decode
import prompt4all.api.context_type as ContextType
from prompt4all.utils.regex_utils import *
from prompt4all.tools import database_tools
from prompt4all.utils.chatgpt_utils import process_context, process_chat
from prompt4all.utils.tokens_utils import num_tokens_from_history, estimate_used_tokens
#from tiktoken import Tokenizer, TokenizerWrapper
from typing import List, Dict, TypedDict
from prompt4all import context
from prompt4all.context import *
client = OpenAI()
cxt=context._context()

__all__ = ["model_info", "GptBaseApi"]

model_info = {
    # openai
    "gpt-3.5-turbo": {
        "endpoint": 'https://api.openai.com/v1/chat/completions',
        "max_token": 4096
    },
    "gpt-4-1106-preview": {
        "endpoint": 'https://api.openai.com/v1/chat/completions',
        "max_token": 128000
    },
    "gpt-4": {
        "endpoint": 'https://api.openai.com/v1/chat/completions',
        "max_token": 8192
    },

    "gpt-3.5-turbo-0613": {
        "endpoint": 'https://api.openai.com/v1/chat/completions',
        "max_token": 4096
    },
    "gpt-3.5-turbo-16k-0613": {
        "endpoint": 'https://api.openai.com/v1/chat/completions',
        "max_token": 16384
    },

    "gpt-4-0613": {
        "endpoint": 'https://api.openai.com/v1/chat/completions',
        "max_token": 8192
    },
    "gpt-4-0314": {
        "endpoint": 'https://api.openai.com/v1/chat/completions',
        "max_token": 8192
    },

    "gpt-4-32k": {
        "endpoint": 'https://api.openai.com/v1/chat/completions',
        "max_token": 32768
    },

    "gpt-4-32k-0314": {
        "endpoint": 'https://api.openai.com/v1/chat/completions',
        "max_token": 32768
    },
    "azure gpt-3.5-turbo": {
        "endpoint": 'https://prd-gpt-scus.openai.azure.com',
        "max_token": 4096
    },
    "azure 2023-03-15-preview": {
        "api_version":"2023-03-15-preview",
        "endpoint": 'https://ltc-to-openai.openai.azure.com/',
        "max_token": 4096
    },

    "azure gpt-4": {
        "endpoint": 'https://prd-gpt-scus.openai.azure.com',
        "max_token": 8192
    },

    "azure gpt-4-0314": {
        "endpoint": 'https://prd-gpt-scus.openai.azure.com',
        "max_token": 8192
    },

    "azure gpt-4-32k": {
        "endpoint": 'https://prd-gpt-scus.openai.azure.com',
        "max_token": 32768
    },

    "azure gpt-4-32k-0314": {
        "endpoint": 'https://prd-gpt-scus.openai.azure.com',
        "max_token": 32768
    }
}



# from typing import List, Literal, TypedDict
#
# MessageRole = Literal["system", "user", "assistant"]
# MessageType = Literal["ai_response", "action_result"]
#
#
# class Message(TypedDict):
#     role: MessageRole
#     content: str
#
#
# @dataclass
# class Message:
#     """OpenAI Message object containing a role and the message content"""
#
#     role: MessageRole
#     content: str
#     type: Union[MessageType ,None ]= None
#
#     def raw(self) -> MessageDict:
#         return {"role": self.role, "content": self.content}
#


class GptBaseApi:
    def __init__(self, model="gpt-4-1106-preview", temperature=0.5, system_message='#zh-TW 所有內容以繁體中文書寫'):
        self.API_MODEL=None
        self.API_TYPE=None
        self.BASE_URL =None
        self.MAX_TOKENS =NOT_GIVEN
        self.API_KEY = os.getenv("OPENAI_API_KEY")

        self.change_model(model)


        self.BASE_IDENTITY = uuid.uuid4()
        self.functions = NOT_GIVEN
        self.tools = NOT_GIVEN
        # if cxt.is_db_enable:
        #     self.functions = [open("tools/query_sql.json", encoding="utf-8").read()]
        #     self.tools = [
        #         {
        #             "type": "function",
        #             "function": {
        #                 "name": "query_sql",
        #                 "description": "將使用者查詢資料庫或者是取得某個彙總數據的需求轉成T-SQL後直接執行並回傳結果",
        #                 "parameters": {
        #                     "type": "object",
        #                     "properties": {
        #                         "query_intent": {
        #                             "type": "string",
        #                             "description": "使用者查詢資料庫或者是取得某個彙總數據的需求"
        #                         }
        #                     },
        #                     "required": ["query_intent"]
        #                 }
        #             },
        #         }
        #     ]

        self.enable_database_query(cxt.is_db_enable)

        self.API_HEADERS = {
            'Accept': 'text/event-stream',
            'Accept-Language': 'zh-TW',
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.API_KEY}"
        }
        self.SYSTEM_MESSAGE = system_message
        self.API_PARAMETERS = {'top_p': 1, 'temperature': temperature, 'top_k': 1, 'presence_penalty': 0.5,
                               'frequency_penalty': 0 }
        self.FULL_HISTORY = [{"role": "system", "content": self.SYSTEM_MESSAGE,
                              "estimate_tokens": estimate_used_tokens(self.SYSTEM_MESSAGE, model_name=self.api_model)}]
    @property
    def api_model(self):
        return self.API_MODEL

    @api_model.setter
    def api_model(self,value):
        self.change_model(value)

    @property
    def api_type(self):
        return openai.api_type

    @api_type.setter
    def api_type(self, value):
        openai.api_type=value
    def change_model(self,model="gpt-3.5-turbo-0613"):
        need_change=True
        if model.startswith('azure '):
            if self.API_TYPE is not None and self.API_TYPE =='azure' and model.replace('azure ','')==self.API_MODEL:
                need_change=False
            else:
                self.API_MODEL = model.replace('azure ','')
                self.API_TYPE ='azure'
                openai.api_type='azure'
        else:
            if self.API_TYPE is not None and self.API_TYPE =='openai' and model == self.API_MODEL:
                need_change = False
            else:
                self.API_MODEL = model
                self.API_TYPE = 'openai'
                openai.api_type = 'openai'
        if need_change:
            self.BASE_URL = model_info[model]["endpoint"]
            self.MAX_TOKENS = model_info[model]["max_token"]
            self.API_KEY = os.getenv("OPENAI_API_KEY")
        self.enable_database_query(cxt.is_db_enable)

    def enable_database_query(self, is_enable:bool):
        self.functions = NOT_GIVEN
        self.tools = NOT_GIVEN
        if is_enable:
            self.functions = [open("tools/query_sql.json", encoding="utf-8").read()]
            self.tools = [
                {
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
                }
            ]

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

    def process_context(self, prompt, context_type, full_history: list):
        #確認每筆對話紀錄都有estimate_tokens，若無則計算寫入
        for i in range(len(full_history)):
            if 'estimate_tokens' not in full_history[i]:
                full_history[i]['estimate_tokens']=estimate_used_tokens(full_history[i]['content']) + estimate_used_tokens(full_history[i]['role'],model_name=self.API_MODEL) + 4

        #避免重複送出或是查詢時網路中斷
        if full_history[-1]['role']=='user' and full_history[-1]['content']==prompt:
            full_history.pop(-1)
        #最低需求量等於 本次prompt tokens+系統prompt tokens(除非 ContextType.sandbox)+200  預留輸出用
        this_input_tokens = estimate_used_tokens(prompt) + estimate_used_tokens('user',model_name=self.api_model) + 4
        if this_input_tokens + (full_history[0]['estimate_tokens'] if context_type != ContextType.sandbox else 0) + 200 > self.MAX_TOKENS:
            raise ValueError('輸入prompt加上預留回覆總耗用tokens數為{0},超過模型上限{1}'.format(this_input_tokens + full_history[0]['estimate_tokens'] + 200, self.MAX_TOKENS))

        if context_type == ContextType.skip:
            message_context = [self.build_message(message['role'], message['content']) for message in full_history if
                               message['role'] == 'system']
        elif context_type == ContextType.sandbox:
            message_context = []
        else:
            remain_tokens = self.MAX_TOKENS - this_input_tokens - 200

            estimate_tokens = sum([message['estimate_tokens'] for message in full_history]) + 2
            message_context = [self.build_message(message['role'], message['content']) for message in full_history]
            if estimate_tokens > remain_tokens:
                message_context = [self.build_message(message['role'], message['summary'] if message[
                                                                                                 'role'] == 'assistant' and 'summary' in message and 'auto_continue' in message else message['content']) for message in full_history]
                estimate_tokens = sum([message['summary_tokens'] if message[
                                                                        'role'] == 'assistant' and 'summary' in message and 'auto_continue' in message else
                                       message['estimate_tokens'] for message in full_history]) + 2
                if estimate_tokens > remain_tokens:
                    message_context = [self.build_message(message['role'], message['summary'] if message[
                                                                                                     'role'] == 'assistant' and 'summary' in message else message['content']) for message in full_history]
                    estimate_tokens = sum([message['summary_tokens'] if message[
                                                                            'role'] == 'assistant' and 'summary' in message else
                                           message['estimate_tokens'] for message in full_history]) + 2
                    if estimate_tokens > remain_tokens:
                        message_context_tokens = [
                            message['summary_tokens'] if message['role'] == 'assistant' and 'summary' in message else
                            message['estimate_tokens'] for message in full_history]
                        if len(message_context) >= 5 and sum(message_context_tokens[:3]) < remain_tokens:
                            while (sum(message_context_tokens) + 2 > remain_tokens):
                                remove_index = -1
                                for i in range(message_context):
                                    if message_context[i]['role'] == 'assistant':
                                        remove_index = i
                                        break
                                if remove_index == -1:
                                    for i in range(message_context):
                                        if i > 1 and message_context[i]['role'] == 'user':
                                            remove_index = i
                                            break
                                    if remove_index == -1:
                                        break
                                message_context.pop(remove_index)
                                message_context_tokens.pop(remove_index)
        message_context.append({"role": "user", "content": prompt})
        context_tokens=sum(
                    [estimate_used_tokens(message['content']) + estimate_used_tokens(message['role']) + 4 for message in
                     message_context]) + 2
        # with open(
        #         os.path.join('context_log', "{0}.json".format(int(datetime.now().timestamp()))),
        #         'w') as f:
        #     f.write(json.dumps({
        #         "message_context": message_context,
        #         "tokens": context_tokens
        #     }, ensure_ascii=False, indent=3))

        return message_context,context_tokens

    def parameters2payload(self, model, message_with_context, parameters,stream=True):
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
        return client.chat.completions.create(
            model=model,
            messages=message_with_context,
            temperature=parameters.get('temperature'),
            top_p=parameters.get('top_p'),
            n=parameters.get('top_k',1),
            max_tokens=parameters.get('max_tokens', NOT_GIVEN),
            presence_penalty=parameters.get('presence_penalty'),
            frequency_penalty=parameters.get('frequency_penalty'),
            stream=stream,
            tools=self.tools,
            tool_choice=NOT_GIVEN if self.tools==NOT_GIVEN else "auto"

        )

    async def make_async_response(self, model, message_with_context, parameters, stream=False):
        self.functions = functions
        return await client.chat.completions.acreate(
            model=model,
            messages=message_with_context,
            temperature=parameters.get('temperature'),
            top_p=parameters.get('top_p'),
            n=parameters.get('top_k',1),
            max_tokens=int(parameters.get('max_tokens'),NotGiven()),
            presence_penalty=parameters.get('presence_penalty'),
            frequency_penalty=parameters.get('frequency_penalty'),
            stream=stream,
            tools=self.tools,
            tool_choice=NOT_GIVEN if self.tools==NOT_GIVEN else "auto"
        )

    def post_a_streaming_chat(self, input_prompt, context_type, parameters, full_history):
        """post 串流形式的對話

        :param input_prompt:
        :param context_type:
        :param parameters:
        :param full_history:
        :return:
        """
        if context_type == ContextType.globals:
            full_history[0]["content"] = full_history[0]["content"] + '/n' + input_prompt
            full_history[0]["estimate_tokens"] = estimate_used_tokens(full_history[0]["content"],
                                                                      model_name=self.API_MODEL) + estimate_used_tokens(
                'system', model_name=self.API_MODEL) + 4

        elif context_type == ContextType.override:
            full_history[0]["content"] = input_prompt
            full_history[0]["estimate_tokens"] = estimate_used_tokens(input_promp,
                                                                      model_name=self.API_MODEL) + estimate_used_tokens(
                'system', model_name=self.API_MODEL) + 4

        elif input_prompt and len(full_history)>=3 and full_history[-1]['role']=='assistant' and full_history[-2]['role']=='user' and full_history[-2]['content']=='input_prompt':
            pass
        elif input_prompt:
            status_word='...執行中'
            # 調用openai.ChatCompletion.create來生成機器人的回答
            estimate_tokens = estimate_used_tokens(input_prompt) + estimate_used_tokens('user',
                                                                                        model_name=self.API_MODEL) + 4
            message_context,context_tokens = self.process_context(input_prompt, context_type, full_history)
            partial_words = ''
            token_counter = 0
            #payload = self.parameters2payload(self.API_MODEL, message_context,parameters)
            full_history.append({"role": "user", "content": input_prompt, "context_type": context_type,
                                 "estimate_tokens": estimate_tokens})
            status_word = '執行中...'

            completion =self.make_response(self.api_model,message_context,parameters,stream=True)
            fake_full_history=copy.deepcopy(full_history)
            fake_full_history.append({"role": "assistant", "content": status_word, "context_type": context_type})
            chat = [(process_chat(fake_full_history[i]), process_chat(fake_full_history[i + 1])) for i in
                    range(1, len(fake_full_history) - 1, 2) if fake_full_history[i]['role'] != 'system']

            yield chat, status_word, full_history

            tool_calls = []
            start = True
            finish_reason = 'None'
            try:
                full_history.append({"role": "assistant", "content": partial_words, "context_type": context_type})
                for chunk in completion:
                    try:
                        this_choice = chunk_message = chunk.choices[0]
                        this_delta = this_choice.delta
                        finish_reason = this_choice.finish_reason
                        if not this_delta:
                            break
                        elif this_delta and this_delta.content:
                            partial_words += this_delta.content
                            full_history[-1]['content'] = status_word if len(partial_words)<5 else partial_words
                            token_counter += 1

                        if not this_delta.function_call and not this_delta.tool_calls:
                            if start:
                                continue
                            else:
                                break
                        start = False
                        if this_delta.function_call:
                            if index == len(tool_calls):
                                tool_calls.append({})
                            if delta.function_call.name:
                                tool_calls[index]['function']['name'] = delta.function_call.name
                                tool_calls[index]['function']['arguments']=''
                                status_word="解析查詢需求..."
                                full_history[-1]['content'] = status_word if len(partial_words) < 5 else partial_words
                            if delta.function_call.arguments:
                                tool_calls[index]['function']['arguments'] += (
                                    delta.function_call.arguments)
                        elif this_delta.tool_calls:
                            tool_call = this_delta.tool_calls[0]
                            index = tool_call.index
                            if index == len(tool_calls):
                                tool_calls.append({})
                            if tool_call.id:
                                tool_calls[index]['id'] = tool_call.id
                                tool_calls[index]['type']= 'function'
                            if tool_call.function:
                                if 'function' not in tool_calls[index]:
                                    tool_calls[index]['function'] = {}
                                if tool_call.function.name:
                                    tool_calls[index]['function']['name'] = tool_call.function.name
                                    tool_calls[index]['function']['arguments']=''
                                    status_word = "解析查詢需求..."
                                    full_history[-1]['content'] = status_word if len(partial_words) < 5 else partial_words
                                if tool_call.function.arguments:
                                    tool_calls[index]['function']['arguments'] += (
                                        tool_call.function.arguments)
                        chat = [(process_chat(full_history[i]), process_chat(full_history[i + 1])) for i in
                                range(1, len(full_history) - 1, 2) if full_history[i]['role'] != 'system']
                        answer = full_history[-1]['content']
                        yield chat, answer, full_history

                        if finish_reason == 'stop':
                            break

                    except Exception as e:
                        finish_reason = '[EXCEPTION]'
                        if len(partial_words) == 0:
                            pass
                        else:
                            full_history[-1]['exception'] = str(e)
                        PrintException()
                    chat = [(process_chat(full_history[i]), process_chat(full_history[i + 1])) for i in
                            range(1, len(full_history) - 1, 2) if full_history[i]['role'] != 'system']
                    answer = full_history[-1]['content']

                    yield chat, answer, full_history
                print('')
            except Exception as e:
                finish_reason = '[EXCEPTION]'
                print(e)
                PrintException()
            # 檢查finish_reason是否為length
            while finish_reason =='length' :
                # 自動以user角色發出「繼續寫下去」的PROMPT
                prompt = "繼續"
                # 調用openai.ChatCompletion.create來生成機器人的回答
                message_context,context_tokens = self.process_context(prompt, context_type, full_history)
                #payload = self.parameters2payload(self.API_MODEL, message_context, self.API_PARAMETERS)
                completion2 =self.make_response(self.api_model,message_context,parameters,stream=True)
                full_history[-1]['auto_continue'] = 1 if 'auto_continue' not in full_history[-1] else full_history[-1][
                                                                                                          'auto_continue'] + 1
                finish_reason = 'None'

                for chunk in completion2:
                    try:
                        this_choice = chunk_message = chunk.choices[0]
                        this_delta = this_choice.delta
                        finish_reason = this_choice.finish_reason
                        # if (
                        #         'data: [DONE]' in this_choice):  # or (len(json.loads(chunk_decoded[6:])['choices'][0]["delta"]) == 0):
                        #     finish_reason = '[DONE]'
                        #     break

                        if this_choice.delta.content is not None:
                            partial_words += this_delta.content
                            full_history[-1]['content'] = partial_words
                            token_counter += 1
                    except Exception as e:
                        finish_reason = '[EXCEPTION]'
                        if len(partial_words) == 0:
                            pass
                        else:
                            full_history[-1]['exception'] = str(e)

                    chat = [(process_chat(full_history[i]), process_chat(full_history[i + 1])) for i in
                            range(1, len(full_history) - 1, 2) if full_history[i]['role'] != 'system']
                    answer = full_history[-1]['content']

                    yield chat, answer, full_history

            # 檢查接續後的完整回覆是否過長
            # print('bot_output: ',len(bot_output))


            if tool_calls:
                # Step 3: call the function
                # Note: the JSON response may not always be valid; be sure to handle errors
                available_functions = {
                    "query_sql": database_tools.query_sql,
                }  # only one function in this example, but you can have multiple

                #message_context.append({"role": "assistant", "content":  full_history[-1]['content'], 'tool_calls':tool_calls})
                status_word = "生成SQL語法..."
                full_history[-1]['content'] = status_word
                chat = [(process_chat(full_history[i]), process_chat(full_history[i + 1])) for i in
                        range(1, len(full_history) - 1, 2) if full_history[i]['role'] != 'system']
                answer = full_history[-1]['content']
                yield chat, answer, full_history
                for tool_call in tool_calls:
                    function_name = tool_call['function']['name']

                    function_to_call = available_functions[function_name]
                    function_args = json.loads(tool_call['function']['arguments'])

                    function_response = function_to_call(
                        query_intent=function_args.get("query_intent")
                    )


                    message_context.append({'role': 'assistant', 'content': None,'tool_calls': tool_calls})
                    message_context.append({"role": "tool", "tool_call_id": tool_call['id'],"name": tool_call["function"]["name"],"content": function_response})

                    status_word = "執行資料庫查詢..."
                    full_history[-1]['content'] = status_word
                    chat = [(process_chat(full_history[i]), process_chat(full_history[i + 1])) for i in
                            range(1, len(full_history) - 1, 2) if full_history[i]['role'] != 'system']
                    answer = full_history[-1]['content']
                    yield chat, answer, full_history
                    # message_context.append(
                    #     {
                    #         "tool_call_id": tool_call['id'],
                    #         "role": "tool",
                    #         "name": function_name,
                    #         "content": eval(function_response),
                    #     }
                    # )
                    #full_history.append(message_context[-1])
                    # if message_context[-1]['content'] is None:
                    #     message_context[-1]['content']=''
                    # message_context[-1]['content']+='\n'+function_response
                    second_response = client.chat.completions.create(
                        model=self.api_model,
                        messages=message_context,
                        stream=True)
                    for second_chunk in second_response:

                        this_second_choice = chunk_message = second_chunk.choices[0]
                        this_second_delta = this_second_choice.delta
                        finish_reason = this_second_choice.finish_reason
                        if not this_second_delta:
                            break
                        elif this_second_delta and this_second_delta.content:
                            partial_words += this_second_delta.content
                            full_history[-1]['content'] = partial_words
                            token_counter += 1
                            chat = [(process_chat(full_history[i]), process_chat(full_history[i + 1])) for i in
                                    range(1, len(full_history) - 1, 2) if full_history[i]['role'] != 'system']
                            answer = full_history[-1]['content']

                            yield chat, answer, full_history
            full_history[-1]["estimate_tokens"] = estimate_used_tokens(partial_words,
                                                                       model_name=self.API_MODEL) + estimate_used_tokens(
                'assistant', model_name=self.API_MODEL) + 4
            chat = [(process_chat(full_history[i]), process_chat(full_history[i + 1])) for i in
                    range(1, len(full_history) - 1, 2) if full_history[i]['role'] in ['user','assistant']]
            answer = full_history[-1]['content']

            yield chat, answer, full_history

            # if len(partial_words) > 200:
            #
            #     summarization_text =self.summarize_text(partial_words,60)
            #     full_history[-1]['summary'] = summarization_text
            #     full_history[-1]['summary_tokens'] = estimate_used_tokens(summarization_text,model_name=self.API_MODEL) + estimate_used_tokens('assistant',model_name=self.API_MODEL) + 4

            full_history[-1]["estimate_tokens"] = estimate_used_tokens(partial_words,
                                                                       model_name=self.API_MODEL) + estimate_used_tokens(
                'assistant', model_name=self.API_MODEL) + 4
            chat = [(process_chat(full_history[i]), process_chat(full_history[i + 1])) for i in
                    range(1, len(full_history) - 1, 2) if full_history[i]['role'] != 'system']
            answer = full_history[-1]['content']

            yield chat, answer, full_history

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
        #payload = self.parameters2payload(self.API_MODEL, message_context, parameters)
        try:
            if len(full_history)==0:
                full_history = message_context
            completion = self.make_response(self.API_MODEL, message_context, parameters, stream=True)

            finish_reason = 'None'
            full_history.append( {"role": "assistant", "content": partial_words, "context_type": context_type})
            for chunk in completion:
                try:
                    this_choice=chunk_message = chunk.choices[0]
                    this_delta= this_choice.delta
                    finish_reason = this_choice.finish_reason

                    if this_delta.content is not None:
                        partial_words +=this_delta.content
                        full_history[-1]['content'] = partial_words
                        token_counter += 1
                except Exception as e:
                    if len(partial_words) == 0:
                        pass
                    else:
                        print('Exception',e)
                        finish_reason = '[EXCEPTION]'
                        full_history[-1]['exception'] = str(e)
                        break
                answer = full_history[-1]['content']
                yield answer, full_history
            while finish_reason != 'stop' and finish_reason != '[EXCEPTION]':
                # 自動以user角色發出「繼續寫下去」的PROMPT
                prompt = "繼續"
                # 調用openai.ChatCompletion.create來生成機器人的回答
                message_context,context_tokens = self.process_context(prompt, context_type, full_history)
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



    def summarize_text(self, text_input,timeout=120):
        """post 串流形式的對話
        :param text_input:
        :param timeout:
        :return:
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
        paras=copy.deepcopy(self.API_PARAMETERS)
        paras['temperature']=1e-5
        completion = self.make_response(self.API_MODEL, conversation, paras, stream=False)
        return completion.choices[0].message.content


    def post_and_get_answer(self, message_context, parameters, full_history=None):
        """發問並獲取答案
        :param message_context:包含上下文以及本次問題之對話記錄
        :param parameters:
        :param full_history:若為None，表示此次對話無須紀錄於對話歷史中
        :return:
        """

        # 調用openai.ChatCompletion.create來生成機器人的回答

        partial_words = ''
        token_counter = 0
        finish_reason = 'None'
        if full_history is not None:
            last_message=copy.deepcopy(message_context[-1])
            last_message["context_type"]=ContextType.prompt
            full_history.append(last_message)
        estimate_tokens = sum(
            [estimate_used_tokens(message['content']) + estimate_used_tokens(message['role']) + 4 for message in
             message_context]) + 2
        try:
            completion =self.make_response(self.API_MODEL, message_context, parameters,stream=False)
            return completion.choices[0].message.content
        except Exception as e:
            PrintException()



    def generate_images(self, input_prompt, shorter_prompt=None, image_size=1024):

        response = openai.images.generate(
            model="dall-e-3",
            prompt=input_prompt,
            size="{0}x{1}".format(image_size, image_size),
            quality="standard",
            n=1,
        )

        response2 = openai.images.generate(
            model="dall-e-3",
            prompt=input_prompt,
            size="{0}x{1}".format(image_size, image_size),
            quality="standard",
            n=1,
        )


        images = []


        image_file = "generate_images/{0}-{1}.png".format(response.created, 0)
        if shorter_prompt is not None:
            image_file = "generate_images/{0}-{1}-{2}.png".format(response.created,
                                                                  replace_special_chars(shorter_prompt), 0)
        images.append(image_file)
        img_data = requests.get(response.data[0].url).content
        with open(image_file, 'wb') as handler:
            handler.write(img_data)

        image_file = "generate_images/{0}-{1}.png".format(response.created, 1)
        if shorter_prompt is not None:
            image_file = "generate_images/{0}-{1}-{2}.png".format(response.created,
                                                                  replace_special_chars(shorter_prompt), 1)
        images.append(image_file)
        img_data = requests.get(response2.data[0].url).content
        with open(image_file, 'wb') as handler:
            handler.write(img_data)



            # image_data=cv2.imdecode(np.fromstring(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)


        return images

    def get_embedding(self,text):
        text = text.replace("\n", " ")
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def save_history(self, filename=None):
        history_json = json.dumps(self.FULL_HISTORY, ensure_ascii=False, indent=4)

    def load_history(self, filename=None):
        history_json = json.dumps(self.FULL_HISTORY, ensure_ascii=False, indent=4)
