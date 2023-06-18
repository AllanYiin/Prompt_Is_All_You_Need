import json
import uuid
import os
import openai
import openai_async
from datetime import datetime
import asyncio
import nest_asyncio

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
from utils.regex_utils import *
from utils.chatgpt_utils import process_context, process_chat
from utils.tokens_utils import num_tokens_from_history, estimate_used_tokens
#from tiktoken import Tokenizer, TokenizerWrapper
from typing import List, Dict, TypedDict


__all__ = ["model_info", "GptBaseApi"]

model_info = {
    # openai
    "gpt-3.5-turbo": {
        "endpoint": 'https://api.openai.com/v1/chat/completions',
        "max_token": 4096
    },

    "gpt-4": {
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
    def __init__(self, model="gpt-3.5-turbo", temperature=0.5, system_message='所有內容以繁體中文書寫'):
        if model.startswith('azure '):
            self.API_MODEL = model.replace('azure ','')
            self.API_TYPE ='azure'
            openai.api_type='azure'
        else:
            self.API_MODEL = model

        self.BASE_URL = model_info[model]["endpoint"]
        self.MAX_TOKENS = model_info[model]["max_token"]
        self.BASE_IDENTITY = uuid.uuid4()
        self.API_KEY = os.getenv("OPENAI_API_KEY")

        self.API_HEADERS = {
            'Accept': 'text/event-stream',
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.API_KEY}"
        }
        self.SYSTEM_MESSAGE = system_message
        self.API_PARAMETERS = {'top_p': 1, 'temperature': temperature, 'top_k': 1, 'presence_penalty': 0,
                               'frequency_penalty': 0, 'max_tokens': 2500}
        self.FULL_HISTORY = [{"role": "system", "content": self.SYSTEM_MESSAGE,
                              "estimate_tokens": estimate_used_tokens(self.SYSTEM_MESSAGE, model_name=self.API_MODEL)}]

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
        this_input_tokens = estimate_used_tokens(prompt) + estimate_used_tokens('user',model_name=self.API_MODEL) + 4
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
            # 調用openai.ChatCompletion.create來生成機器人的回答
            estimate_tokens = estimate_used_tokens(input_prompt) + estimate_used_tokens('user',
                                                                                        model_name=self.API_MODEL) + 4
            message_context,context_tokens = self.process_context(input_prompt, context_type, full_history)
            partial_words = ''
            token_counter = 0
            payload = self.parameters2payload(self.API_MODEL, message_context, self.API_PARAMETERS)
            full_history.append({"role": "user", "content": input_prompt, "context_type": context_type,
                                 "estimate_tokens": estimate_tokens})
            response = requests.post(self.BASE_URL, headers=self.API_HEADERS, json=payload, stream=True)

            finish_reason = 'None'
            try:
                for chunk in response.iter_lines():
                    try:
                        chunk_decoded = chunk.decode()
                        if ( 'data: [DONE]' in chunk_decoded):  # or (len(json.loads(chunk_decoded[6:])['choices'][0]["delta"]) == 0):
                            finish_reason = '[DONE]'
                            break
                        this_choice = json.loads(chunk_decoded[6:])['choices'][0]
                        finish_reason = this_choice['finish_reason']

                        if 'content' in this_choice['delta']:
                            partial_words += this_choice['delta']['content']

                            if token_counter == 0:
                                full_history.append(
                                    {"role": "assistant", "content": partial_words, "context_type": context_type})
                            else:
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
            except Exception as e:
                finish_reason = '[EXCEPTION]'
                print(e)
            # 檢查finish_reason是否為length
            while finish_reason != '[DONE]' and finish_reason != '[EXCEPTION]':
                # 自動以user角色發出「繼續寫下去」的PROMPT
                prompt = "繼續"
                # 調用openai.ChatCompletion.create來生成機器人的回答
                message_context,context_tokens = self.process_context(prompt, context_type, full_history)
                payload = self.parameters2payload(self.API_MODEL, message_context, self.API_PARAMETERS)
                response = requests.post(self.BASE_URL, headers=self.API_HEADERS, json=payload, stream=True)
                full_history[-1]['auto_continue'] = 1 if 'auto_continue' not in full_history[-1] else full_history[-1][
                                                                                                          'auto_continue'] + 1
                finish_reason = 'None'

                for chunk in response.iter_lines():
                    try:
                        chunk_decoded = chunk.decode()
                        if ('data: [DONE]' in chunk_decoded):  # or (len(json.loads(chunk_decoded[6:])['choices'][0]["delta"]) == 0):
                            finish_reason = '[DONE]'
                            break
                        this_choice = json.loads(chunk_decoded[6:])['choices'][0]
                        finish_reason = this_choice['finish_reason']

                        if 'content' in this_choice['delta']:
                            partial_words += this_choice['delta']['content']

                            if token_counter == 0:
                                full_history.append(
                                    {"role": "assistant", "content": partial_words, "context_type": context_type})
                            else:
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
            full_history[-1]["estimate_tokens"] = estimate_used_tokens(partial_words,
                                                                       model_name=self.API_MODEL) + estimate_used_tokens(
                'assistant', model_name=self.API_MODEL) + 4
            chat = [(process_chat(full_history[i]), process_chat(full_history[i + 1])) for i in
                    range(1, len(full_history) - 1, 2) if full_history[i]['role'] != 'system']
            answer = full_history[-1]['content']

            yield chat, answer, full_history

            if len(partial_words) > 200:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                tasks = [asyncio.ensure_future(self.summarize_text(text_input=partial_words, timeout=60))]
                loop.run_until_complete(asyncio.wait(tasks))
                response =tasks[0].result()
                summarization_text = response["content"].strip()
                full_history[-1]['summary'] = summarization_text
                full_history[-1]['summary_tokens'] = response["total_tokens"] + estimate_used_tokens('assistant',model_name=self.API_MODEL) + 4

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
        payload = self.parameters2payload(self.API_MODEL, message_context, self.API_PARAMETERS)
        try:
            if len(full_history)==0:
                full_history = message_context
            else:
                full_history.append(message_context[-1])
            response = requests.post(self.BASE_URL, headers=self.API_HEADERS, json=payload, stream=True)

            finish_reason = 'None'

            for chunk in response.iter_lines():
                try:
                    chunk_decoded = chunk.decode()
                    if 'data: [DONE]' in chunk_decoded:  # or (len(json.loads(chunk_decoded[6:])['choices'][0]["delta"]) == 0):
                        finish_reason = '[DONE]'
                        break
                    this_choice = json.loads(chunk_decoded[6:])['choices'][0]
                    finish_reason = this_choice['finish_reason']

                    if 'content' in this_choice['delta']:
                        partial_words += this_choice['delta']['content']

                        if token_counter == 0:
                            full_history.append(
                                {"role": "assistant", "content": partial_words, "context_type": context_type})
                        else:
                            full_history[-1]['content'] = partial_words
                        token_counter += 1

                except Exception as e:
                    finish_reason = '[EXCEPTION]'
                    if len(partial_words) == 0:
                        pass
                    else:
                        full_history[-1]['exception'] = str(e)

                answer = full_history[-1]['content']

                yield answer, full_history

            # 檢查finish_reason是否為length
            while finish_reason != '[DONE]' and finish_reason != '[EXCEPTION]':
                # 自動以user角色發出「繼續寫下去」的PROMPT
                prompt = "繼續"
                # 調用openai.ChatCompletion.create來生成機器人的回答
                payload = self.parameters2payload(self.API_MODEL, message_context, self.API_PARAMETERS)
                response = requests.post(self.BASE_URL, headers=self.API_HEADERS, json=payload, stream=True)

                finish_reason = 'None'
                for chunk in response.iter_lines():

                    try:

                        chunk_decoded = chunk.decode()
                        if ( 'data: [DONE]' in chunk_decoded):  # or (len(json.loads(chunk_decoded[6:])['choices'][0]["delta"]) == 0):
                            finish_reason = '[DONE]'
                            break
                        this_choice = json.loads(chunk_decoded[6:])['choices'][0]
                        finish_reason = this_choice['finish_reason']
                        if 'content' in this_choice['delta']:
                            partial_words += this_choice['delta']['content']

                            if token_counter == 0:
                                full_history.append(
                                    {"role": "assistant", "content": partial_words, "context_type": context_type})
                            else:
                                full_history[-1]['content'] = partial_words
                            token_counter += 1
                    except Exception as e:
                        finish_reason = '[EXCEPTION]'
                        print(e)
                        if len(partial_words) == 0:
                            pass
                        else:
                            full_history[-1]['exception'] = str(e)

                    answer = full_history[-1]['content']

                    yield answer, full_history

            full_history[-1]["estimate_tokens"] = estimate_used_tokens(partial_words, model_name=self.API_MODEL)
            chat = [(process_chat(full_history[i]), process_chat(full_history[i + 1])) for i in
                    range(1, len(full_history) - 1, 2) if full_history[i]['role'] != 'system']
            answer = full_history[-1]['content']
            yield answer, full_history
        except Exception as e:
            print(e)



    async def summarize_text(self, text_input,timeout=120):
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
        payload = self.parameters2payload(self.API_MODEL, conversation, paras,stream=False)

        response = await asyncio.to_thread(
            requests.post,
            self.BASE_URL, headers=self.API_HEADERS, json=payload, stream=False
        )


        # 解析返回的JSON結果
        this_choice = json.loads(response.content.decode())['choices'][0]
        summary = this_choice["message"]
        summary['total_tokens'] = response.json()["usage"]['completion_tokens']
        return summary


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
        response = openai.ChatCompletion.create(
            model=self.API_MODEL,
            messages=message_context,
            temperature=parameters.get('temperature')
        )
        try:
            # 解析返回的JSON結果
            answer = response.choices[0].message['content'].strip()
            prompt_tokens = response["usage"]['prompt_tokens']
            completion_tokens = response["usage"]['completion_tokens']
            estimate_tokens2 = estimate_used_tokens(answer) + 4
            return response.choices[0].message['content'].strip()
        except Exception as e:
            return response.choices[0].message['content'].strip() + "\n" + str(e)


    def generate_images(self, input_prompt, shorter_prompt=None, image_size=256):
        response = openai.Image.create(
            api_key=os.getenv("OPENAI_API_KEY"),
            prompt=input_prompt,
            response_format="b64_json",
            n=4,
            size="{0}x{1}".format(image_size, image_size)
        )
        images = []
        for index, image_dict in enumerate(response["data"]):
            decoded_data = b64decode(image_dict["b64_json"])
            # image_data=cv2.imdecode(np.fromstring(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            image_file = "generate_images/{0}-{1}.png".format(response['created'], index)
            if shorter_prompt is not None:
                image_file = "generate_images/{0}-{1}-{2}.png".format(response['created'],
                                                                      replace_special_chars(shorter_prompt), index)
            with open(image_file, 'wb') as f:
                f.write(decoded_data)
            images.append(pil_image.open(image_file))
            # pil_image.fromarray(image_data, 'RGB').save(image_file)
        return images


    def save_history(self, filename=None):
        history_json = json.dumps(self.FULL_HISTORY, ensure_ascii=False, indent=4)

    def load_history(self, filename=None):
        history_json = json.dumps(self.FULL_HISTORY, ensure_ascii=False, indent=4)
