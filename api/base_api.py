import json
import uuid
import os
import openai
import openai_async

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
import api.context_type as ContextType
from utils.regex_utils import *
from utils.chatgpt_utils import process_context,process_chat
from utils.tokens_utils import num_tokens_from_history,estimate_used_tokens

class GptBaseApi:
    def __init__(self, url="https://api.openai.com/v1/chat/completions",model="gpt-3.5-turbo",temperature=0.5,system_message='所有內容以繁體中文書寫'):
        self.BASE_URL = url
        self.BASE_IDENTITY = uuid.uuid4()
        self.API_KEY = os.getenv("OPENAI_API_KEY")
        self.API_MODEL=model
        self.API_HEADERS = {
            'Accept': 'text/event-stream',
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.API_KEY}"
        }
        self.SYSTEM_MESSAGE=system_message
        self.API_PARAMETERS={'top_p':1, 'temperature':temperature, 'top_k':1,'presence_penalty':0, 'frequency_penalty':0,'max_tokens':2500}
        self.FULL_HISTORY =[{"role": "system", "content": self.SYSTEM_MESSAGE,"estimate_tokens": estimate_used_tokens(self.SYSTEM_MESSAGE,model_name=self.API_MODEL)}]


    def build_message(self,role, content):
        """
        Build a chat message with the given role and content.

        Args:
        role (str): The role of the message sender, e.g., "system", "user", or "assistant".
        content (str): The content of the message.

        Returns:
        dict: A dictionary containing the role and content of the message.
        """
        return {"role": str(role), "content": str(content)}

    def process_context(self,prompt, context_type, full_history: list):
        if context_type==ContextType.skip:
            message_context = [self.build_message(message['role'], message['content']) for message in full_history if message['role']=='system']
        elif context_type==ContextType.sandbox:
            message_context=[]
        else:
            message_context = [self.build_message(message['role'], message['summary'] if message[
                                                                                        'role'] == 'assistant' and 'summary' in message else
            message['content']) for message in full_history]
        message_context.append({"role": "user", "content": prompt})
        return message_context


    def parameters2payload(self,model,message_with_context,parameters):
        payload = {
            "model": model,
            "messages": message_with_context,
            "temperature": parameters.get('temperature'),
            "top_p": parameters.get('top_p'),
            "n": parameters.get('top_k'),
            "max_tokens":parameters.get('max_tokens'),
            "stream": True,
            "presence_penalty":parameters.get('presence_penalty'),
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
            full_history[0]["estimate_tokens"]=estimate_used_tokens(full_history[0]["content"] + '/n' + input_prompt,model_name=self.API_MODEL)

        elif context_type == ContextType.override:
            full_history[0]["content"] = input_prompt
            full_history[0]["estimate_tokens"] = estimate_used_tokens( input_promp,model_name=self.API_MODEL)

        elif input_prompt:
            # 調用openai.ChatCompletion.create來生成機器人的回答
            estimate_tokens= estimate_used_tokens(input_prompt)
            message_context = self.process_context(input_prompt, context_type, full_history)
            partial_words = ''
            token_counter = 0
            payload = self.parameters2payload(self.API_MODEL,message_context,self.API_PARAMETERS)
            full_history.append({"role": "user", "content": input_prompt, "context_type": context_type,"estimate_tokens":estimate_tokens})
            request = requests.post(self.BASE_URL, headers=self.API_HEADERS, json=payload, stream=True)

            finish_reason = 'None'
            # client = sseclient.SSEClient(request)
            for chunk in request.iter_content(chunk_size=512, decode_unicode=False):

                try:
                    if chunk.decode('utf-8-sig').endswith('data: [DONE]\n\n'):
                        finish_reason = '[DONE]'
                        break

                    jstrs = chunk.decode('utf-8-sig').replace(':null', ':\"None\"')
                    this_choice = eval(choice_pattern.findall(jstrs)[-1])
                    finish_reason = this_choice['finish_reason']

                    if 'content' in this_choice['delta']:
                        # if partial_words == '' and this_choice['delta']['content'] == '\n\n':
                        #     pass
                        # elif this_choice['delta']['content'] == '\n\n':
                        #     partial_words += '\n  '
                        # else:
                        partial_words += this_choice['delta']['content']

                        if token_counter == 0:
                            full_history.append(
                                {"role": "assistant", "content": partial_words, "context_type": context_type})
                        else:
                            full_history[-1]['content'] = partial_words

                        token_counter += 1

                except Exception as e:
                    if len(partial_words) == 0:
                        pass
                    else:
                        full_history[-1]['exception'] = str(e)

                chat = [(process_chat(full_history[i]), process_chat(full_history[i + 1])) for i in
                        range(1, len(full_history) - 1, 2) if full_history[i]['role'] != 'system']
                answer = full_history[-1]['content']

                yield chat, answer, full_history


            # 檢查finish_reason是否為length
            while finish_reason != '[DONE]':
                # 自動以user角色發出「繼續寫下去」的PROMPT
                prompt = "繼續"
                # 調用openai.ChatCompletion.create來生成機器人的回答
                message_context = self.process_context(prompt, context_type, full_history)
                payload = self.parameters2payload(self.API_MODEL,message_context,self.API_PARAMETERS)
                request = requests.post(self.BASE_URL, headers=self.API_HEADERS, json=payload, stream=True)

                finish_reason = 'None'
                # client = sseclient.SSEClient(request)
                for chunk in request.iter_content(chunk_size=512):

                    try:

                        jstrs = chunk.decode('utf-8-sig').replace(':null', ':\"None\"')
                        this_choice = eval(choice_pattern.findall(jstrs)[-1])
                        finish_reason = this_choice['finish_reason']
                        if chunk.decode('utf-8').endswith('data: [DONE]\n\n'):
                            finish_reason = '[DONE]'
                            break
                        if 'content' in this_choice['delta']:
                            # if partial_words == '' and this_choice['delta']['content'] == '\n\n':
                            #     pass
                            # elif this_choice['delta']['content'] == '\n\n':
                            #     partial_words += '\n'
                            # else:
                            partial_words += this_choice['delta']['content']
                            if token_counter == 0:
                                full_history.append(
                                    {"role": "assistant", "content": partial_words, "context_type": context_type})
                            else:
                                full_history[-1]['content'] = partial_words
                            token_counter += 1
                    except Exception as e:
                        if len(partial_words) == 0:
                            pass
                        else:
                            full_history[-1]['exception'] = str(e)

                    chat = [(process_chat(full_history[i]), process_chat(full_history[i + 1])) for i in
                            range(1, len(full_history) - 1, 2) if full_history[i]['role'] != 'system']
                    answer=full_history[-1]['content']

                    yield chat, answer, full_history

            # 檢查接續後的完整回覆是否過長
            # print('bot_output: ',len(bot_output))

            if len(partial_words) > 200:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                tasks =[asyncio.ensure_future(self.summarize_text(text_input=partial_words,timeout=60))]
                loop.run_until_complete(asyncio.wait(tasks))
                response= tasks[0].result()
                summarization_text=response["content"].strip()
                full_history[-1]['summary'] = summarization_text
                full_history[-1]['summary_tokens'] =response["total_tokens"]


            full_history[-1]['tokens']=token_counter
            full_history[-1]["estimate_tokens"] = estimate_used_tokens(partial_words,model_name=self.API_MODEL)
            chat = [(process_chat(full_history[i]), process_chat(full_history[i + 1])) for i in
                    range(1, len(full_history) - 1, 2) if full_history[i]['role'] != 'system']
            answer = full_history[-1]['content']

            yield chat, answer, full_history

    def post_and_get_streaming_answer(self, message_context,  parameters, full_history):
        """post 串流形式的對話

        :param message_context:
        :param parameters:
        :param full_history:
        :return:
        """

        partial_words = ''
        token_counter = 0
        context_type=ContextType.prompt
        payload = self.parameters2payload(self.API_MODEL,message_context,self.API_PARAMETERS)
        full_history.append(message_context[-1])
        request = requests.post(self.BASE_URL, headers=self.API_HEADERS, json=payload, stream=True)

        finish_reason = 'None'
        # client = sseclient.SSEClient(request)
        for chunk in request.iter_content(chunk_size=512, decode_unicode=False):

            try:
                if chunk.decode('utf-8-sig').endswith('data: [DONE]\n\n'):
                    finish_reason = '[DONE]'
                    break

                jstrs = chunk.decode('utf-8-sig').replace(':null', ':\"None\"')
                this_choice = eval(choice_pattern.findall(jstrs)[-1])
                finish_reason = this_choice['finish_reason']

                if 'content' in this_choice['delta']:
                    # if partial_words == '' and this_choice['delta']['content'] == '\n\n':
                    #     pass
                    # elif this_choice['delta']['content'] == '\n\n':
                    #     partial_words += '\n  '
                    # else:
                    partial_words += this_choice['delta']['content']

                    if token_counter == 0:
                        full_history.append(
                            {"role": "assistant", "content": partial_words, "context_type": context_type})
                    else:
                        full_history[-1]['content'] = partial_words

                    token_counter += 1

            except Exception as e:
                if len(partial_words) == 0:
                    pass
                else:
                    full_history[-1]['exception'] = str(e)

            answer = full_history[-1]['content']

            yield answer, full_history


        # 檢查finish_reason是否為length
        while finish_reason != '[DONE]':
            # 自動以user角色發出「繼續寫下去」的PROMPT
            prompt = "繼續"
            # 調用openai.ChatCompletion.create來生成機器人的回答
            message_context = self.process_context(prompt, context_type, full_history)
            payload = self.parameters2payload(self.API_MODEL,message_context,self.API_PARAMETERS)
            request = requests.post(self.BASE_URL, headers=self.API_HEADERS, json=payload, stream=True)

            finish_reason = 'None'
            # client = sseclient.SSEClient(request)
            for chunk in request.iter_content(chunk_size=512):

                try:

                    jstrs = chunk.decode('utf-8-sig').replace(':null', ':\"None\"')
                    this_choice = eval(choice_pattern.findall(jstrs)[-1])
                    finish_reason = this_choice['finish_reason']
                    if chunk.decode('utf-8').endswith('data: [DONE]\n\n'):
                        finish_reason = '[DONE]'
                        break
                    if 'content' in this_choice['delta']:
                        # if partial_words == '' and this_choice['delta']['content'] == '\n\n':
                        #     pass
                        # elif this_choice['delta']['content'] == '\n\n':
                        #     partial_words += '\n'
                        # else:
                        partial_words += this_choice['delta']['content']
                        if token_counter == 0:
                            full_history.append(
                                {"role": "assistant", "content": partial_words, "context_type": context_type})
                        else:
                            full_history[-1]['content'] = partial_words
                        token_counter += 1
                except Exception as e:
                    if len(partial_words) == 0:
                        pass
                    else:
                        full_history[-1]['exception'] = str(e)


                answer=full_history[-1]['content']

                yield answer, full_history


        # full_history[-1]['tokens']=token_counter
        # answer = full_history[-1]['content']
        #
        # yield answer, full_history

    async def summarize_text(self, text_input,timeout=60):
        """post 串流形式的對話
        :param text_input:
        :return:
        """
        partial_words = ''
        token_counter = 0
        context_type=ContextType.explain
        conversation = [
            {
                "role": "system",
                "content": "你是萬能的文字助手，你擅長將任何文字在保持原意不變，但必須保留[人名、公司機構名稱、事物名稱、地點、時間、數值]、陳述事實與知識點前提下，作最精簡的摘要。當使用者輸入長文本，你將會回傳保持原意不變的精簡版本內容"
            },
            {
                "role": "user",
                "content": text_input
            }
        ]

        response=await openai_async.chat_complete(
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=timeout,
                payload={
                    "model": self.API_MODEL,
                    "messages": conversation,
                    "temperature" :0.01
                },
            )
        try:
            # 解析返回的JSON結果
            summary = response.json()["choices"][0]["message"]
            total_tokens=response.json()["usage"]['completion_tokens']
            summary['total_tokens']=total_tokens
            return summary
        except Exception as e:
            return await response.choices[0].message['content'].strip() + "\n" + str(e)



    def post_and_get_answer(self,message_context, parameters, full_history=None):
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
            full_history.append({"role": "user", "content": input_prompt, "context_type": ContextType.prompt})

        response = openai.ChatCompletion.create(
            model=self.API_MODEL,
            messages=message_context,
            temperature=parameters.get('temperature')
        )
        try:
            # 解析返回的JSON結果
            answer=response.choices[0].message['content'].strip()
            total_tokens=response["usage"]['completion_tokens']



            if full_history is not None:
                full_history.append({"role": "assistant", "content": answer, "context_type": ContextType.prompt,"total_tokens":total_tokens})
            return response.choices[0].message['content'].strip()
        except Exception as e:
            return response.choices[0].message['content'].strip() + "\n" + str(e)


    def generate_images(self,input_prompt, shorter_prompt=None,image_size=256):
        response =openai.Image.create(
            api_key=os.getenv("OPENAI_API_KEY"),
            prompt=input_prompt,
            response_format="b64_json",
            n=4,
            size="{0}x{1}".format(image_size,image_size)
        )
        images=[]
        for index, image_dict in enumerate(response["data"]):
            decoded_data=b64decode(image_dict["b64_json"])
            #image_data=cv2.imdecode(np.fromstring(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            image_file = "generate_images/{0}-{1}.png".format(response['created'],  index)
            if shorter_prompt is not None:
                image_file ="generate_images/{0}-{1}-{2}.png".format(response['created'],replace_special_chars(shorter_prompt),index)
            with open(image_file, 'wb') as f:
                f.write(decoded_data)
            images.append(pil_image.open(image_file))
            #pil_image.fromarray(image_data, 'RGB').save(image_file)
        return images

    # def summary_text(self,):

    def save_history(self, filename=None):
        history_json=json.dumps(self.FULL_HISTORY, ensure_ascii=False, indent=4)
    def load_history(self, filename=None):
        history_json=json.dumps(self.FULL_HISTORY, ensure_ascii=False, indent=4)








