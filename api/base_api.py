import json
import uuid
import os
import regex
import openai
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
        self.FULL_HISTORY =[{"role": "system", "content": self.SYSTEM_MESSAGE}]

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

        elif context_type == ContextType.override:
            full_history[0]["content"] = input_prompt
        elif input_prompt:
            # 調用openai.ChatCompletion.create來生成機器人的回答
            message_context = self.process_context(input_prompt, context_type, full_history)
            partial_words = ''
            token_counter = 0
            payload = self.parameters2payload(self.API_MODEL,message_context,self.API_PARAMETERS)
            full_history.append({"role": "user", "content": input_prompt, "context_type": context_type})
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

            if len(partial_words) > 500:
                # 自動以user角色要求「將此內容進行簡短摘要」的prompt需
                print(len(partial_words))

                prompt = "將以下文字進行摘要: \n" + partial_words
                # 調用openai.ChatCompletion.create來生成機器人的回答
                _parameters=copy.deepcopy(self.API_PARAMETERS)
                _parameters['temperature'] =0.1
                _parameters['max_tokens']=len(partial_words)//2
                message_context = self.process_context(prompt, ContextType.sandbox, full_history)
                print('\n[簡短摘要]\n',message_context)
                self.parameters2payload(self.API_MODEL, message_context, _parameters)
                request = requests.post(self.BASE_URL, headers=self.API_HEADERS, json=payload, stream=True)
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
                            full_history[-1]['summary'] = summary
                            token_counter += 1
                    except Exception as e:
                        if len(summary) == 0:
                            pass
                        else:
                            full_history[-1]['exception'] = str(e)
                # messages_history.append({"role": "assistant", "content": summary})
                print('summary: ', len(summary), summary)
            full_history[-1]['tokens']=token_counter
            chat = [(process_chat(full_history[i]), process_chat(full_history[i + 1])) for i in
                    range(1, len(full_history) - 1, 2) if full_history[i]['role'] != 'system']
            answer = full_history[-1]['content']

            yield chat, answer, full_history




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
            if full_history is not None:
                full_history.append({"role": "assistant", "content": answer, "context_type": ContextType.prompt})
            return response.choices[0].message['content'].strip()
        except Exception as e:
            return response.choices[0].message['content'].strip() + "\n" + str(e)


    def generate_images(self,input_prompt):
        response =openai.Image.create(
            api_key=os.getenv("OPENAI_API_KEY"),
            prompt=input_prompt,
            response_format="b64_json",
            n=4,
            size="512x512"
        )
        images=[]
        for index, image_dict in enumerate(response["data"]):
            image_data=b64decode(image_dict["b64_json"])
            image_data=cv2.imdecode(np.fromstring(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            image_file ="generate_images/{0}-{1}.png".format(response['created'],index)
            images.append(image_data)
            pil_image.fromarray(image_data, 'RGB').save(image_file)
        return images








