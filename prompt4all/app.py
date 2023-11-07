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
import copy
import requests
import asyncio
import time
import nest_asyncio
import openai_async
import whisper
import math
import numpy as np
from pydub import AudioSegment
from datetime import datetime, timedelta
from queue import Queue

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


def prompt_api(inputs, context_type, top_p, temperature, top_k, frequency_penalty, full_history=[]):
    _context_type = index2context(context_type)
    baseChatGpt.API_PARAMETERS['temperature'] = temperature
    baseChatGpt.API_PARAMETERS['top_p'] = top_p
    baseChatGpt.API_PARAMETERS['top_k'] = top_k
    baseChatGpt.API_PARAMETERS['frequency_penalty'] = frequency_penalty
    if isinstance(full_history, gr.State):
        full_history=full_history.value
    streaming_chat = baseChatGpt.post_a_streaming_chat(inputs, _context_type, baseChatGpt.API_PARAMETERS, full_history)

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
            _current_item=cxt.conversation_history.selected_item.current_item
            mid = str(uuid.uuid4())
            if _current_item.children is None:
                _current_item.children=[]
            _current_item.children.append(mid)
            cxt.conversation_history.selected_item.mapping[mid]=Mapping(mapping_id=mid,parent=_current_item.id).new_user_message_mapping(inputs)
            cxt.conversation_history.selected_item.current_node=mid

            pass
        try:
            chat, answer, full_history = next(streaming_chat)
            yield chat, full_history, full_history
        except StopIteration:
            break
        except Exception as e:
            raise gr.Error(str(e))


def nlu_api(text_input):
    # 創建與API的對話

    text_inputs = text_input.split('\n')
    _parameters = copy.deepcopy(baseChatGpt.API_PARAMETERS)
    _parameters['temperature'] = 0.1
    results = []
    for txt in text_inputs:
        conversation = [
            {
                "role": "system",
                "content": "請逐一讀取下列句子，每個句子先理解語意後再進行分詞、情感偵測、命名實體偵測以及意圖偵測。分詞結果是指將輸入的文字，先進行語意理解，然後基於語意理解合理性的前提下，將輸入文字進行分詞(tokenize)，若非必要，盡量不要出現超過3個字的詞，然後使用「|」插入至分詞詞彙即構成分詞結果。\n需要偵測的情感類型\n正面情緒(positive_emotions):[自信,快樂,體貼,幸福,信任,喜愛,尊榮,期待,感動,感謝,熱門,獨特,稱讚]\n負面情緒(negative_emotions):[失望,危險,後悔,冷漠,懷疑,恐懼,悲傷,憤怒,擔心,無奈,煩悶,虛假,討厭,貶責,輕視]\n當句子中有符合以上任何情感種類時，請盡可能的將符合的「情感種類」及句子中的那些「觸及到情感種類的內容」成對的列舉出來，一個句子可以觸及不只一種情感。\n需要偵測的實體類型(entities)應該包括但不僅限於[中文人名,中文翻譯人名,外語人名,歌手/樂團/團體名稱,地名/地點,時間,公司機構名/品牌名,商品名,商品規格,化合物名/成分名,歌曲/書籍/作品名稱,其他專有名詞,金額,其他數值]，你可以視情況擴充，\n此外，若是句子中有偵測到符合上述實體類型時，也請盡可能的將符合的「實體類型」及句子中的那些「觸及到實體類型內容｣成對的列舉出來，一個句子可以觸及不只一種實體類型。當你偵測到句子中有要求你代為執行某個任務、或是表達自己想要的事物或是行動、或是想查詢某資訊的意圖(intents)時，根據以意圖最普遍的英文講法之「名詞+動詞-ing」的駝峰式命名形式來組成意圖類別(例如使用者說「請幫我訂今天下午5點去高雄的火車票」其意圖類別為TicketOrdering)，及句子中的那些「觸及到意圖類別的內容」成對的列舉出來，一個句子可以觸及不只一種意圖。以下為「張大帥的人生是一張茶几，上面放滿了杯具。而本身就是杯具」的範例解析結果\n"
                           "{\nsentence:  \"張大帥的人生是一張茶几，上面放滿了杯具。而本身就是杯具\",\nsegmented_sentence:  \"張大帥|的|人生|是|一|張|茶几|，|上面|放滿了|杯具|。|而|本身|就是|杯具\",\npositive_emotions:  [\n0:  {\ntype:  \"煩悶\",\ncontent:  \"放滿了杯具\"\n} ,\n1:  {\ntype:  \"無奈\",\ncontent:  \"本身就是杯具\"\n}\n],\nnegative_emotions:  [\n0:  {\ntype:  \"失望\",\ncontent:  \"上面放滿了杯具\"\n} \n],\nentities:  [\n0:  {\ntype:  \"中文人名\",\ncontent:\"張大帥\"\n}\n]\n}\n\r最後將每個句子的解析結果整合成單一json格式，縮進量為1。"
            },
            {
                "role": "user",
                "content": txt
            }
        ]
        jstrs = json_pattern.findall(baseChatGpt.post_and_get_answer(conversation, _parameters))
        jstrs = jstrs[0] if len(jstrs) == 1 else '[' + ', '.join(jstrs) + ']'
        output_json = json.loads(jstrs)
        results.append(json.dumps(output_json, ensure_ascii=False, indent=3))

        yield '[' + ', '.join(results) + ']'


def image_api(text_input, image_size, temperature=1.2):
    # 創建與API的對話
    _system_prompt = open("prompts/dalle2.md", encoding="utf-8").read()
    _parameters = copy.deepcopy(baseChatGpt.API_PARAMETERS)
    _parameters['temperature'] = temperature
    _parameters['max_tokens'] = 100
    results = []
    conversation = [
        {
            "role": "system",
            "content": _system_prompt
        },
        {
            "role": "user",
            "content": text_input
        }
    ]
    image_prompt = imageChatGpt.post_and_get_answer(conversation, _parameters)
    if ':' in image_prompt:
        image_prompt = ' '.join(image_prompt.split(':')[1:])
    images_urls = imageChatGpt.generate_images(image_prompt, text_input, image_size)
    return image_prompt, images_urls


def rewrite_api(text_input, style_name):
    # 創建與API的對話

    style_name = style_name.split('(')[0].strip()
    _parameters = copy.deepcopy(baseChatGpt.API_PARAMETERS)
    _parameters['temperature'] = 1.2
    _parameters['frequency_penalty'] = 0
    _parameters['presence_penalty'] = 0
    results = []
    conversation = [
        {
            "role": "system",
            "content": "#zh-TW 你是一個寫作高手，你擅長使用{0}的語氣來改寫輸入之文字，並依照語氣風格特性適時加入表情符號、emoji與調整文字排版，無須解釋，直接改寫".format(
                style_name)
        },
        {
            "role": "user",
            "content": text_input
        }
    ]
    streaming_answer = baseChatGpt.post_and_get_streaming_answer(conversation, _parameters, conversation)
    while True:
        try:
            answer, full_history = next(streaming_answer)
            yield answer
        except StopIteration:
            break


async def summarize_text(text_input, system_prompt):
    """post 串流形式的對話
    :param system_prompt:
    :param text_input:
    :return:
    """
    partial_words = ''
    token_counter = 0
    context_type = ContextType.skip
    passage = "輸入文字內容:\"\"\"\n{0}\n\"\"\"\n".format(text_input)
    conversation = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": passage
        }
    ]
    _parameters = copy.deepcopy(summaryChatGpt.API_PARAMETERS)
    _parameters['temperature'] = 0.001
    _parameters['presence_penalty'] = 1.2
    payload = summaryChatGpt.parameters2payload(summaryChatGpt.API_MODEL, conversation, _parameters, stream=False)

    response = await asyncio.to_thread(
        requests.post,
        summaryChatGpt.BASE_URL, headers=summaryChatGpt.API_HEADERS, json=payload, stream=False
    )

    try:
        # 解析返回的JSON結果
        this_choice = json.loads(response.content.decode())['choices'][0]
        print(this_choice)
        summary = this_choice["message"]
        total_tokens = response.json()["usage"]['completion_tokens']
        summary['total_tokens'] = total_tokens
        return summary
    except Exception as e:
        raise gr.Error(str(response.json()) + "\n" + str(e))


async def rolling_summary(large_inputs, full_history, summary_method, summary_options):
    _parameters = copy.deepcopy(summaryChatGpt.API_PARAMETERS)
    _parameters['temperature'] = 0
    _parameters['presence_penalty'] = 1.2
    large_inputs = large_inputs.split('\n') if isinstance(large_inputs, str) else large_inputs
    large_inputs_tokens = builtins.sum([estimate_used_tokens(w) + 1 for w in large_inputs])
    large_inputs_bk = copy.deepcopy(large_inputs)
    st = datetime.now()

    is_final_stage = False
    keep_summary = True
    summary_repository = OrderedDict()
    cleansed_summary = []
    mindmap_history = ""
    mindmap_head = '# 摘要心智圖'
    meeting_minutes = ''
    meeting_head = '# 會議記錄'
    topic_shortcuts = ''
    topic_head = '# 主題重點'

    if summary_method == 0:
        _system_prompt = open("prompts/rolling_summary.md", encoding="utf-8").read()
        _final_prompt = open("prompts/summary_final_cleansing.md", encoding="utf-8").read()

        summary_history = '空的清單'
        this_system_tokens = estimate_used_tokens(_system_prompt) + estimate_used_tokens('system',
                                                                                         model_name=summaryChatGpt.API_MODEL) + 4
        this_final_tokens = estimate_used_tokens(_final_prompt) + estimate_used_tokens('system',
                                                                                       model_name=summaryChatGpt.API_MODEL) + 4
        available_tokens = summaryChatGpt.MAX_TOKENS - this_system_tokens - 4 - 2
        this_summary_tokens = estimate_used_tokens(summary_history)
        this_available_tokens = (available_tokens - 2 * this_summary_tokens) * 0.667 - 100

        partial_words = ''

        cnt = 0
        unchanged_summary = []
        while keep_summary:
            summary_size_ratio = 2 * this_summary_tokens / available_tokens
            print('summary_size_ratio:{0:.2%}'.format(summary_size_ratio))
            # 摘要量過大需要縮減
            if summary_size_ratio > 0.4 and not is_final_stage:
                content = summary_repository[cnt]
                this_tokens = builtins.sum([estimate_used_tokens(c) + 1 for c in content])
                part1, part2 = split_summary(content, int(this_tokens * 0.667))
                summary_history = '\n'.join(part2)
                unchanged_summary.extend(part1)
                this_summary_tokens = estimate_used_tokens(summary_history)
                this_available_tokens = (available_tokens - 2 * this_summary_tokens) * 0.667 - 100
                new_summary_size_ratio = 2 * this_summary_tokens / available_tokens
                print('summary_size_ratio  {0:.2%}=>{1:.2%}'.format(summary_size_ratio, new_summary_size_ratio))

            try:
                this_available_tokens = (
                                                available_tokens - 2 * this_summary_tokens) * 0.667 - 100 if not is_final_stage else (
                                                                                                                                         available_tokens) // 2 - 100
                # get tokens
                if len(large_inputs) == 0:
                    if is_final_stage:
                        break
                    else:
                        is_final_stage = True
                        keep_summary = False
                        available_tokens = summaryChatGpt.MAX_TOKENS - this_final_tokens - 4 - 2
                        this_summary_tokens = 0
                        this_available_tokens = (available_tokens) // 2 - 100
                        large_inputs = copy.deepcopy(unchanged_summary)
                        large_inputs.extend(summary_history.split('\n'))
                        keep_summary = True

                if not is_final_stage:
                    partial_words, large_inputs = get_next_paragraph(large_inputs, this_available_tokens)
                    remain_tokens = builtins.sum([estimate_used_tokens(w) + 1 for w in large_inputs])
                    print('partial_words:{0} large_inputs:{1}'.format(
                        builtins.sum([estimate_used_tokens(w) + 1 for w in partial_words]), remain_tokens))

                else:

                    partial_words, large_inputs = get_next_paragraph(large_inputs, this_available_tokens)
                    remain_tokens = builtins.sum([estimate_used_tokens(w) + 1 for w in large_inputs])
                    if remain_tokens < 50:
                        partial_words.extend(large_inputs)
                        remain_tokens = 0
                        large_inputs = []
                    print('partial_words:{0} large_inputs:{1}'.format(
                        builtins.sum([estimate_used_tokens(w) + 1 for w in partial_words]), remain_tokens))
                    if len(large_inputs) == 0:
                        keep_summary = False

                passage = "累積摘要清單:\n\n\"\"\"\n\n{0}\n\n\"\"\"\n\n輸入文字內容:\n\n\"\"\"\n\n{1}\n\n\"\"\"\n\n".format(
                    summary_history, '\n'.join(partial_words))
                passage_final = "摘要清單:\n\n\"\"\"\n\n{0}\n\n\"\"\"\n\n標號起始數字:{1}\n".format(
                    '\n'.join(partial_words), get_last_ordered_index(cleansed_summary))

                conversation = [
                    {
                        "role": "system",
                        "content": _system_prompt if not is_final_stage else _final_prompt
                    },
                    {
                        "role": "user",
                        "content": passage if not is_final_stage else passage_final
                    }
                ]
                print(conversation)

                _max_tokens = builtins.min(summaryChatGpt.MAX_TOKENS,
                                           estimate_used_tokens(str(conversation)) + estimate_used_tokens(
                                               '\n'.join(partial_words)) * (0.3 if not is_final_stage else 1))
                _parameters['max_tokens'] = _max_tokens

                streaming_answer = summaryChatGpt.post_and_get_streaming_answer(conversation, _parameters, full_history)
                answer = ''
                answer_head = """  \n## 第{0}部分摘要 {1:.2%}  \n\n\n""".format(cnt + 1, float(
                    large_inputs_tokens - remain_tokens) / large_inputs_tokens).replace('\n\n\n',
                                                                                        '\n{0} \n') if not is_final_stage else """  \n## 最終版摘要  \n{0} \n"""

                while True:
                    try:
                        answer, full_history = next(streaming_answer)
                        if not is_final_stage:
                            yield answer_head.format(text2markdown(('\n'.join(unchanged_summary) if len(
                                unchanged_summary) > 0 else '') + '  \n' + '  \n'.join(
                                get_rolling_summary_results(answer)))), full_history
                        else:
                            yield answer_head.format(text2markdown(('\n'.join(cleansed_summary) if len(
                                cleansed_summary) > 0 else '') + '  \n' + '  \n'.join(
                                get_rolling_summary_results(answer)))), full_history
                    except StopIteration:
                        break
                print(answer_head.format(answer))
                print('\n\n')
                if not is_final_stage:
                    summary_repository[cnt + 1] = get_rolling_summary_results(answer)
                    summary_history = '\n'.join(summary_repository[cnt + 1])
                    this_summary_tokens = estimate_used_tokens(summary_history)
                    this_available_tokens = (available_tokens - 2 * this_summary_tokens) * 0.667 - 100

                    cnt += 1
                else:
                    cleansed_summary.extend(get_rolling_summary_results(answer))
                    this_available_tokens = (available_tokens) // 2 - 100

                yield answer_head.format(text2markdown('\n'.join(
                    unchanged_summary) + '\n' + summary_history)) if not is_final_stage else answer_head.format(
                    text2markdown('\n'.join(cleansed_summary))), full_history

            except Exception as e:
                PrintException()
                raise gr.Error(str(e))

    elif summary_method == 1:
        _system_prompt = open("prompts/incremental_rolling_summary.md", encoding="utf-8").read()
        _final_prompt = open("prompts/summary_final_cleansing.md", encoding="utf-8").read()

        summary_history = '空的清單'
        this_system_tokens = estimate_used_tokens(_system_prompt) + estimate_used_tokens('system',
                                                                                         model_name=summaryChatGpt.API_MODEL) + 4
        this_final_tokens = estimate_used_tokens(_final_prompt) + estimate_used_tokens('system',
                                                                                       model_name=summaryChatGpt.API_MODEL) + 4
        available_tokens = summaryChatGpt.MAX_TOKENS - this_system_tokens - 4 - 2
        this_summary_tokens = estimate_used_tokens(summary_history)
        this_available_tokens = (available_tokens - 2 * this_summary_tokens) * 0.667 - 100

        partial_words = ''

        cnt = 0
        unchanged_summary = []
        while keep_summary:
            summary_size_ratio = 2 * this_summary_tokens / available_tokens
            print('summary_size_ratio:{0:.2%}'.format(summary_size_ratio))
            # 摘要量過大需要縮減
            if summary_size_ratio > 0.4 and not is_final_stage:
                content = summary_repository[cnt]
                this_tokens = builtins.sum([estimate_used_tokens(c) + 1 for c in content])
                part1, part2 = split_summary(content, int(this_tokens * 0.667))
                summary_history = '\n'.join(part2)
                unchanged_summary.extend(part1)
                this_summary_tokens = estimate_used_tokens(summary_history)
                this_available_tokens = (available_tokens - 2 * this_summary_tokens) * 0.667 - 100
                new_summary_size_ratio = 2 * this_summary_tokens / available_tokens
                print('summary_size_ratio  {0:.2%}=>{1:.2%}'.format(summary_size_ratio, new_summary_size_ratio))

            try:
                this_available_tokens = (
                                                available_tokens - 2 * this_summary_tokens) * 0.667 - 100 if not is_final_stage else (
                                                                                                                                         available_tokens) // 2 - 100
                # get tokens
                if len(large_inputs) == 0:
                    if is_final_stage:
                        break
                    else:
                        is_final_stage = True
                        keep_summary = False
                        available_tokens = summaryChatGpt.MAX_TOKENS - this_final_tokens - 4 - 2
                        this_summary_tokens = 0
                        this_available_tokens = (available_tokens) // 2 - 100
                        large_inputs = copy.deepcopy(unchanged_summary)
                        large_inputs.extend(summary_history.split('\n'))
                        keep_summary = True

                if not is_final_stage:
                    partial_words, large_inputs = get_next_paragraph(large_inputs, this_available_tokens)
                    remain_tokens = builtins.sum([estimate_used_tokens(w) + 1 for w in large_inputs])
                    print('partial_words:{0} large_inputs:{1}'.format(
                        builtins.sum([estimate_used_tokens(w) + 1 for w in partial_words]), remain_tokens))

                else:

                    partial_words, large_inputs = get_next_paragraph(large_inputs, this_available_tokens)
                    remain_tokens = builtins.sum([estimate_used_tokens(w) + 1 for w in large_inputs])
                    if remain_tokens < 50:
                        partial_words.extend(large_inputs)
                        remain_tokens = 0
                        large_inputs = []
                    print('partial_words:{0} large_inputs:{1}'.format(
                        builtins.sum([estimate_used_tokens(w) + 1 for w in partial_words]), remain_tokens))
                    if len(large_inputs) == 0:
                        keep_summary = False

                passage = "累積摘要清單:\n\n\"\"\"\n\n{0}\n\n\"\"\"\n\n輸入文字內容:\n\n\"\"\"\n\n{1}\n\n\"\"\"\n\n".format(
                    summary_history, '\n'.join(partial_words))
                passage_final = "摘要清單:\n\n\"\"\"\n\n{0}\n\n\"\"\"\n\n標號起始數字:{1}\n".format(
                    '\n'.join(partial_words), get_last_ordered_index(cleansed_summary))

                conversation = [
                    {
                        "role": "system",
                        "content": _system_prompt if not is_final_stage else _final_prompt
                    },
                    {
                        "role": "user",
                        "content": passage if not is_final_stage else passage_final
                    }
                ]
                print(conversation)

                _max_tokens = builtins.min(summaryChatGpt.MAX_TOKENS,
                                           estimate_used_tokens(str(conversation)) + estimate_used_tokens(
                                               '\n'.join(partial_words)) * (0.3 if not is_final_stage else 1))
                _parameters['max_tokens'] = _max_tokens

                streaming_answer = summaryChatGpt.post_and_get_streaming_answer(conversation, _parameters, full_history)
                answer = ''
                answer_head = """  \n## 第{0}部分摘要 {1:.2%}  \n\n\n""".format(cnt + 1, float(
                    large_inputs_tokens - remain_tokens) / large_inputs_tokens).replace('\n\n\n',
                                                                                        '\n{0} \n') if not is_final_stage else """  \n## 最終版摘要  \n{0} \n"""

                while True:
                    try:
                        answer, full_history = next(streaming_answer)
                        if not is_final_stage:
                            yield answer_head.format(text2markdown(('\n'.join(unchanged_summary) if len(
                                unchanged_summary) > 0 else '') + '  \n' + '  \n'.join(
                                get_rolling_summary_results(answer)))), full_history
                        else:
                            yield answer_head.format(text2markdown(('\n'.join(cleansed_summary) if len(
                                cleansed_summary) > 0 else '') + '  \n' + '  \n'.join(
                                get_rolling_summary_results(answer)))), full_history
                    except StopIteration:
                        break
                    except Exception as e:
                        gr.Error(str(e))
                print(answer_head.format(answer))
                print('\n\n')
                if not is_final_stage:
                    merged_summary_history = summary_history.split('\n') if summary_history != '空的清單' else []
                    number_list = [extract_numbered_list_member(txt) for txt in merged_summary_history]
                    max_number = 0
                    if len(merged_summary_history) > 0:
                        max_number = int(extract_numbered_list_member(merged_summary_history[-1]).split('.')[0])

                    new_summary = get_rolling_summary_results(answer)
                    for i in range(len(new_summary)):
                        this_summary = new_summary[i]
                        this_number = extract_numbered_list_member(this_summary)
                        if this_number in number_list:
                            merged_summary_history[number_list.index(this_number)] = this_summary
                        else:
                            merged_summary_history.append(this_summary)

                    summary_repository[cnt + 1] = merged_summary_history
                    summary_history = '\n'.join(summary_repository[cnt + 1])
                    this_summary_tokens = estimate_used_tokens(summary_history)
                    this_available_tokens = (available_tokens - 2 * this_summary_tokens) * 0.667 - 100

                    cnt += 1
                else:
                    cleansed_summary.extend(get_rolling_summary_results(answer))
                    this_available_tokens = (available_tokens) // 2 - 100

                yield answer_head.format(text2markdown('\n'.join(unchanged_summary) + '\n' + '  \n'.join(
                    get_rolling_summary_results(answer)))) if not is_final_stage else answer_head.format(text2markdown(
                    '\n'.join(cleansed_summary) + '\n' + '  \n'.join(
                        get_rolling_summary_results(answer)))), full_history

            except Exception as e:
                PrintException()
                raise gr.Error(str(e))
    elif summary_method == 2:
        _system_prompt = open("prompts/parallel_chunks_summary.md", encoding="utf-8").read()
        _final_prompt = open("prompts/summary_final_cleansing.md", encoding="utf-8").read()

        summary_history = '空的清單'
        this_system_tokens = estimate_used_tokens(_system_prompt) + estimate_used_tokens('system',
                                                                                         model_name=summaryChatGpt.API_MODEL) + 4
        this_final_tokens = estimate_used_tokens(_final_prompt) + estimate_used_tokens('system',
                                                                                       model_name=summaryChatGpt.API_MODEL) + 4
        available_tokens = summaryChatGpt.MAX_TOKENS - this_system_tokens - 4 - 2
        this_summary_tokens = estimate_used_tokens(summary_history)
        this_available_tokens = (available_tokens - 2 * this_summary_tokens) * 0.667 - 100

        text_dict = OrderedDict()
        tasks = []
        cnt = 0
        while keep_summary:
            partial_words, large_inputs = get_next_paragraph(large_inputs, this_available_tokens)
            remain_tokens = builtins.sum([estimate_used_tokens(w) + 1 for w in large_inputs])
            print('partial_words:{0} large_inputs:{1}'.format(
                builtins.sum([estimate_used_tokens(w) + 1 for w in partial_words]), remain_tokens))

            summary_repository[cnt] = OrderedDict()
            summary_repository[cnt]['text'] = '\n'.join(partial_words)
            tasks.append(summarize_text('\n'.join(partial_words), _system_prompt))
            time.sleep(2)
            if len(large_inputs) == 0:
                keep_summary = False
        print('預計切成{0}塊'.format(len(tasks)))
        return_values = await asyncio.gather(*tasks)
        print(datetime.now() - st)
        print(return_values)
        for k in range(len(return_values)):
            # handle process fail
            if isinstance(return_values[k], str) and 'Error' in return_values[k]:
                _parameters = copy.deepcopy(summaryChatGpt.API_PARAMETERS)
                _parameters['temperature'] = 0.001
                _parameters['presence_penalty'] = 1.2
                passage = "輸入文字內容:\n\n\"\"\"\n\n{0}\n\n\"\"\"\n\n".format(summary_repository[k]['text'])
                conversation = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": passage
                    }
                ]

                summaryChatGpt.make_response()
                payload = summaryChatGpt.parameters2payload(summaryChatGpt.API_MODEL, conversation, _parameters,
                                                            stream=False)
                response = requests.post(summaryChatGpt.BASE_URL, headers=summaryChatGpt.API_HEADERS, json=payload,
                                         stream=False)
                return_values[k] = json.loads(response.content.decode())['choices'][0]["message"]

        all_summary = aggregate_summary(return_values)
        is_final_stage = True
        keep_summary = False
        available_tokens = summaryChatGpt.MAX_TOKENS - this_final_tokens - 4 - 2
        this_summary_tokens = 0
        this_available_tokens = (available_tokens) // 2 - 100
        large_inputs = copy.deepcopy(all_summary)
        keep_summary = True
        while keep_summary:
            partial_words, large_inputs = split_summary(large_inputs, this_available_tokens)
            remain_tokens = builtins.sum([estimate_used_tokens(w) + 1 for w in large_inputs])
            print('partial_words:{0} large_inputs:{1}'.format(
                builtins.sum([estimate_used_tokens(w) + 1 for w in partial_words]), remain_tokens))
            if len(large_inputs) == 0:
                keep_summary = False
            passage_final = "摘要清單:\n\n\"\"\"\n\n{0}\n\n\"\"\"\n\n標號起始數字:{1}\n".format(
                '\n'.join(partial_words), get_last_ordered_index(cleansed_summary))

            conversation = [
                {
                    "role": "system",
                    "content": _final_prompt
                },
                {
                    "role": "user",
                    "content": passage_final
                }
            ]
            print(conversation)
            streaming_answer = summaryChatGpt.post_and_get_streaming_answer(conversation, _parameters, full_history)
            answer = ''
            answer_head = """  \n## 最終版摘要  \n{0} \n"""
            while True:
                try:
                    answer, full_history = next(streaming_answer)
                    yield answer_head.format(text2markdown(
                        '\n'.join(cleansed_summary) if len(cleansed_summary) > 0 else '' + '  \n' + '  \n'.join(
                            get_rolling_summary_results(answer)))), full_history
                except StopIteration:
                    break
            print(answer_head.format(answer))
            print('\n\n')

            cleansed_summary.extend(get_rolling_summary_results(answer))
            this_available_tokens = (available_tokens) // 2 - 100
            yield answer_head.format(text2markdown('\n'.join(cleansed_summary))), full_history

    if '心智圖' in summary_options:
        _system_prompt = open("prompts/mindmap_summary.md", encoding="utf-8").read()

        base_summary = copy.deepcopy(cleansed_summary)
        keep_summary = True
        this_system_tokens = estimate_used_tokens(_system_prompt) + estimate_used_tokens('system',
                                                                                         model_name=summaryChatGpt.API_MODEL) + 4
        available_tokens = summaryChatGpt.MAX_TOKENS - this_system_tokens - 4 - 2
        this_summary_tokens = estimate_used_tokens(mindmap_history)
        this_available_tokens = (available_tokens - 2 * this_summary_tokens) * 0.667 - 100
        large_inputs = base_summary
        partial_words = ''

        cnt = 0
        try:
            while keep_summary:
                this_system_tokens = estimate_used_tokens(str(mindmap_history))
                this_available_tokens = (available_tokens - this_system_tokens) - 100
                # get tokens

                partial_words, large_inputs = get_next_paragraph(large_inputs, this_available_tokens)
                print('partial_words:{0} large_inputs:{1}'.format(len(''.join(partial_words)),
                                                                  len(''.join(large_inputs))))

                passage = "摘要心智圖:\n\n{0}\n\n摘要清單:\n\n{1}\n\n".format(mindmap_history, '\n'.join(partial_words))

                conversation = [
                    {
                        "role": "system",
                        "content": _system_prompt
                    },
                    {
                        "role": "user",
                        "content": passage
                    }
                ]
                print(conversation)
                streaming_answer = summaryChatGpt.post_and_get_streaming_answer(conversation, _parameters, full_history)
                answer = ''

                while True:
                    try:
                        answer, full_history = next(streaming_answer)
                    except StopIteration:
                        break
                print(mindmap_head)
                print(answer)
                print('\n\n')
                if len(large_inputs) == 0:
                    keep_summary = False

                mindmap_history = answer
                available_tokens = int((summaryChatGpt.MAX_TOKENS - 200 - estimate_used_tokens(
                    answer) - this_system_tokens - 4 - 2) * 0.667)
                cnt += 1

        except Exception as e:
            raise gr.Error(str(e))
        yield answer_head.format(
            text2markdown('\n'.join(cleansed_summary))) + '\n\n\n' + mindmap_head + '\n' + mindmap_history, full_history

    if '會議記錄' in summary_options:
        _system_prompt = open("prompts/meeting_minutes_summary.md", encoding="utf-8").read()
        base_summary = copy.deepcopy(cleansed_summary)
        keep_summary = True
        this_system_tokens = estimate_used_tokens(_system_prompt) + estimate_used_tokens('system',
                                                                                         model_name=summaryChatGpt.API_MODEL) + 4
        this_final_tokens = estimate_used_tokens(_final_prompt) + estimate_used_tokens('system',
                                                                                       model_name=summaryChatGpt.API_MODEL) + 4
        available_tokens = summaryChatGpt.MAX_TOKENS - this_system_tokens - 4 - 2
        this_summary_tokens = estimate_used_tokens(meeting_minutes)
        this_available_tokens = (available_tokens - 2 * this_summary_tokens) * 0.667 - 100
        large_inputs = base_summary
        partial_words = ''
        unchanged_summary = []
        cnt = 0
        try:
            while keep_summary:
                this_system_tokens = estimate_used_tokens(str(meeting_minutes))
                this_available_tokens = (available_tokens - this_system_tokens) - 100
                # get tokens

                partial_words, large_inputs = get_next_paragraph(large_inputs, this_available_tokens)
                print('partial_words:{0} large_inputs:{1}'.format(len(''.join(partial_words)),
                                                                  len(''.join(large_inputs))))

                passage = "會議記錄重點:\n\n{0}\n\n摘要清單:\n\n{1}\n\n".format(meeting_minutes,
                                                                                '\n'.join(partial_words))

                conversation = [
                    {
                        "role": "system",
                        "content": _system_prompt
                    },
                    {
                        "role": "user",
                        "content": passage
                    }
                ]
                print(conversation)
                streaming_answer = summaryChatGpt.post_and_get_streaming_answer(conversation, _parameters, full_history)
                answer = ''
                meeting_head = '# 會議記錄'
                while True:
                    try:
                        answer, full_history = next(streaming_answer)
                    except StopIteration:
                        break
                print(meeting_head)
                print(answer)
                print('\n\n')
                if len(large_inputs) == 0:
                    keep_summary = False

                meeting_minutes = answer
                available_tokens = int((summaryChatGpt.MAX_TOKENS - 200 - estimate_used_tokens(
                    answer) - this_system_tokens - 4 - 2) * 0.667)
                cnt += 1

        except Exception as e:
            raise gr.Error(str(e))
        yield answer_head.format(text2markdown('\n'.join(
            cleansed_summary))) + '\n\n\n' + mindmap_head + '\n' + mindmap_history + '\n\n\n' + meeting_head + '\n' + meeting_minutes, full_history

    if '重點主題' in summary_options:
        _system_prompt = open("prompts/topic_driven_summary.md", encoding="utf-8").read()
        base_summary = copy.deepcopy(cleansed_summary)
        keep_summary = True
        this_system_tokens = estimate_used_tokens(_system_prompt) + estimate_used_tokens('system',
                                                                                         model_name=summaryChatGpt.API_MODEL) + 4
        this_final_tokens = estimate_used_tokens(_final_prompt) + estimate_used_tokens('system',
                                                                                       model_name=summaryChatGpt.API_MODEL) + 4
        available_tokens = summaryChatGpt.MAX_TOKENS - this_system_tokens - 4 - 2
        this_summary_tokens = estimate_used_tokens(topic_shortcuts)
        this_available_tokens = (available_tokens - 2 * this_summary_tokens) * 0.667 - 100
        large_inputs = base_summary
        partial_words = ''

        cnt = 0
        try:
            while keep_summary:
                this_system_tokens = estimate_used_tokens(str(topic_shortcuts))
                this_available_tokens = (available_tokens - this_system_tokens) - 100
                # get tokens

                partial_words, large_inputs = get_next_paragraph(large_inputs, this_available_tokens)
                print('partial_words:{0} large_inputs:{1}'.format(len(''.join(partial_words)),
                                                                  len(''.join(large_inputs))))

                passage = "重點主題:\n\n{0}\n\n摘要清單:\n\n{1}\n\n".format(topic_shortcuts, '\n'.join(partial_words))

                conversation = [
                    {
                        "role": "system",
                        "content": _system_prompt
                    },
                    {
                        "role": "user",
                        "content": passage
                    }
                ]
                print(conversation)
                streaming_answer = summaryChatGpt.post_and_get_streaming_answer(conversation, _parameters, full_history)
                answer = ''

                while True:
                    try:
                        answer, full_history = next(streaming_answer)
                    except StopIteration:
                        break
                print(topic_head)
                print(answer)
                print('\n\n')
                if len(large_inputs) == 0:
                    keep_summary = False
                topic_shortcuts = answer
                available_tokens = int((summaryChatGpt.MAX_TOKENS - 200 - estimate_used_tokens(
                    topic_shortcuts) - this_system_tokens - 4 - 2) * 0.667)
                cnt += 1

        except Exception as e:
            raise gr.Error(str(e))
        yield answer_head.format(text2markdown('\n'.join(cleansed_summary))) + '\n\n\n' + mindmap_history, full_history


def estimate_tokens(text, text2, state):
    text = '' if text is None else text
    text2 = '' if text2 is None else text2
    t1 = '輸入文本長度為{0},預計耗用tokens數為:{1}'.format(len(text),
                                                           estimate_used_tokens(text, summaryChatGpt.API_MODEL) + 4)
    if len(text2) == 0:
        return t1, state
    else:
        t2 = '輸出文本長度為{0},預計耗用tokens數為:{1}'.format(len(text2), estimate_used_tokens(text2,
                                                                                                summaryChatGpt.API_MODEL) + 4)
        return t1 + '\t\t' + t2, state


def reformat_freq(sr, y):
    if sr not in (
            48000,
            16000,
    ):  # Deepspeech only supports 16k, (we convert 48k -> 16k)
        raise ValueError("Unsupported rate", sr)
    if sr == 48000:
        y = (
            ((y / max(np.max(y), 1)) * 32767)
            .reshape((-1, 3))
            .mean(axis=1)
            .astype("int16")
        )
        sr = 16000
    return sr, y


def transcribe(audio, need_timestamp=False, state=None):
    # if audio == None : return ""
    time.sleep(2)
    print(datetime.now(), audio)

    # _, y = reformat_freq(*audio)
    # phrase_complete=True
    # if state is None:
    #     state=[]
    # if len(state)==0:
    #     state.append(OrderedDict())
    #     state[0]['phrase_time']=  None
    #     state[0]['last_sample'] = bytes()
    #
    #     state[0]['data_queue'] = Queue()
    #     state[0]['phrase_complete']=False
    # now = datetime.utcnow()
    # Pull raw recorded audio from the queue.
    # if not state[0]['data_queue'].empty():
    #     state[0]['phrase_complete'] = False
    #     # If enough time has passed between recordings, consider the phrase complete.
    #     # Clear the current working audio buffer to start over with the new data.
    #     if state[0]['phrase_time'] and now - state[0]['phrase_time'] > timedelta(seconds=phrase_timeout):
    #         state[0]['last_sample']  = bytes()
    #         state[0]['phrase_complete'] = True
    #     # This is the last time we received new audio data from the queue.
    #     state[0]['phrase_time']  = now

    # Concatenate our current audio data with the latest audio data.
    # while not state[0]['data_queue'].empty():
    #     data = state[0]['data_queue'].get()
    #     state[0]['last_sample'] += data

    # while True:
    try:
        results = recognize_whisper(audio_data=audio, word_timestamps=need_timestamp)
        state.append(results)
        if len(state[-1]['text'] if len(state) > 0 else '') > 0:
            print(state[-1]['text'] if len(state) > 0 else '')

        return '\n'.join([result['text'] for result in state if len(result['text']) > 0]) if len(
            state) > 0 else '', state

    except KeyboardInterrupt:
        return '\n'.join([result['text'] for result in state if len(result['text']) > 0]) if len(
            state) > 0 else '', state


def update_rolling_state(state):
    return '\n'.join([result['text'] for result in state if len(result['text']) > 0]) if len(state) > 0 else '', state


def SpeechToText(audio, need_timestamp=False, state=None):
    if audio == None: return ""
    time.sleep(1)

    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Detect the Max probability of language ?
    _, probs = model.detect_language(mel)
    language = max(probs, key=probs.get)

    #  Decode audio to Text
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    return (language, result.text)


def process_file(file, state):
    if file is None:
        return '', state
    else:
        folder, filename, ext = context.split_path(file.name)
        if file.name.lower().endswith('.pdf'):
            doc_map = get_document_text(file.name)
            return_text = ''
            for pg, offset, text in doc_map:
                return_text += text + '\n'
                return_text += 'page {0}'.format(pg + 1) + '\n''\n'
            yield return_text, state
        else:
            with open(file.name, encoding="utf-8") as f:
                content = f.read()
                print(content)
            yield content, state


def process_audio_file(file, state, initial_prompt, need_timestamp=False):
    if file is None:
        return '', state
    else:
        folder, filename, ext = context.split_path(file.name)
        transcript = ""
        chunk_start = 0
        if ext.lower() in ['.mp4', '.avi']:
            import moviepy.editor
            video = moviepy.editor.VideoFileClip(file.name)
            audio = video.audio
            context.make_dir_if_need(os.path.join(cxt.get_prompt4all_dir(), 'audio', filename + '.wav'))
            audio.write_audiofile(os.path.join(cxt.get_prompt4all_dir(), 'audio', filename + '.wav'))
            audio_file = AudioSegment.from_wav(os.path.join(cxt.get_prompt4all_dir(), 'audio', filename + '.wav'))
        elif ext.lower() in ['.mp3']:
            audio_file = AudioSegment.from_mp3(file.name)
        elif ext.lower() in ['.wav']:
            audio_file = AudioSegment.from_wav(file.name)
        load_whisper_model()
        # audio_samples = np.array(audio_file.get_array_of_samples() )   # 獲取採樣點數據陣列
        # audio_samples = audio_samples.reshape( (-1, audio_file.channels))
        # rms = np.sqrt(np.mean(audio_samples ** 2, axis=-1))
        #
        # ref = 2 ** (8 * audio_file.sample_width - 1)  # 計算參考值
        # dBFS = 20 * np.log10(np.abs(samples) / ref)  # 計算每個採樣點的分貝數

        chunk_size = 100 * 1000  # 100 秒
        chunks = [audio_file[i:i + chunk_size] for i in range(0, len(audio_file), chunk_size)]
        for chunk in chunks:
            dbfs = chunk.dBFS
            if dbfs == -math.inf or dbfs < -30:
                chunk_start += chunk.duration_seconds
                pass
            else:
                with chunk.export("temp.wav", format="wav") as f:
                    result = cxt.whisper_model.transcribe("temp.wav", word_timestamps=need_timestamp, verbose=False,
                                                          language="zh", fp16=False,
                                                          no_speech_threshold=0.5, logprob_threshold=-1,
                                                          temperature=0.2,
                                                          initial_prompt="#zh-tw 會議逐字稿。" + initial_prompt)

                    for seg in result["segments"]:
                        if need_timestamp:
                            start, end, text = seg["start"] + chunk_start, seg["end"] + chunk_start, seg["text"]
                            if len(text) == 0:
                                pass
                            else:
                                line = f"[{to_formated_time(start)} --> {to_formated_time(end)} {text}"
                                print(line, flush=True)
                                transcript += line + '\n'
                        else:
                            if len(seg['text']) == 0:
                                pass
                            else:
                                print('{0}'.format(seg['text']), flush=True)
                                transcript += '{0}'.format(seg['text']) + '\n'

                    chunk_start += chunk.duration_seconds
            yield transcript, state
        yield transcript, state


def clear_history():
    FULL_HISTORY = [{"role": "system", "content": baseChatGpt.SYSTEM_MESSAGE,
                     "estimate_tokens": estimate_used_tokens(baseChatGpt.SYSTEM_MESSAGE,
                                                             model_name=baseChatGpt.API_MODEL)}]
    return [], FULL_HISTORY, FULL_HISTORY


def reset_textbox():
    return gr.Textbox(placeholder="什麼是LLM?",value="",
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
    with gr.Blocks(title="Prompt is what you need!", css=advanced_css, analytics_enabled=False,
                   theme=adjust_theme()) as demo:
        baseChatGpt = GptBaseApi(model="gpt-4-1106-preview")
        summaryChatGpt = GptBaseApi(model="gpt-3.5-turbo-16k-0613")
        imageChatGpt = GptBaseApi(model="gpt-4-1106-preview")
        otherChatGpt = GptBaseApi(model="gpt-4-1106-preview")
        state = gr.State([{"role": "system", "content": '所有內容以繁體中文書寫',
                           "estimate_tokens": estimate_used_tokens('所有內容以繁體中文書寫',
                                                                   model_name=baseChatGpt.API_MODEL)}])  # s
        cxt.state = state
        baseChatGpt.FULL_HISTORY = state.value
        gr.HTML(title)

        with gr.Tabs():
            with gr.TabItem("對話"):
                with gr.Row():
                    with gr.Tabs():
                        with gr.TabItem("聊天"):
                            with gr.Column(scale=1):
                                with gr.Row():
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
                            with gr.Column(elem_id="col_container"):
                                conversation_history_share_btm = gr.Button('更名', scale=1, size='sm')
                                conversation_history_delete_btm = gr.Button('刪除', scale=1, size='sm')

                            with gr.Row():
                                history_list = gr.templates.List(value=cxt.conversation_history.titles, height=550,
                                                                 headers=["歷史"], datatype=["str"],
                                                                 col_count=(1, 'fixed'), interactive=True)
                    with gr.Column(scale=4, elem_id="col_container"):
                        chatbot = gr.Chatbot(elem_id='chatbot', container=True, scale=1, height=550,render_markdown=True,
                                             show_copy_button=True,bubble_full_width=True,show_share_button=True,layout="panel")
                        b3.add(chatbot)

            with gr.TabItem("歷史"):
                with gr.Column(elem_id="col_container"):
                    history_viewer = gr.JSON(elem_id='history_viewer')
            with gr.TabItem("NLU"):
                with gr.Column(elem_id="col_container"):
                    gr.Markdown(
                        "將文本輸入到下面的方塊中，按下「送出」按鈕將文本連同上述的prompt發送至OpenAI ChatGPT API，然後將返回的JSON顯示在視覺化界面上。")
                    with gr.Row():
                        with gr.Column(scale=1):
                            nlu_inputs = gr.Textbox(lines=6, placeholder="輸入句子...")
                        with gr.Column(scale=2):
                            nlu_output = gr.Text(label="回傳的JSON視覺化", interactive=True, max_lines=40,
                                                 show_copy_button=True)
                    nlu_button = gr.Button("送出")
            with gr.TabItem("Dall.E3"):
                with gr.Column(variant="panel"):
                    with gr.Row(variant="compact"):
                        image_text = gr.Textbox(
                            label="請輸入中文的描述",
                            show_label=False,
                            max_lines=1,
                            placeholder="請輸入中文的描述",
                            container=False
                        )
                    image_btn = gr.Button("設計與生成圖片", scale=1)
                    image_prompt = gr.Markdown("")
                    image_gallery = gr.Gallery(value=None, show_label=False, columns=[4], object_fit="contain",
                                               height="auto")
                with gr.Accordion("超參數", open=False):
                    temperature2 = gr.Slider(minimum=-0, maximum=2.0, value=0.7, step=0.1, interactive=True,
                                             label="溫度 (Temperature)", )
                    image_size = gr.Radio([1024], label="圖片尺寸", value=1024)
            with gr.TabItem("風格改寫"):
                with gr.Column(elem_id="col_container"):
                    rewrite_dropdown = gr.Dropdown(
                        ["抽象 (Abstract)",
                         "冒險 (Adventurous)",
                         "比喻體 (Allegorical)",
                         "曖昧 (Ambiguous)",
                         "擬人化 (Anthropomorphic)",
                         "對比 (Antithetical)",
                         "領悟 (Aphoristic)",
                         "思辯 (Argumentative)",
                         "聲音式 (Auditory)",
                         "喚醒 (Awakening)",
                         "無邊際 (Boundless)",
                         "突破 (Breakthrough)",
                         "古典 (Classical)",
                         "口語 (Colloquial)",
                         "逆袭 (Comeback)",
                         "喜劇 (Comedic)",
                         "舒適 (Comforting)",
                         "簡潔 (Concise)",
                         "自信 (Confident)",
                         "體悟 (Contemplative)",
                         "反向思考 (Counterintuitive)",
                         "勇敢 (Courageous)",
                         "創意無限 (Creative)",
                         "深奧 (Cryptic)",
                         "可愛 (Cute)",
                         "飛舞 (Dancing)",
                         "燦爛 (Dazzling)",
                         "細緻 (Delicate)",
                         "描繪 (Descriptive)",
                         "冷漠 (Detached)",
                         "保持距離 (Distant)",
                         "夢幻 (Dreamy)",
                         "優雅 (Elegant)",
                         "感性 (Emotional)",
                         "迷人 (Enchanting)",
                         "無盡 (Endless)",
                         "隱喻 (Euphemistic)",
                         "精緻 (Exquisite)",
                         "充滿信念 (Faithful)",
                         "無畏 (Fearless)",
                         "無懈可擊 (Flawless)",
                         "靈活 (Flexible)",
                         "正式 (Formal)",
                         "自由 (Free Verse)",
                         "未來主義 (Futuristic)",
                         "天賦異禀 (Gifted)",
                         "壯麗 (Grandiose)",
                         "溫馨 (Heartwarming)",
                         "豪邁 (Heroic)",
                         "幽默 (Humorous)",
                         "誇張 (Hyperbolic)",
                         "個性化 (Idiomatic)",
                         "獨立 (Independent)",
                         "強烈 (Intense)",
                         "問答 (Interrogative)",
                         "疑問 (Interrogative)",
                         "道出内心 (Introspective)",
                         "反諷 (Ironic)",
                         "歡樂 (Joyful)",
                         "傳奇 (Legendary)",
                         "人生哲理 (Life Wisdom)",
                         "抒情 (Lyric)",
                         "魔幻 (Magical)",
                         "隱喻 (Metonymic)",
                         "現代 (Modern)",
                         "神秘 (Mysterious)",
                         "敘事 (Narrative)",
                         "自然主義 (Naturalistic)",
                         "高貴 (Noble)",
                         "懷舊 (Nostalgic)",
                         "客觀 (Objective)",
                         "原聲 (Onomatopoeic)",
                         "充滿激情 (Passionate)",
                         "激情 (Passionate)",
                         "個人 (Personal)",
                         "哲學 (Philosophical)",
                         "淺白 (Plain)",
                         "俏皮 (Playful)",
                         "詩意 (Poetic)",
                         "正能量 (Positive)",
                         "實用主義 (Pragmatic)",
                         "頌揚 (Praising)",
                         "亮麗 (Radiant)",
                         "叛逆 (Rebellious)",
                         "高雅 (Refined)",
                         "文藝復興 (Renaissance)",
                         "復古 (Retro)",
                         "啟示 (Revelatory)",
                         "革命 (Revolutionary)",
                         "修辭 (Rhetorical)",
                         "諷刺 (Satirical)",
                         "科幻 (Science Fiction)",
                         "魅惑 (Seductive)",
                         "聳人聽聞 (Sensational)",
                         "感傷 (Sentimental)",
                         "銳利 (Sharp)",
                         "疑問 (Skeptical)",
                         "社會評論 (Social Commentary)",
                         "嚴肅 (Solemn)",
                         "心靈 (Soulful)",
                         "靈性 (Spiritual)",
                         "主觀 (Subjective)",
                         "奇幻 (Surreal)",
                         "懸疑 (Suspenseful)",
                         "象徵 (Symbolic)",
                         "道家 (Taoist)",
                         "格調 (Tone)",
                         "傳統 (Traditional)",
                         "超凡脫俗 (Transcendent)",
                         "過渡 (Transitional)",
                         "流行 (Trendy)",
                         "從容 (Unhurried)",
                         "奔放 (Unrestrained)",
                         "充滿活力 (Vibrant)",
                         "漫遊式 (Wandering)",
                         "溫暖 (Warm)",
                         "充滿智慧 (Wise)",
                         "俏皮 (Witty)",
                         "瑜珈式 (Yogic)",
                         "青春 (Youthful)"], value="正式 (Formal)", multiselect=False, label="改寫文字風格形容詞",
                        interactive=True)
                    gr.Markdown("將文本輸入到下面的方塊中，選取改寫風格後，點選改寫後即可將文字基於選取風格進行改寫")
                    with gr.Row():
                        with gr.Column(scale=1):
                            rewrite_inputs = gr.Textbox(lines=30, placeholder="輸入句子...")
                        with gr.Column(scale=1):
                            rewrite_output = gr.Text(label="改寫", interactive=True, lines=30, show_copy_button=True)
                    rewrite_button = gr.Button("送出")
            with gr.TabItem("長文本摘要"):
                with gr.Tabs():
                    with gr.TabItem("長文本處理"):
                        rolling_state = gr.State([])
                        text_statistics = gr.Markdown()
                        with gr.Row():
                            with gr.Column(scale=1):
                                with gr.Row():
                                    with gr.Tabs():
                                        with gr.TabItem("文字"):
                                            rolliing_source_file = gr.File(value=None, file_count="single",
                                                                           label='請將檔案拖曳至此或是點擊後上傳',
                                                                           file_types=[".txt", ".json", ".csv", ".pdf"],
                                                                           scale=2,
                                                                           elem_id='rolling_file')
                                        with gr.TabItem("影音"):
                                            whisper_timestamp_checkbox1 = gr.Checkbox(label="附加時間戳", value=True,
                                                                                      scale=1)
                                            initial_prompt_textbox = gr.Textbox(
                                                placeholder="請輸入描述影音內容的初始prompt", label="初始prompt")
                                            audio_source_file = gr.File(value=None, file_count="single",
                                                                        label='請將檔案拖曳至此或是點擊後上傳',
                                                                        file_types=[".mp3", ".mp4", ".avi", ".wav"],
                                                                        scale=2,
                                                                        elem_id='rolling_file')

                                        with gr.TabItem("即時whisper"):
                                            with gr.Row():
                                                whisper_state = gr.State([])
                                                whisper_timestamp_checkbox = gr.Checkbox(label="附加時間戳",
                                                                                         value=False, scale=1)
                                                rolling_audio = gr.Button('🎙️', size='sm', )
                                                invisible_whisper_text = gr.Text(visible=False)
                                        with gr.TabItem("Arxiv"):
                                            gr.Textbox(label="請輸入Arxiv完整網址或是論文編號")
                                        with gr.TabItem("Youtube"):
                                            gr.Textbox(label="請輸入Youtube影片完整網址")
                                            gr.Radio(["字幕檔", "音檔轉文字"], label="信息來源")
                            with gr.Column(scale=1):
                                with gr.Box():
                                    summary_radio = gr.Dropdown(
                                        ["滾動式整合摘要", "滾動式累加摘要", "平行分塊摘要"], multiselect=False,
                                        label="摘要技術", type="index",
                                        value="滾動式整合摘要", interactive=True, min_width=150)
                                    summary_options = gr.CheckboxGroup(["心智圖", "會議記錄", "重點主題"],
                                                                       label="輔助功能")
                                    rolling_button = gr.Button("▶️", size='sm', scale=1, min_width=80)
                                    rolling_clear_button = gr.ClearButton([rolliing_source_file], value="🗑️", size='sm',
                                                                          scale=1, min_width=80)
                                    rolling_cancel_button = gr.Button("⏹️", size='sm', scale=1, min_width=80)

                        with gr.Row():
                            with gr.Column(scale=1):
                                large_inputs = gr.Text(label="來源文字", lines=30, max_lines=5000)
                            with gr.Column(scale=1, elem_id="col_container"):
                                summary_output = gr.Markdown(label="摘要", interactive=True, elem_classes='markdown',
                                                             container=True)
                            rolling_clear_button.add(large_inputs)
                            rolling_clear_button.add(summary_output)
                    with gr.TabItem("存檔"):
                        with gr.Column(elem_id="col_container"):
                            with gr.Row():
                                file_obj = gr.File(label="摘要檔", file_types=[".md"], value=None, interactive=False,
                                                   min_width=60, show_label=False)
                                rolling_save_button = gr.Button("💾", size='sm', scale=1)
                    with gr.TabItem("紀錄"):
                        with gr.Column(elem_id="col_container"):
                            rolling_history_viewer = gr.JSON(elem_id='rolling_history_viewer')
            with gr.TabItem("設定"):
                with gr.Column():
                    dropdown_api1 = gr.Dropdown(choices=[k for k in model_info.keys()], value="gpt-4-1106-preview",
                                                label="對話使用之api", interactive=True)
                    dropdown_api4 = gr.Dropdown(choices=[k for k in model_info.keys()], value="gpt-4-1106-preview",
                                                label="以文生圖使用之api", interactive=True)
                    dropdown_api2 = gr.Dropdown(choices=[k for k in model_info.keys()], value="gpt-3.5-turbo-16k-0613",
                                                label="長文本摘要使用之api", interactive=True)
                    dropdown_api3 = gr.Dropdown(choices=[k for k in model_info.keys()], value="gpt-4-1106-preview",
                                                label="其他功能使用之api", interactive=True)


        inputs_event = inputs.submit(prompt_api,
                                     [inputs, context_type, top_p, temperature, top_k, frequency_penalty, state],
                                     [chatbot, state, history_viewer])
        cancel_handles.append(inputs_event)
        inputs_event.then(reset_context, [], [context_type]).then(reset_textbox, [], [inputs])
        b1_event = b1.click(prompt_api, [inputs, context_type, top_p, temperature, top_k, frequency_penalty, state],
                            [chatbot, state, history_viewer])
        cancel_handles.append(b1_event)
        b1_event.then(reset_context, [], [context_type]).then(reset_textbox, [], [inputs])
        b3.click(clear_history, [], [chatbot, state, history_viewer]).then(reset_textbox, [], [inputs])
        b2.click(fn=None, inputs=None, outputs=None, cancels=cancel_handles)
        chatbot.change(scroll_to_output=True)


        def new_chat(state):
            cxt.conversation_history.new_chat()
            history_list.value = cxt.conversation_history.titles
            chat=cxt.conversation_history.selected_item.get_gradio_chat()
            return chat,state


        b4.click(fn=new_chat, inputs=[state], outputs=[chatbot, state])

        nlu_inputs.submit(nlu_api, nlu_inputs, nlu_output)
        nlu_button.click(nlu_api, nlu_inputs, nlu_output)

        image_text.submit(image_api, [image_text, image_size, temperature2], [image_prompt, image_gallery])
        image_btn.click(image_api, [image_text, image_size, temperature2], [image_prompt, image_gallery])

        rewrite_inputs.submit(rewrite_api, [rewrite_inputs, rewrite_dropdown], rewrite_output)
        rewrite_button.click(rewrite_api, [rewrite_inputs, rewrite_dropdown], rewrite_output)

        rolling_cancel_handel = []

        rolling_inputs_event = rolling_button.click(rolling_summary,
                                                    [large_inputs, rolling_state, summary_radio, summary_options],
                                                    [summary_output, rolling_state]).then(estimate_tokens,
                                                                                          [large_inputs, summary_output,
                                                                                           rolling_state],
                                                                                          [text_statistics,
                                                                                           rolling_state])
        # large_inputs.submit(rolling_summary.md, [large_inputs, rolling_state,rolling_parallel_checkbox], [summary_output, rolling_state]).then(estimate_tokens, [large_inputs,summary_output, rolling_state],[text_statistics,rolling_state])
        large_inputs_change_event = large_inputs.change(estimate_tokens, [large_inputs, summary_output, rolling_state],
                                                        [text_statistics, rolling_state])
        source_file_change_event = rolliing_source_file.change(process_file, [rolliing_source_file, rolling_state],
                                                               [large_inputs, rolling_state])
        audio_file_change_event = audio_source_file.change(process_audio_file,
                                                           [audio_source_file, whisper_state, initial_prompt_textbox,
                                                            whisper_timestamp_checkbox1], [large_inputs, whisper_state])
        rolling_cancel_handel.append(rolling_inputs_event)
        rolling_cancel_handel.append(large_inputs_change_event)
        rolling_cancel_handel.append(source_file_change_event)
        rolling_cancel_handel.append(audio_file_change_event)
        rolling_cancel_button.click(fn=None, inputs=None, outputs=None, cancels=rolling_cancel_handel)


        def select_conversation(evt: gr.SelectData):
            conversations = cxt.conversation_history.conversations[evt.index[0]].get_prompt_messages(only_final=True)
            cxt.conversation_history.selected_index = evt.index[0]
            chat=cxt.conversation_history.selected_item.get_gradio_chat()
            return chat,cxt.state


        history_list.select(select_conversation, None, [chatbot, state])


        def save_file(contents, state):
            text_file = "generate_text/summary_{0}.txt".format(
                str(datetime.now()).replace(' ', '').replace(':', '').replace('-', '').replace('.', ''))
            if summary_radio.value == 0:
                text_file = "generate_text/rolling_summary_{0}.txt".format(
                    str(datetime.now()).replace(' ', '').replace(':', '').replace('-', '').replace('.', ''))
            elif summary_radio.value == 1:
                text_file = "generate_text/incremental_rolling_summary_{0}.txt".format(
                    str(datetime.now()).replace(' ', '').replace(':', '').replace('-', '').replace('.', ''))
            elif summary_radio.value == 2:
                text_file = "generate_text/parallel_summary_{0}.txt".format(
                    str(datetime.now()).replace(' ', '').replace(':', '').replace('-', '').replace('.', ''))
            elif summary_radio.value == 3:
                text_file = "generate_text/mindmap_summary_{0}.txt".format(
                    str(datetime.now()).replace(' ', '').replace(':', '').replace('-', '').replace('.', ''))
            elif summary_radio.value == 4:
                text_file = "generate_text/meeting_summary_{0}.txt".format(
                    str(datetime.now()).replace(' ', '').replace(':', '').replace('-', '').replace('.', ''))
            elif summary_radio.value == 5:
                text_file = "generate_text/topic_summary_{0}.txt".format(
                    str(datetime.now()).replace(' ', '').replace(':', '').replace('-', '').replace('.', ''))

            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(contents)
            return text_file, state


        rolling_save_button.click(save_file, [summary_output, rolling_state], [file_obj, rolling_state])

        invisible_whisper_text.change(update_rolling_state, [whisper_state], [large_inputs, rolling_history_viewer])

        dropdown_api1.change(lambda x: baseChatGpt.change_model(x), [dropdown_api1], [])
        dropdown_api2.change(lambda x: summaryChatGpt.change_model(x), [dropdown_api2], [])
        dropdown_api3.change(lambda x: otherChatGpt.change_model(x), [dropdown_api3], [])
        dropdown_api4.change(lambda x: imageChatGpt.change_model(x), [dropdown_api4], [])

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
        demo.queue(concurrency_count=5, api_open=False).launch(show_error=True, max_threads=200, share=True,
                                                               server_port=PORT)
