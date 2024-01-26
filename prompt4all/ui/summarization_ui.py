import gradio as gr
from prompt4all import context
from prompt4all.utils.io_utils import process_file
from prompt4all.utils.whisper_utils import *
from prompt4all.common import *
from prompt4all.utils.summary_utils import *
from prompt4all.utils.pdf_utils import *
from prompt4all.utils.tokens_utils import estimate_used_tokens
import copy
import glob

cxt = context._context()


def estimate_tokens(text, text2, state):
    text = '' if text is None else text
    text2 = '' if text2 is None else text2
    t1 = '輸入文本長度為{0},預計耗用tokens數為:{1}'.format(len(text),
                                                           estimate_used_tokens(text, cxt.summaryChatGpt.API_MODEL) + 4)
    if len(text2) == 0:
        return t1, state
    else:
        t2 = '輸出文本長度為{0},預計耗用tokens數為:{1}'.format(len(text2), estimate_used_tokens(text2,
                                                                                                cxt.summaryChatGpt.API_MODEL) + 4)
        return t1 + '\t\t' + t2, state


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
    _parameters = copy.deepcopy(cxt.summaryChatGpt.API_PARAMETERS)
    _parameters['temperature'] = 0.001
    _parameters['presence_penalty'] = 1.2
    payload = cxt.summaryChatGpt.parameters2payload(cxt.summaryChatGpt.API_MODEL, conversation, _parameters,
                                                    stream=False)

    response = await asyncio.to_thread(
        requests.post,
        cxt.summaryChatGpt.BASE_URL, headers=cxt.summaryChatGpt.API_HEADERS, json=payload, stream=False
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
    _parameters = copy.deepcopy(cxt.summaryChatGpt.API_PARAMETERS)
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
                                                                                         model_name=cxt.summaryChatGpt.API_MODEL) + 4
        this_final_tokens = estimate_used_tokens(_final_prompt) + estimate_used_tokens('system',
                                                                                       model_name=cxt.summaryChatGpt.API_MODEL) + 4
        available_tokens = cxt.summaryChatGpt.MAX_TOKENS - this_system_tokens - 4 - 2
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
                        available_tokens = cxt.summaryChatGpt.MAX_TOKENS - this_final_tokens - 4 - 2
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

                _max_tokens = builtins.min(cxt.summaryChatGpt.MAX_TOKENS,
                                           estimate_used_tokens(str(conversation)) + estimate_used_tokens(
                                               '\n'.join(partial_words)) * (0.3 if not is_final_stage else 1))
                _parameters['max_tokens'] = _max_tokens

                streaming_answer = cxt.summaryChatGpt.post_and_get_streaming_answer(conversation, _parameters,
                                                                                    full_history)
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
                                                                                         model_name=cxt.summaryChatGpt.API_MODEL) + 4
        this_final_tokens = estimate_used_tokens(_final_prompt) + estimate_used_tokens('system',
                                                                                       model_name=cxt.summaryChatGpt.API_MODEL) + 4
        available_tokens = cxt.summaryChatGpt.MAX_TOKENS - this_system_tokens - 4 - 2
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
                        available_tokens = cxt.summaryChatGpt.MAX_TOKENS - this_final_tokens - 4 - 2
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

                _max_tokens = builtins.min(cxt.summaryChatGpt.MAX_TOKENS,
                                           estimate_used_tokens(str(conversation)) + estimate_used_tokens(
                                               '\n'.join(partial_words)) * (0.3 if not is_final_stage else 1))
                _parameters['max_tokens'] = _max_tokens

                streaming_answer = cxt.summaryChatGpt.post_and_get_streaming_answer(conversation, _parameters,
                                                                                    full_history)
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
                                                                                         model_name=cxt.summaryChatGpt.API_MODEL) + 4
        this_final_tokens = estimate_used_tokens(_final_prompt) + estimate_used_tokens('system',
                                                                                       model_name=cxt.summaryChatGpt.API_MODEL) + 4
        available_tokens = cxt.summaryChatGpt.MAX_TOKENS - this_system_tokens - 4 - 2
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
                _parameters = copy.deepcopy(cxt.summaryChatGpt.API_PARAMETERS)
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

                cxt.summaryChatGpt.make_response()
                payload = cxt.summaryChatGpt.parameters2payload(cxt.summaryChatGpt.API_MODEL, conversation, _parameters,
                                                                stream=False)
                response = requests.post(cxt.summaryChatGpt.BASE_URL, headers=cxt.summaryChatGpt.API_HEADERS,
                                         json=payload,
                                         stream=False)
                return_values[k] = json.loads(response.content.decode())['choices'][0]["message"]

        all_summary = aggregate_summary(return_values)
        is_final_stage = True
        keep_summary = False
        available_tokens = cxt.summaryChatGpt.MAX_TOKENS - this_final_tokens - 4 - 2
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
            streaming_answer = cxt.summaryChatGpt.post_and_get_streaming_answer(conversation, _parameters, full_history)
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
                                                                                         model_name=cxt.summaryChatGpt.API_MODEL) + 4
        available_tokens = cxt.summaryChatGpt.MAX_TOKENS - this_system_tokens - 4 - 2
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
                streaming_answer = cxt.summaryChatGpt.post_and_get_streaming_answer(conversation, _parameters,
                                                                                    full_history)
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
                available_tokens = int((cxt.summaryChatGpt.MAX_TOKENS - 200 - estimate_used_tokens(
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
                                                                                         model_name=cxt.summaryChatGpt.API_MODEL) + 4
        this_final_tokens = estimate_used_tokens(_final_prompt) + estimate_used_tokens('system',
                                                                                       model_name=cxt.summaryChatGpt.API_MODEL) + 4
        available_tokens = cxt.summaryChatGpt.MAX_TOKENS - this_system_tokens - 4 - 2
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
                streaming_answer = cxt.summaryChatGpt.post_and_get_streaming_answer(conversation, _parameters,
                                                                                    full_history)
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
                available_tokens = int((cxt.summaryChatGpt.MAX_TOKENS - 200 - estimate_used_tokens(
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
                                                                                         model_name=cxt.summaryChatGpt.API_MODEL) + 4
        this_final_tokens = estimate_used_tokens(_final_prompt) + estimate_used_tokens('system',
                                                                                       model_name=cxt.summaryChatGpt.API_MODEL) + 4
        available_tokens = cxt.summaryChatGpt.MAX_TOKENS - this_system_tokens - 4 - 2
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
                streaming_answer = cxt.summaryChatGpt.post_and_get_streaming_answer(conversation, _parameters,
                                                                                    full_history)
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
                available_tokens = int((cxt.summaryChatGpt.MAX_TOKENS - 200 - estimate_used_tokens(
                    topic_shortcuts) - this_system_tokens - 4 - 2) * 0.667)
                cnt += 1

        except Exception as e:
            raise gr.Error(str(e))
        yield answer_head.format(text2markdown('\n'.join(cleansed_summary))) + '\n\n\n' + mindmap_history, full_history


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


def summerization_panel():
    with gr.Tabs() as _panel:
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
                    with gr.Group():
                        summary_radio = gr.Dropdown(
                            ["滾動式整合摘要", "滾動式累加摘要", "平行分塊摘要"], multiselect=False,
                            label="摘要技術", type="index",
                            value="滾動式整合摘要", interactive=True, min_width=150)
                        summary_options = gr.CheckboxGroup(["心智圖", "會議記錄", "重點主題"],
                                                           label="輔助功能")
                        with gr.Row():
                            rolling_button = gr.Button("▶️", size='sm', scale=1, min_width=80)
                            rolling_clear_button = gr.ClearButton([rolliing_source_file], value="🗑️",
                                                                  size='sm',
                                                                  scale=1, min_width=80)
                            rolling_cancel_button = gr.Button("⏹️", size='sm', scale=1, min_width=80)

            with gr.Row():
                with gr.Column(scale=1):
                    large_inputs = gr.Text(label="來源文字", lines=30, max_lines=5000)
                with gr.Column(scale=1, elem_id="col_container"):
                    summary_output = gr.Markdown(label="摘要", elem_classes='markdown')
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

    return _panel
