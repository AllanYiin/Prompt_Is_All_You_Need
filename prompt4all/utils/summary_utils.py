import os
import re
import builtins
import markdown
import copy
import string
from collections import OrderedDict
from prompt4all.utils.regex_utils import is_numbered_list_member,extract_numbered_list_member
from prompt4all.utils.tokens_utils import estimate_used_tokens
__all__ = [ "get_rolling_summary_results", "get_last_ordered_index", "split_summary","text2markdown",'reorder_list','aggregate_summary']


def text2markdown(txt):
    lines=txt.split('\n')
    for i in range(len(lines)):
        if is_numbered_list_member(lines[i]):
            if extract_numbered_list_member(lines[i]).endswith('.'):
                lines[i]= lines[i].lstrip()
            elif not extract_numbered_list_member(lines[i]).endswith('.'):
                ex_value=extract_numbered_list_member(lines[i])
                lines[i] = ''.join(['&emsp;']*len([ t for t in ex_value if t=='.']))+lines[i].lstrip()
    lines=[t for t in lines if len(t)>0]
    return '    \n'.join(lines)


def reorder_list(lst):
    # 將列表轉換為字符串
    lst_str = '\n'.join(lst)
    # 使用正則表達式查找所有的標號
    pattern = r'(\d+\.)+'
    matches = re.findall(pattern, lst_str)
    # 將所有的標號替換為新的標號
    for i, match in enumerate(matches):
        new_mark = f'{i+1}.'
        lst_str = lst_str.replace(match, new_mark)
    # 將字符串轉換回列表
    new_lst = lst_str.split('\n')
    return new_lst


def get_rolling_summary_results(answer):
    content_dict=OrderedDict()
    lines = answer.split('\n') if isinstance(answer,str) else answer
    #remove blank row
    lines=[t for t in lines if len(t.strip(' \t\r'))>0]
    start = -1
    number_start=-1
    this_num=-1
    last_num=-1
    for i in range(len(lines)):
        if start<0:
            start=i
        if number_start<0 and is_numbered_list_member(lines[i]):
            number_start=i
            this_num=int(extract_numbered_list_member(lines[i]).split('.')[0])

        if i>0 and number_start > 0 and not is_numbered_list_member(lines[i]) and is_unordered_list_member(lines[i]):
            offset1=len(lines[i])-len(lines[i].lstrip(' \t\r　'))
            offset0 = len(lines[i-1]) - len(lines[i-1].lstrip(' \t\r　'))
            if offset1-offset0==0:
                if extract_numbered_list_member(lines[i-1]).endswith('.'):
                    lines[i]='{0}. '.format(this_num+1)+lines[i].lstrip(' \t\r　')
                else:
                    last_num_piece=last_num.split('.')
                    last_num_piece[-1]=str(int(last_num_piece[-1])+1)
                    lines[i] = ''.join((['    ']*len(last_num_piece)-1))+' '+'.'.join(last_num_piece)+' '+ lines[i].lstrip(' \t\r　')
            elif offset1-offset0>0:
                if extract_numbered_list_member(lines[i-1]).endswith('.'):
                    lines[i]=regex.sub(numbered_list_member_pattern,extract_numbered_list_member(lines[i-1])+"1",lines[i])
                else:
                    lines[i] = regex.sub(numbered_list_member_pattern, extract_numbered_list_member(lines[i - 1]) + ".1",lines[i])
            elif offset1 - offset0 <0:
                    last_num_piece = last_num.split('.')
                    lines[i] = regex.sub(numbered_list_member_pattern, extract_numbered_list_member(lines[i - 1]) + ".1",lines[i])

        if number_start>0 and is_numbered_list_member(lines[i]):
            this_num=int(extract_numbered_list_member(lines[i]).split('.')[0])
            last_num=extract_numbered_list_member(lines[i])

        if number_start>=0 and ((i==len(lines)-1) or ((i<len(lines)-1) and is_numbered_list_member(lines[i]) and not is_numbered_list_member(lines[i+1]) )):
            header_canidate=lines[start:number_start]
            header_canidate=[t.strip(' \t\n\r') for t in header_canidate if len(t.strip(' \t\n\r'))>0]
            header=None
            if len(header_canidate)>0:
                for line in header_canidate:
                    if '輸出摘要清單' in line:
                        header='輸出摘要清單'
                        break
                    elif '輸出' in line:
                        header=line.replace(':','')
                        break
                    elif line.endswith(':'):
                        header=line[:-1]
                        break
            if header is None:
                header ='輸出'
            if header not in content_dict:
                content_dict[header]=lines[number_start:i+1]
            else:
                content_dict[header+'_'] = lines[number_start:i + 1]
            # else:
            #     if '輸出' not in content_dict:
            #         content_dict[ '輸出'] = lines[number_start:i + 1]
            #     else:
            #         content_dict[ '輸出_'] = lines[number_start:i + 1]
            start=i+1
            number_start=-1
            last_num=-1
    if len(content_dict) == 0:
        return []
    elif len(content_dict)==1:
        return content_dict[list(content_dict.keys())[0]]
    else:
        if '輸出摘要清單' in content_dict:
            return content_dict['輸出摘要清單']
        else:
            value_list=[len(k) for k,v in content_dict.items()]
            max_value=builtins.max(value_list)
            return content_dict[list(content_dict.keys())[value_list.index(max_value)]]


def convert_bullet_to_number_list(content, linesep=os.linesep):
    """Replace bullet point list with number list, for example

    Sample input:
    - 第03章新商業智慧平台安裝與設定
      - 安裝SSRS 2012的前置需求
        - 版本限制
          - 標準版（Standard Edition）
            - 提供報表設計、管理和部署功能
            - 不支援進階功能如Power View、資料驅動訂閱和Web Farm架構
          - 商業智慧版（Business Intelligence Edition）

    Sample output:
    1. 第03章新商業智慧平台安裝與設定
      1.1 安裝SSRS 2012的前置需求
        1.1.1 版本限制
          1.1.1.1 標準版（Standard Edition）
            1.1.1.1.1 提供報表設計、管理和部署功能
            1.1.1.1.2 不支援進階功能如Power View、資料驅動訂閱和Web Farm架構
          1.1.1.2 商業智慧版（Business Intelligence Edition）
    """
    lines = content.split(linesep)

    # region Step 1: Find number of spaces used for indentation
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    min_indent = min(indents)
    other_indents = [indent for indent in indents if indent > min_indent]

    if not other_indents:  # the same indent for all or it's a single line
        return min_indent

    indent_unit = min(other_indents) - min_indent
    # endregion

    def process_line(line, counters, indent_unit):
        num_spaces = len(line) - len(line.lstrip())
        indent_level = num_spaces // indent_unit

        if indent_level >= len(counters):
            counters.extend([0] * (indent_level - len(counters) + 1))
        else:
            counters = counters[: indent_level + 1]

        counters[-1] += 1
        numbering = ".".join(map(str, counters))

        # if there is only one number, add a dot at the end
        if numbering.count(".") == 0:
            numbering += "."

        new_line = re.sub(r"^(\s*)\S+\s+", r"\1", line)  # del any list marker
        new_line = f"{' ' * num_spaces}{numbering} {new_line.lstrip()}"  # add marker

        return new_line, counters

    counters = []
    new_lines = []

    for line in lines:
        new_line, counters = process_line(line, counters, indent_unit)
        new_lines.append(new_line)

    return linesep.join(new_lines)



def aggregate_summary(results):
    aggs=[]
    raw_lines = []
    for result in results:
        if isinstance(result,dict):
            lines = result['content'].split(os.linesep)
            items = [line for line in lines if is_numbered_list_member(line)]
            if all([item[:4]=="    " for item in items]):
                items=[item[4:]for item in items]
            aggs.extend(items)
            raw_lines.extend(lines)
        elif isinstance(result,str):
            if len(aggs) == 0:
                aggs.append(result.split('\n')[0])
            aggs.extend([c for c in result.split('\n') if c.startswith('-')])
    # plan b: generate num list from raw lines when it's not num list
    if len(aggs) == 0:
        # print(os.linesep.join(raw_lines)) # for debug
        # print("=" * 80) # for debug
        aggs = convert_bullet_to_number_list(raw_lines)
        # print(os.linesep.join(aggs)) # for debug
    aggs = os.linesep.join(aggs) # convert to a string for display
    return aggs

def split_summary(summary_list,max_tokens):
    if isinstance(summary_list,str):
        summary_list=summary_list.split('\n')
    total_tokens=builtins.sum([estimate_used_tokens(w) + 1 for w in summary_list])
    if max_tokens>=total_tokens:
        return summary_list, []
    results=[]
    current_tokens=0
    bk_summary_list=copy.deepcopy(summary_list)
    for summary in summary_list:
        this_tokens=estimate_used_tokens(summary)+1
        if current_tokens+this_tokens>max_tokens:
            break
        else:
            results.append(summary)
            bk_summary_list.pop(0)
            current_tokens+=this_tokens

    if len(bk_summary_list)>0 and extract_numbered_list_member(bk_summary_list[0]).endswith('.'):
        return results,bk_summary_list
    else:
        while True:
            line=results.pop(-1)
            bk_summary_list.insert(0,line)
            if len(bk_summary_list)>0 and extract_numbered_list_member(bk_summary_list[0]).endswith('.'):
                return results, bk_summary_list
        return results, bk_summary_list


def get_last_ordered_index(summary_list):
    summary_list=[s for s in summary_list if is_numbered_list_member(s)]
    if len(summary_list)==0:
        return 1
    else:
        line = summary_list[- 1]
        line_number = extract_numbered_list_member(line)
        return int(line_number.split('.')[0])+1




