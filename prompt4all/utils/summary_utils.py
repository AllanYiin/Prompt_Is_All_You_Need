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



def aggregate_summary(results):
    aggs=[]
    for  result in results:
        if isinstance(result,dict):
            aggs.extend(get_rolling_summary_results(result['content']))
        elif isinstance(result,str):
            aggs.extend(get_rolling_summary_results([c for c in result.split('\n')]))
    return '\n'.join(aggs)


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




