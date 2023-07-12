import builtins
import markdown
import copy
from collections import OrderedDict
from prompt4all.utils.regex_utils import is_numbered_list_member,extract_numbered_list_member
from prompt4all.utils.tokens_utils import estimate_used_tokens
__all__ = [ "get_rolling_summary_results", "get_last_ordered_index",'aggregate_summary', "split_summary","text2markdown"]


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



def get_rolling_summary_results(answer):
    content_dict=OrderedDict()

    examples_parts=["1. 第一個項目","1.1 第一個子項目","1.2 第二個子項目","2. 第二個項目"]
    for part in examples_parts:
        if part in answer:
            answer=answer.replace(part,"")
    lines = answer.split('\n')


    start=0
    header=None
    is_content_start=False
    for i in range(len(lines)):
        if  ':' in lines[i]  and not is_numbered_list_member(lines[i].split(':')[0]) and header is None:
            first_line = lines[i].strip().replace("\"\"\"", "")
            header = first_line.split(':')[0]
            is_content_start=False
            if len(first_line.split(':')[1].strip().replace(" ", ""))>0:
                lines[i]=first_line.split(':')[1]
                start = i
            else:
                lines[i] =""
                start = i+1
        elif header is None and not is_numbered_list_member(lines[i]):
            start = i + 1
        elif not is_content_start and is_numbered_list_member(lines[i]):
            if header is  None:
                header='tmp摘要'
            is_content_start=True
            start = i
        elif is_content_start and (i<len(lines)-1 and (":" in  lines[i+1]  )  or (i==len(lines)-1 and start<i)):
            content = lines[start:i]
            content=[c for c in content if len(c.strip())>0]
            content_dict[header] = content
            header = None
            is_content_start=False
            start=i+1
        elif  is_content_start and is_numbered_list_member(lines[i]):
            pass
        elif  is_content_start and  len(lines[i])>0 and not is_numbered_list_member(lines[i]) :
            pass
        else:
            print(lines[i] )

    keys=list(content_dict)
    keys=[k for k in keys if '輸出' in k or '摘要' in k]
    if len(keys) == 0:
        return []
    elif len(keys)==1:
        return [t for t in content_dict[keys[0]] if len(t.replace('\"\"\"',''))>0 and is_numbered_list_member(t)]
    elif len(keys)>1:
        if '輸入摘要清單' in keys:
            return [t for t in content_dict['輸入摘要清單'] if len(t.replace('\"\"\"', '')) > 0 and is_numbered_list_member(t)]

        item_check=[len([item for item in content_dict[k] if  len(item.replace('\"\"\"', '')) > 0 and is_numbered_list_member(item)]) for k in keys]
        max_items=builtins.max(item_check)
        return [t for t in content_dict[keys[item_check.index(max_items)]]if len(t.replace('\"\"\"',''))>0 and is_numbered_list_member(t)]



def aggregate_summary(results):
    aggs=[]
    for result in results:
        if isinstance(result,dict):
            items=[line for line in result['content'].split('\n') if is_numbered_list_member(line)]
            if all([item[:4]=="    " for item in items]):
                items=[item[4:]for item in items]
            aggs.extend(items)
        elif isinstance(result,str):
            if len(aggs) == 0:
                aggs.append(result.split('\n')[0])
            aggs.extend([c for c in result.split('\n') if c.startswith('-')])
    return aggs
def split_summary(summary_list,max_tokens):
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




