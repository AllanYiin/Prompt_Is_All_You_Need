from sqlalchemy import create_engine
import sqlalchemy
import json
from prompt4all import context
from prompt4all.context import *
from prompt4all.utils import regex_utils
import pandas as pd
import struct
from openai import OpenAI
client = OpenAI()
cxt=context._context()
engine =create_engine(cxt.conn_string)

def query_sql(query_intent:str):

    query_prompt='基於以下資料結構，請為我撰寫可以查詢出"{0}"的sql語法，盡量避免CTE，請善用子查詢以及WHERE條件式來篩選案例以提升計算效率，注意別犯除以零的錯誤以及別在排序時重複引用同個欄位，直接輸出，無須解釋。\n"""\n{1}\n"""\n'.format(query_intent,cxt.databse_schema)
    is_success=False
    this_query_prompt = query_prompt
    retry=0
    while not is_success and retry<3:

        response=client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
            {'role': 'user', 'content':this_query_prompt}
            ],
            temperature=0.1,
            n=1,
            presence_penalty=1,
            stream=False
        )
        response_message = response.choices[0].message
        try:
            tsql=regex_utils.extract_code(response_message.content)
            if tsql:
                with engine.begin() as conn:
                    df = pd.DataFrame(conn.execute(sqlalchemy.text(tsql).execution_options(autocommit=True)))
                is_success=True
                save_query_cache(query_intent,tsql)
                return r'"""\n#Database query results\n\n##tsql\n\n{0}\n\n##queryresults\n\n{1}\n\n"""'.format(tsql,df.to_string(index=False))
            else:
                raise RuntimeError('Get No SQL')
        except Exception as e:
            print(e)
            retry+=1
            this_query_prompt=query_prompt+'\n'+str(e)


def save_query_cache(query_intent,  generated_tsql):
    def get_embedding(text):
        text = text.replace("\n", " ")
        response=client.embeddings.create(
            model="text-embedding-ada-002",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding
    try:
        embedding = get_embedding(query_intent)
        embedding_string =','.join([str(num) for num in embedding])
        with engine.begin() as conn:
            query = sqlalchemy.text("INSERT INTO SqlRAG.dbo.QueryIntentCache(QueryIntent, VectorizedQueryIntent, GeneratedTSQL) VALUES (:query_intent, :embedding, :generated_tsql)").execution_options(autocommit=True)
            result = conn.execute(query, {"query_intent": query_intent, "embedding": embedding_string , "generated_tsql": generated_tsql})
            print(query.text)
    except Exception as e:
        print(e)








