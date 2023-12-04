from sqlalchemy import create_engine
import sqlalchemy
import json
from prompt4all import context
from prompt4all.context import *
from prompt4all.utils import regex_utils
import pandas as pd
import struct
from openai import OpenAI
import gradio as gr

client = OpenAI()
client._custom_headers['Accept-Language'] = 'zh-TW'
cxt = context._context()


def build_connection():
    try:
        if cxt.sql_engine is None:
            cxt.sql_engine = create_engine(cxt.conn_string)

    except Exception as e:
        gr.Error(str(e))


def query_sql(query_intent: str):
    query_prompt = '基於以下資料結構，請為我撰寫可以回答"{0}"的sql語法，在語法中用來排序、篩選所用到的量值或欄位，或是計算某個量值所用到的分子與分母，請在你的SQL語法中盡量保留，盡量避免CTE，請善用子查詢以及WHERE條件式來篩選案例以提升計算效率，注意別犯除以零的錯誤以及別在排序時重複引用同個欄位，直接輸出，無須解釋。\n"""\n{1}\n"""\n'.format(
        query_intent, cxt.databse_schema)
    is_success = False
    this_query_prompt = query_prompt
    retry = 0
    while not is_success and retry < 3:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {'role': 'system', 'content': '#zh-TW 你是一個有多年經驗、熟悉T-SQL語法的數據科學家'},
                {'role': 'user', 'content': this_query_prompt}
            ],
            temperature=0.3,
            n=1,
            presence_penalty=0,
            stream=False
        )
        response_message = response.choices[0].message
        try:
            tsql = regex_utils.extract_code(response_message.content)
            if tsql:
                build_connection()
                with cxt.sql_engine.begin() as conn:
                    df = pd.DataFrame(conn.execute(sqlalchemy.text(tsql).execution_options(autocommit=True)))
                is_success = True
                save_query_cache(query_intent, tsql)
                return r'"""\n#資料庫查詢相關內容(請根據查詢結果回答)  \n\n##查詢使用的t-sql  \n\n{0}  \n\n##查詢結果  \n\n  {1}\n\n  """'.format(
                    tsql, df.to_string(index=False))
            else:
                raise RuntimeError('Get No SQL')
        except Exception as e:
            print(e)
            retry += 1
            this_query_prompt = query_prompt + '\n' + str(e)


def save_query_cache(query_intent, generated_tsql):
    def get_embedding(text):
        text = text.replace("\n", " ")
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    try:
        embedding = get_embedding(query_intent)
        embedding_string = ','.join([str(num) for num in embedding])

        with cxt.sql_engine.begin() as conn:
            query = sqlalchemy.text(
                "INSERT INTO SqlRAG.dbo.QueryIntentCache(QueryIntent, VectorizedQueryIntent, GeneratedTSQL) VALUES (:query_intent, :embedding, :generated_tsql)").execution_options(
                autocommit=True)
            result = conn.execute(query, {"query_intent": query_intent, "embedding": embedding_string,
                                          "generated_tsql": generated_tsql})
            print(query.text)
    except Exception as e:
        print(e)
