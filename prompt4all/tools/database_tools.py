from sqlalchemy import create_engine, MetaData, Table
import sqlalchemy
import json
from prompt4all import context
from prompt4all.context import *
from prompt4all.common import *
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
    ##檢查這意圖是否已經存在查詢意圖快取
    if len(query_intent.strip()) > 0:
        build_connection()
        cxt.status_word = "查詢意圖快取確認中..."
        print(cxt.status_word)
        with cxt.sql_engine.begin() as conn:
            query = sqlalchemy.text("select SqlRAG.dbo.GetCachedSQL(:question)").execution_options(autocommit=True)
            old_sql = conn.scalars(query, {"question": query_intent}).first()

        if old_sql and len(old_sql) > 10:
            tsql = old_sql
            try:
                cxt.status_word = "資料庫查詢中..."
                print(cxt.status_word)
                with cxt.sql_engine.begin() as conn:
                    df = pd.DataFrame(conn.execute(sqlalchemy.text(tsql).execution_options(autocommit=True)))
                is_success = True
                save_query_cache(query_intent, tsql, '\n\n{0}\n\n'.format(df.to_string(index=False)))
                return r'"""\n#資料庫查詢相關內容(請根據查詢結果回答)  \n\n##查詢使用的t-sql  \n\n{0}  \n\n##查詢結果  \n\n  {1}\n\n  """'.format(
                    tsql, df.to_string(index=False))
            except Exception as e:
                save_query_cache(query_intent, tsql, exec_status=str(e))
                print(e)
        cxt.status_word = "生成T-SQL語法中..."
        print(cxt.status_word)
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
            except Exception as e:
                save_query_cache(query_intent, tsql, exec_status=str(e))
                print(e)
                retry += 1
                this_query_prompt = query_prompt + '\n' + str(e)
            if tsql:
                try:
                    cxt.status_word = "資料庫查詢中..."
                    print(cxt.status_word)
                    build_connection()
                    with cxt.sql_engine.begin() as conn:
                        df = pd.DataFrame(conn.execute(sqlalchemy.text(tsql).execution_options(autocommit=True)))
                    is_success = True
                    save_query_cache(query_intent, tsql, '\n\n{0}\n\n'.format(df.to_string(index=False)))
                    return r'"""\n#資料庫查詢結果  \n\n##查詢使用的t-sql  \n\n{0}  \n\n##查詢結果  \n\n  {1}\n\n  """'.format(
                        tsql, df.to_string(index=False))
                except Exception as e:
                    save_query_cache(query_intent, tsql, exec_status=str(e))
                    print(e)
            else:
                raise RuntimeError('Get No SQL')


def save_query_cache(query_intent, generated_tsql, generated_data='', exec_status=''):
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
                "Exec SQLRAG.dbo.InsertQueryIntentCache :query_intent,:embedd, :tsql,:data,:exec_status").execution_options(
                autocommit=True)
            result = conn.execute(query, {"query_intent": query_intent, "embedd": embedding_string,
                                          "tsql": generated_tsql, "data": generated_data,
                                          "exec_status": exec_status})
            print(cyan_color(query.text))
    except Exception as e:
        print(magenta_color(e))


def save_knowledge_base(part_id, text_content, parent_id=None, ordinal=None, is_rewrite=0, source_type=None, url=None,
                        raw=None):
    try:
        build_connection()
        with cxt.sql_engine.begin() as conn:
            query = sqlalchemy.text(
                "Exec SQLRAG.dbo.InsertKnowledgeBase :id,:parent_id,:ordinal,:is_rewrite, :source_type,:url,:text_content,:raw").execution_options(
                autocommit=True)
            result = conn.execute(query, {"id": part_id, "parent_id": parent_id,
                                          "ordinal": ordinal, "is_rewrite": is_rewrite,
                                          "source_type": source_type, "url": url,
                                          "text_content": text_content, "raw": raw})
            print(cyan_color(query.text))
    except Exception as e:
        print(magenta_color(e))


def save_webpilot_log(query_intent, generated_tsql, generated_data=None, exec_status=None):
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
                "INSERT INTO SqlRAG.dbo.WebPilotLogs(QueryIntent, VectorizedQueryIntent, GeneratedTSQL,GeneratedData,ExecStatus) VALUES (:query_intent, :embedding, :generated_tsql,:generated_data,:exec_status)").execution_options(
                autocommit=True)
            result = conn.execute(query, {"query_intent": query_intent, "embedding": embedding_string,
                                          "generated_tsql": generated_tsql, "generated_data": generated_data,
                                          "exec_status": exec_status})
            print(query.text)
    except Exception as e:
        print(e)
