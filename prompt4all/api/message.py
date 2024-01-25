import os
import os.path
import shutil
from collections import OrderedDict
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Union
import copy
from abc import ABC
import itertools
from prompt4all import context
from prompt4all.context import split_path
from prompt4all.common import unpack_singleton
from prompt4all.utils.chatgpt_utils import *

__all__ = ["ConversationHistory", "Conversation", "Message", "Mapping", "Content", "Author",
           "initialize_conversation_history"]

cxt = context._context()


def ymdhms_to_timestamp_utc(datetime_value) -> int:
    naive_dt = datetime_value
    utc_dt = naive_dt.replace(tzinfo=timezone.utc)
    return str(int(utc_dt.timestamp()))


class Author:
    def __init__(self, role: str = None, name: str = None, metadata: dict = None):
        super(Author, self).__init__()
        self.role = role
        self.name = name
        self.metadata = metadata

    def __repr__(self):
        return json.dumps(self.__dict__.copy(), ensure_ascii=False)


class Content:
    def __init__(self, content_type: str, parts: List[str] = None, text=None, language=None, result=None, url=None,
                 domain=None, title=None, summary=None, name=None, **kwargs):
        super(Content, self).__init__()
        self.content_type = content_type
        self.parts = parts
        self.language = language
        self.text = text
        self.result = result
        self.url = url
        self.domain = domain
        self.title = title
        self.summary = summary
        self.name = name
        if len(kwargs) > 0:
            print('Content kwargs', kwargs)

    # def toJSON(self):
    #     return str(super(Content).__str__())

    def __repr__(self):
        return json.dumps(self.__dict__.copy(), ensure_ascii=False)


class Message:
    def __init__(self, message_id: str = None, author: Author = None, create_time: int = None,
                 update_time: float = None, content: Content = None,
                 status: str = None, end_turn: bool = None, weight: float = 1.0, metadata: dict = None,
                 is_complete=False, recipient: str = "all"):
        super(Message, self).__init__()
        self.id = message_id
        self.author = author
        self.create_time = datetime.fromtimestamp(create_time) if create_time else None
        self.update_time = datetime.fromtimestamp(update_time) if update_time else None
        self.content = content
        self.status = status
        self.end_turn = end_turn
        self.weight = weight
        self.metadata = metadata
        self.is_complete = is_complete
        self.recipient = recipient

    # def toJSON(self):
    #     return str(super(Message).__str__())

    def __hash__(self):
        return hash((self.id))

    def __eq__(self, other):
        return self.id == other.id

    def __repr__(self):
        _dict = self.__dict__.copy()
        _dict['author'] = _dict['author'].__repr__() if _dict['author'] is not None else None
        _dict['content'] = _dict['content'].__repr__() if _dict['content'] is not None else None
        _dict['create_time'] = ymdhms_to_timestamp_utc(_dict['create_time']) if _dict[
                                                                                    'create_time'] is not None else None
        _dict['update_time'] = ymdhms_to_timestamp_utc(_dict['update_time']) if _dict[
                                                                                    'update_time'] is not None else None
        _dict['metadata'] = json.dumps(_dict['metadata'], ensure_ascii=False) if _dict['metadata'] is not None else None

        return json.dumps(_dict, ensure_ascii=False)


class Mapping:
    def __init__(self, mapping_id: str, message: Message = None, parent: str = None, children: List[str] = []):
        super(Mapping, self).__init__()
        self.id = mapping_id
        self.message = message if message is not None else Message(message_id=mapping_id, author=Author(role=None),
                                                                   content=Content(content_type="text", parts=[""]),
                                                                   metadata={}, status="default")
        self.parent = parent
        self.children = children

    def new_system_message_mapping(self):
        self.message = Message(message_id=self.id, author=Author(role='system'),
                               content=Content(content_type="text", parts=[""]),
                               metadata={"is_user_system_message": True,
                                         "user_context_message_data": {
                                             "about_model_message": "#zh-TW"}}, status="finished_successfully")
        return self

    def new_user_message_mapping(self, input_prompt=""):
        self.message = Message(message_id=self.id, author=Author(role='user'),
                               content=Content(content_type="text", parts=[input_prompt]),
                               metadata={"timestamp_": "absolute", "message_type": None}, recipient="all")
        return self

    def new_assistant_message_mapping(self, output_prompt=""):
        self.message = Message(message_id=self.id, author=Author(role='assistant'),
                               content=Content(content_type="text", parts=[output_prompt]),
                               metadata={"finish_details": {"type": "stop", "stop_tokens": [0]}, "is_complete": False,
                                         "timestamp_": "absolute"}, recipient="all")
        return self

    def __repr__(self):
        _dict = self.__dict__.copy()
        _dict['message'] = _dict['message'].__repr__() if _dict['message'] is not None else None
        return json.dumps(_dict, ensure_ascii=False)


# class Node:
#     # 初始化每個node的值、parent和children
#     def __init__(self, message, parent=None, children=None):
#         self.message = message
#         self.parent = parent
#         self.children = children
#
#     # 定義一個方法來添加child
#     def add_child(self, child):
#         if self.children is None:
#             self.children = [child]
#         else:
#             self.children.append(child)
#
#     # 定義一個方法來移除child
#     def remove_child(self, child):
#         if self.children is not None and child in self.children:
#             self.children.remove(child)


class Conversation:
    def __init__(self, conversation_id: str,
                 conversation_template_id: str = None, title: str = "new chat",
                 create_time: Union[int, datetime] = None, update_time: Union[int, datetime] = None,
                 mapping: Dict[str, Mapping] = None,
                 moderation_results: List[str] = None, current_node: str = None, plugin_ids: str = None, **kwargs):
        super(Conversation, self).__init__()
        self.id = conversation_id
        self.conversation_id = conversation_id
        self.title = title
        self.create_time = create_time if isinstance(create_time, datetime) else datetime.fromtimestamp(
            create_time) if create_time else None
        self.update_time = update_time if isinstance(update_time, datetime) else datetime.fromtimestamp(
            update_time) if update_time else None
        self.moderation_results = moderation_results
        self.plugin_ids = plugin_ids

        self.conversation_template_id = conversation_template_id

        if current_node is not None:
            self.current_node = current_node
        if mapping is not None:
            self.mapping = mapping

        else:
            self.mapping = OrderedDict()
            mapping0 = Mapping(mapping_id=str(uuid.uuid4()), children=[])
            self.mapping[mapping0.id] = mapping0
            self.current_node = mapping0.id
            self.initialize_mapping()

    @property
    def current_item(self):
        if self.current_node is None:
            item = list(sorted([m for m in list(self.mapping.values()) if m is not None],
                               key=lambda x: x.message.create_time, reverse=True))[0]
            self.current_node = item.id
            return item
        elif self.current_node in self.mapping:
            return self.mapping[self.current_node]
        else:
            return list(sorted([m for m in list(self.mapping.values()) if m is not None],
                               key=lambda x: x.message.create_time, reverse=True))[0]

    def add_mapping(self, item: Mapping):
        self.current_item.children.append(item.id)
        item.parent = self.current_item.id
        self.mapping[item.id] = item
        self.current_node = item.id

    def initialize_mapping(self):
        self.add_mapping(Mapping(mapping_id=str(uuid.uuid4()), parent=None, children=[]).new_system_message_mapping())
        self.add_mapping(Mapping(mapping_id=str(uuid.uuid4()), parent=None, children=[]).new_user_message_mapping(''))

    # def __dir__(self):
    #     keys = super(Conversation, self).__dir__()
    #     keys = [key for key in keys if not key.isdigit()]
    #     return keys

    def get_message_sequences(self, only_final=False):
        # 建立一個字典來儲存每個node的物件
        mapping_list = []
        message_sequences = []
        for mapping_id, mapping in self.mapping.items():
            if (mapping.children is None or len(mapping.children) == 0) and mapping.parent is not None:
                mapping_list.append(mapping)
        if only_final:
            mapping_list = [self.mapping[self.current_node]]
        for mapping in mapping_list:
            series = []
            this_item = mapping
            if this_item.message.status != 'finished_successfully':
                pass
            else:
                series.insert(0, this_item.message)
                while this_item.parent is not None:
                    parent = self.mapping[this_item.parent]
                    if parent.message is not None:
                        series.insert(0, parent.message)
                        this_item = parent
                    else:
                        break
                message_sequences.append(series)

        return message_sequences

    def get_prompt_messages(self, only_final=False):
        message_sequences = self.get_message_sequences(only_final=only_final)

        prompt_messages = []
        for seq in message_sequences:
            prompt_message = []
            for item in seq:
                if item.author.role == 'system':
                    if 'is_user_system_message' in item.metadata and item.metadata['is_user_system_message']:
                        prompt_message.append({"role": item.author.role,
                                               "content": item.metadata['user_context_message_data'][
                                                   'about_model_message']})

                else:
                    if item.content.content_type == 'text':
                        prompt_message.append({"role": item.author.role, "content": '\n'.join(item.content.parts)})
                        if len(item.content.parts) > 1:
                            print(item.content.parts)
                    elif item.content.content_type == 'code':
                        prompt_message.append({"role": item.author.role, "content": item.content.text})

                    else:
                        prompt_message = []
                        print(item.content)
                        break
            if len(prompt_message) > 1:
                prompt_messages.append(prompt_message)
        return prompt_messages

    def get_gradio_chat(self):
        conversations = self.get_prompt_messages(only_final=True)
        conversations = unpack_singleton(conversations)
        if isinstance(conversations, dict):
            conversations = [conversations]
        conversations = [c for c in conversations if c['role'] != 'system']
        return [(process_chat(conversations[i]), process_chat(conversations[i + 1])) for i in
                range(0, len(conversations) - 1, 2) if conversations[i]['role'] != 'system']

    # def toJSON(self):
    #     return str(super(Conversation).__str__())

    def __repr__(self):
        _dict = self.__dict__.copy()

        _dict['mapping'] = json.dumps({k: v.__repr__() for k, v in _dict['mapping'].items()}, ensure_ascii=False) if \
            _dict['mapping'] is not None else None
        _dict['create_time'] = ymdhms_to_timestamp_utc(_dict['create_time']) if _dict[
                                                                                    'create_time'] is not None else None
        _dict['update_time'] = ymdhms_to_timestamp_utc(_dict['update_time']) if _dict[
                                                                                    'update_time'] is not None else None
        return json.dumps(_dict, ensure_ascii=False)


class ConversationHistory:
    def __init__(self):
        super().__init__()
        self.conversations = []
        self._selected_index = None

    @property
    def selected_index(self):
        return self._selected_index

    @selected_index.setter
    def selected_index(self, value):
        self._selected_index = value

    def add(self, conversation: Conversation):
        self.conversations.append(conversation)

    def new_chat(self):
        self.conversations.insert(0, Conversation(conversation_id=str(uuid.uuid4()), create_time=datetime.now(),
                                                  update_time=datetime.now(), mapping=None, current_node=None,
                                                  title="new chat"))
        self._selected_index = 0

    @property
    def selected_item(self):
        if len(self.conversations) == 0 or self._selected_index is None:
            self.new_chat()
        return self.conversations[self._selected_index]

    def get_all_messages(self, only_final=False, flatten=False):
        prompt_messages = [history.get_prompt_messages(only_final=only_final) for history in self.conversations]

        if flatten:
            prompt_messages = list(itertools.chain(*prompt_messages))
        prompt_messages = [item for item in prompt_messages if len(item) > 0]
        return prompt_messages

    # return [ [mapping.message for mapping in history.mapping.values() if
    #          mapping.message is not None and mapping.message.status == 'finished_successfully'] for history in
    #         self.conversations]

    @property
    def titles(self):
        return [[c.title] for c in self.conversations]

    def load(self, data: str):
        data = copy.deepcopy(data)
        if isinstance(data, str):
            data = json.loads(data)
        if isinstance(data, list):
            for item in data:
                self.load(item)
        else:
            mapping_dict = {}
            for key, value in data["mapping"].items():
                author = Author(**value["message"]["author"]) if value["message"] else None
                content = Content(**value["message"]["content"]) if value["message"] else None
                if value["message"] is None:
                    message = None
                else:
                    message_id = value["message"]["id"]
                    value["message"].pop("id")
                    value["message"].pop("author")
                    value["message"].pop("content")
                    message = Message(message_id=message_id, author=author, content=content, **value["message"]) if \
                        value[
                            "message"] else None
                mapping_dict[key] = Mapping(mapping_id=value["id"], message=message,
                                            parent=value["parent"] if "parent" in value else None,
                                            children=value["children"] if "children" in value else None)
            data.pop("mapping")
            cid = data["id"]
            data.pop("id")
            data.pop("conversation_id")
            self.add(Conversation(id=cid, conversation_id=cid, mapping=mapping_dict, **data))

    def save(self, save_path):
        if os.path.exists(save_path):
            folder, filename, ext = split_path(save_path)
            if os.path.exists(os.path.join(folder, filename + '_' + ext)):
                os.remove(os.path.join(folder, filename + '_' + ext))
            os.rename(save_path, os.path.join(folder, filename + '_' + ext))
        with open(save_path, 'w', encoding='utf-8-sig') as f:
            _list = self.conversations.copy()
            _list = [c.__repr__() for c in _list]
            f.write(json.dumps(_list, ensure_ascii=False))

    def __len__(self):
        return len(self.conversations)

    def __repr__(self):
        _list = self.conversations.copy()
        _list = [c.__repr__() for c in _list]
        return json.dumps(_list, ensure_ascii=False)


def initialize_conversation_history():
    _conversation_history_path = os.path.expanduser(os.path.join(cxt.prompt4all_dir, 'conversations.json'))
    _conversation_history = ConversationHistory()
    if cxt.conversation_history is None:
        if os.path.exists(_conversation_history_path):
            with open(_conversation_history_path, 'r', encoding='utf-8-sig') as f:
                _conversation_history.load(json.load(f))
        cxt.conversation_history = _conversation_history
    cxt.databse_schema = open("examples/query_database/schema.txt",
                              encoding="utf-8").read() if cxt.databse_schema is None else cxt.databse_schema
