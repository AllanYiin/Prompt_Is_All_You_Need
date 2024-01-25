from itertools import chain
import re
import time
import os
import copy
from io import StringIO
from dataclasses import dataclass
import threading
from collections import OrderedDict
from typing import List, TypedDict
from itertools import repeat
import string
from abc import ABC, abstractmethod
from typing import Callable
from types import MethodType
import numpy as np
import json
from openai import OpenAI
from prompt4all.common import *
from prompt4all.context import *
from prompt4all import context
from prompt4all.utils.vector_utils import *
import zipfile

client = OpenAI()
client._custom_headers['Accept-Language'] = 'zh-TW'
cxt = context._context()


# class TextFragment(TypedDict):
#     source: str
#     page_num: int
#     paragraph_num: int
#     text: str
#     embeddings: np.ndarray
#     time: float


@dataclass
class TextFragment:
    source: str
    page_num: int
    paragraph_num: int
    text: str
    embeddings: np.ndarray
    time: float

    # def toJSON(self):
    def to_json(self):
        return {
            "source": self.source,
            "page_num": self.page_num,
            "paragraph_num": self.paragraph_num,
            "text": self.text,
            "embeddings": self.embeddings.tolist() if isinstance(self.embeddings, np.ndarray) else self.embeddings,
            "time": self.time,
        }

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.to_json(), indent=4, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str):
        json_dict = json.loads(json_str)
        json_dict['embeddings'] = np.array(json_dict['embeddings'])
        return cls(**json_dict)


def get_embedding(tf: TextFragment):
    text = tf.text
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
        encoding_format="float"
    )
    tf.embeddings = np.array(response.data[0].embedding)
    return tf


def build_text_fragment(source: str, page_num: int, paragraph_num: int, text: str):
    tf = TextFragment(source=source, page_num=page_num, paragraph_num=paragraph_num, text=text, embeddings=None,
                      time=time.time())
    tf = get_embedding(tf)
    return tf


class BaseCache(ABC):
    """Base interface for cache."""

    @abstractmethod
    def lookup(self, query_text: str, top_k: int = 5, min_similarity=0.85):
        """Look up based on prompt and top_k."""

    @abstractmethod
    def update(self, text_fragment: TextFragment) -> None:
        """Update cache based on source and text_fragment."""

    @abstractmethod
    def bulk_update(self, source: str, text_fragments: List[TextFragment]) -> None:
        """Bulk update cache based on source and list of text_fragment."""


class InMemoryCache(BaseCache):
    """Cache that stores things in memory.
    Examples:
        >>> m=InMemoryCache()
        >>> m.update(TextFragment(text='some text', source='http:www.wikipedia.org',embedding= np.array([1.13222,2.64678,3.4567])))
        >>> m.serialize()
        '{"_cache": {"http:www.wikipedia.org": [{"text": "some text", "embeddings": {"__ndarray__": true, "embeddings": "[1.13222,2.64678,3.4567 ]", "dtype": "float64", "shape": [3]}}]}}'

    """

    def __init__(self) -> None:
        """Initialize with empty cache."""
        self._cache: Dict[str, List[TextFragment]] = {}
        self._vector = None
        self._fragments = None

    def lookup(self, query_text: str, top_k: int = 5, min_similarity=0.88):
        """Look up based on prompt and top_k.
        Examples:
            >>> cxt=context._context()
            >>> cxt.memory=InMemoryCache()
            >>> cxt.memory.load()
            >>> cxt.memory.lookup('目前歐盟對於電動車以及充電裝有哪些重要、影響深遠的法案或政策呢?',5)
            '{"_cache": {"http:www.wikipedia.org": [{"text": "some text", "embeddings": {"__ndarray__": true, "embeddings": "[1.13222,2.64678,3.4567 ]", "dtype": "float64", "shape": [3]}}]}}'

        """
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=query_text,
            encoding_format="float"
        )
        question_vector = np.expand_dims(np.array(response.data[0].embedding), 0)
        results = element_cosine_distance(question_vector, self._vector)[0]
        ranking = np.argsort(-results)
        ranked_results = results[ranking]
        filted_results = ranked_results[ranked_results > min_similarity]
        filted_ranking = ranking[ranked_results > min_similarity]
        if top_k > len(ranking):
            top_k = len(ranking)

        if len(filted_results) < top_k:
            filted_ranking = ranking[:top_k]
            filted_results = ranked_results[filted_ranking]

        filted_texts = [self._fragments[i].text for i in filted_ranking.tolist()]
        filted_vector = self._vector[filted_ranking]
        cross_corelation = element_cosine_distance(filted_vector, filted_vector)
        remove_list = []
        for i in range(len(filted_vector)):
            for j in range(i + 1, len(filted_vector)):
                if cross_corelation[i][j] > 0.97:
                    remove_list.append(j)
        filted_ranking = [item for i, item in enumerate(filted_ranking) if i not in remove_list]
        filted_results = [item for i, item in enumerate(filted_results) if i not in remove_list]

        query_results = OrderedDict()
        for k, (idx, prob) in enumerate(zip(filted_ranking, filted_results)):
            query_results[k] = {"similarity": '{0:%}'.format(prob), "text": self._fragments[idx].text,
                                "source": self._fragments[idx].source,
                                "paragraph_num": self._fragments[idx].paragraph_num}

        return query_results

    @property
    def vector(self):
        """Return the vector of the query."""
        return self._vector

    @property
    def cache(self):
        """Return the cache."""
        return self._cache

    @property
    def fragments(self):
        """Return the fragments."""
        return self._fragments

    def _sync(self):
        self._fragments = [tfrag for tfrag in list(chain.from_iterable([v for v in self._cache.values()])) if
                           tfrag is not None and
                           tfrag.embeddings is not None and len(tfrag.embeddings) == 1536]
        if self._fragments is not None and len(self._fragments) > 0:
            self._vector = np.stack([f.embeddings for f in self._fragments])

    def update(self, text_fragment: TextFragment) -> None:
        """Update cache based on source and text_fragment."""
        if text_fragment is None or text_fragment.source is None:
            pass
        else:
            if text_fragment.source not in self._cache:
                self._cache[text_fragment.source] = []
            self._cache[text_fragment.source].append(text_fragment)
            self._sync()

    def bulk_update(self, source: str, text_fragments: List[TextFragment]) -> None:
        """Bulk update cache based on source and list of text_fragment."""
        if source is None or len(text_fragments) == 0:
            pass
        else:
            try:
                if source not in self._cache:
                    self._cache[source] = []

                for i in range(len(text_fragments)):
                    text_fragments[i].source = source

                self._cache[source].extend(text_fragments)

                self._sync()
                self.serialize()
                print(source, 'is update in cache Success!!', flush=True)
            except Exception as e:
                print(source, 'is update in cache Fail!!', flush=True)
                PrintException()

    def dedup_urls(self, urls: List[str]) -> List[str]:
        """Deduplicate urls."""
        return [url for url in urls if url not in self._cache]

    def serialize(self):
        """Serialize the cache."""
        try:
            _dict = copy.deepcopy(self.__dict__)
            _dict.pop('_fragments')
            _dict.pop('_vector')
            for k, v in _dict['_cache'].items():
                for idx in range(len(v)):
                    frag = v[idx]
                    if frag:
                        _dict['_cache'][k][idx] = frag.toJSON()

            f = StringIO()
            f.write(json.dumps(_dict, ensure_ascii=False))
            z = zipfile.ZipFile(os.path.join(cxt.get_prompt4all_dir(), 'cache.zip'), 'w', zipfile.ZIP_DEFLATED)
            z.writestr('cache.json', f.getvalue())
            z.close()

            # with zipfile.ZipFile(os.path.join(cxt.get_prompt4all_dir(), 'cache.zip'), "w",
            #                      compression=zipfile.ZIP_DEFLATED) as zf:
            #     zf.write(os.path.join(cxt.get_prompt4all_dir(), 'cache.json'))
            return json.dumps(_dict, ensure_ascii=False)
        except Exception as e:
            PrintException()

    def load(self):
        """Serialize the cache."""
        if os.path.exists(os.path.join(cxt.get_prompt4all_dir(), 'cache.zip')):
            zip = zipfile.ZipFile(os.path.join(cxt.get_prompt4all_dir(), 'cache.zip'))
            with zip.open('cache.json') as f:
                contents = f.read().decode('utf-8')
                _dict = json.loads(contents)
                _dict['_cache'] = {k: [TextFragment.from_json(frag) for frag in v] for k, v in _dict['_cache'].items()}
                self.__dict__.update(_dict)
                self._sync()
