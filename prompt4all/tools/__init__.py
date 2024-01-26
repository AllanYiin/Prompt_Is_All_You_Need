import prompt4all.tools.database_tools

query_sql = database_tools.query_sql

import inspect
import uuid
import warnings
from abc import ABC
from functools import partial
from typing import Callable
from types import MethodType
import numpy as np
from tqdm.auto import tqdm
from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI, RequestOptions

__all__ = ['BaseTool', 'LambdaCallback']

from prompt4all import context
from prompt4all.context import *

ctx = context._context()

_valid_when = ["on_use_tool_start"
    , "on_use_tool_end"
    , "on_prepare_output_start"
    , "on_prepare_output_end"
    , "on_prepare_output_fail"
    , "on_second_chat_start"
    , "on_second_chat_end"
    , "on_infer_start"
    , "on_infer_end"
    , "on_excution_exception"]


class BaseTool(ABC):
    """
    Objects of derived classes inject functionality in several points of the training process.
    """

    def __init__(self, icon=None, fn=None, api_model="gpt-4-1106-preview"):
        self.icon = icon
        self.uuid = str(uuid.uuid4())[:8].__str__().replace('-', '')
        self.execution_context = {}
        self.version = '0.0.0.1'
        self.client = OpenAI()
        self.api_model = api_model
        if isinstance(fn, Callable):
            self._func = fn

    def __eq__(self, other):
        return self.uuid == other.uuid

    @property
    def fn(self):
        return self._func

    def __call__(self, execution_context, **kwargs):
        execution_context.update(**kwargs)
        self.on_use_tool_start(execution_context)

        self.on_prepare_output_start(execution_context)
        output = self._func(**kwargs)
        self.on_prepare_output_end(execution_context)
        self.on_second_chat_start(execution_context)
        self.do_second_chat(execution_context)
        self.on_second_chat_end(execution_context)
        self.on_use_tool_end(execution_context)

    def do_second_chat(self, execution_context):
        second_response = self.client.chat.completions.create(
            model=self.api_model,
            messages=full_history,
            stream=True,
            temperature=0.1,
            n=1,
            tools=self.tools,
            tool_choice="none"
        )
        for second_chunk in second_response:
            this_second_choice = chunk_message = second_chunk.choices[0]
            this_second_delta = this_second_choice.delta
            finish_reason = this_second_choice.finish_reason
            if not this_second_delta:
                break
            elif this_second_delta and this_second_delta.content:
                partial_words += this_second_delta.content
                for i in range(len(self.temp_state)):
                    if self.temp_state[-i]['role'] == 'assistant':
                        self.temp_state[-i]['content'] = partial_words
                        break

    def on_use_tool_start(self, execution_context):
        """
        Called at the beginning of the training process.
        :param execution_context: Dict containing information regarding the training process.
        """
        pass

    def on_use_tool_end(self, execution_context):
        """
        Called at the end of the training process.
        :param execution_context: Dict containing information regarding the training process.
        """
        pass

    def on_second_chat_start(self, execution_context):
        """
        Called at the end of the training process.
        :param execution_context: Dict containing information regarding the training process.
        """
        pass

    def on_second_chat_end(self, execution_context):
        """

        Called at the beginning of a new epoch.



        :param execution_context: Dict containing information regarding the training process.

        """

        pass

    def on_prepare_output_start(self, execution_context):
        """
        Called at the end of an epoch.



        :param execution_context: Dict containing information regarding the training process.

        """

        pass

    def on_prepare_output_end(self, execution_context):
        """
        Called after a batch has been processed.



        :param execution_context: Dict containing information regarding the training process.

        """

        pass

    def on_prepare_output_fail(self, execution_context):
        """
        Called after a batch has been processed.



        :param execution_context: Dict containing information regarding the training process.

        """

        pass

    def on_excution_exception(self, training_context):
        """
        Called when the expection occure.



        :param training_context: Dict containing information regarding the training process.

        """

        pass


class LambdaTool(BaseTool):
    """
    Objects of derived classes inject functionality in several points of the training process.
    """

    def __init__(self, when='on_batch_end', frequency=1, unit='batch', action=None, is_shared=False):
        super(LambdaTool, self).__init__(is_shared=is_shared)
        self.is_shared = is_shared

        self.frequency = frequency if frequency is not None else 1
        self.action = None
        if action is None:
            raise ValueError("action cannot be None")
        argspec = inspect.getfullargspec(action)
        if 'execution_context' in argspec.args and len(argspec.args) == 1:
            self.action = action
        else:
            raise ValueError("action should has only-one argment 'execution_context")
        if when in _valid_when:
            self.when = when
        else:
            raise ValueError("{0} is not valid event trigger.".format(when))
        self.unit = unit

        def on_trigger(self, execution_context):
            steps = execution_context['steps']
            epoch = execution_context['current_epoch']
            if self.unit == 'epoch' and (epoch + 1) % (self.frequency) == 0:
                if ctx.amp_available == True and ctx.is_autocast_enabled == True and get_device() == 'cuda':
                    with torch.cuda.amp.autocast():
                        self.action(execution_context)
                else:
                    self.action(execution_context)
            elif self.unit == 'batch' and (steps + 1) % (self.frequency) == 0:
                if ctx.amp_available == True and ctx.is_autocast_enabled == True and get_device() == 'cuda':
                    with torch.cuda.amp.autocast():
                        self.action(execution_context)
                else:
                    self.action(execution_context)

        setattr(self, when, MethodType(on_trigger, self))
