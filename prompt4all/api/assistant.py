from prompt4all.api.base_api import GptBaseApi
import time
import os
import uuid
from prompt4all import context
from prompt4all.context import *
from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI, RequestOptions
from openai._types import NotGiven, NOT_GIVEN

client = OpenAI()
client._custom_headers['Accept-Language'] = 'zh-TW'
cxt = context._context()


def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run


class Assistant(GptBaseApi):
    def __init__(self, assistant_id, name='MyGPTs', model="gpt-4-1106-preview", instruction=''):
        super().__init__(model)
        self.name = name
        self.assistant_id = assistant_id
        self.instruction = instruction
        self.API_MODEL = None
        self.API_KEY = os.getenv("OPENAI_API_KEY")

        self.change_model(model)

        self.BASE_IDENTITY = uuid.uuid4()
        self.functions = NOT_GIVEN
        self.tools = NOT_GIVEN

        self.current_thread = None
        self.current_run = None
        self.current_runsteps = None

    def create_thread(self):
        _thread = client.beta.threads.create()
        self.current_thread = _thread
        return _thread

    def create_message(self, client, user_message):
        return client.beta.threads.messages.create(
            thread_id=self.current_thread.id,
            role="user",
            content=user_message
        )

    def submit_message(self, user_message):
        client.beta.threads.messages.create(
            thread_id=self.current_thread.id, role="user", content=user_message
        )
        _runs = client.beta.threads.runs.create(
            thread_id=self.current_thread.id,
            assistant_id=self.assistant_id,
        )

    def create_thread_and_run(self, user_input):
        if self.current_thread is None:
            _thread = client.beta.threads.create()
            self.current_thread = _thread

        run = self.submit_message(user_input)
        self.current_run = run
        return _thread, run
