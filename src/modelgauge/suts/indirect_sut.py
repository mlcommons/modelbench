import contextlib
import threading
import time
from queue import Queue

import fastapi
import uvicorn
from pydantic import BaseModel

from modelgauge.dynamic_sut_factory import DynamicSUTFactory
from modelgauge.prompt import ChatPrompt, TextPrompt, ChatMessage, ChatRole
from modelgauge.ready import ReadyResponse
from modelgauge.secret_values import InjectSecret
from modelgauge.sut import PromptResponseSUT, RequestType, ResponseType, SUTResponse, SUTOptions
from modelgauge.sut_capabilities import AcceptsChatPrompt, AcceptsTextPrompt
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_definition import SUTDefinition


class ThreadsafeIdGenerator:

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._last_id = 0

    def next(self):
        with self._lock:
            self._last_id += 1
            return self._last_id


class IndirectSUTRequest(BaseModel):
    request_id: int
    prompt: ChatPrompt

    class Config:
        frozen = True


class IndirectSUTResponse(BaseModel):
    request_id: int
    response: str


@modelgauge_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
class IndirectSUT(PromptResponseSUT):

    def __init__(self, uid: str):
        super().__init__(uid)
        self._server = IndirectSUTServer(self, 7777)
        self._id_generator = ThreadsafeIdGenerator()

    def is_ready(self) -> ReadyResponse:
        return ReadyResponse(True)

    def translate_text_prompt(self, prompt: TextPrompt, options: SUTOptions) -> RequestType:
        return self.translate_chat_prompt(
            ChatPrompt(messages=[ChatMessage(text=prompt.text, role=ChatRole.user)]), options
        )

    def translate_chat_prompt(self, prompt: ChatPrompt, options: SUTOptions) -> RequestType:
        # TODO handle options
        return IndirectSUTRequest(request_id=self._id_generator.next(), prompt=prompt)

    def evaluate(self, request: IndirectSUTRequest) -> IndirectSUTResponse:
        return self._server.get_response(request)

    def translate_response(self, request: RequestType, response: IndirectSUTResponse) -> SUTResponse:
        return SUTResponse(text=response.response)

    def run(self):
        self._server.run()


class IndirectSUTServer:
    def __init__(self, sut: IndirectSUT, port: int):
        self.sut = sut
        self.port = port
        self.prompts = []
        self.app = self.make_app()
        self.queues: dict[int, Queue] = {}
        self.outstanding_requests: dict[int, IndirectSUTRequest] = {}

    def make_app(self):
        app = fastapi.FastAPI()

        @app.get("/debug")
        def debug_info():
            return {
                "outstanding_requests": self.outstanding_requests,
                "queues": {n: str(q) for n, q in self.queues.items()},
            }

        @app.get("/requests")
        def get_requests() -> list[IndirectSUTRequest]:
            return list(self.outstanding_requests.values())

        @app.post("/responses")
        def post_responses(responses: list[IndirectSUTResponse]):
            for response in responses:
                self.queues[response.request_id].put(response)

        return app

    def get_response(self, request: IndirectSUTRequest) -> IndirectSUTResponse:
        request_id = request.request_id
        self.outstanding_requests[request_id] = request
        my_queue = Queue()
        self.queues[request_id] = my_queue
        response = my_queue.get()
        del self.outstanding_requests[request_id]
        del self.queues[request_id]
        return response

    def run(self):

        def start():
            uvicorn.run(self.app, host="0.0.0.0", port=self.port, use_colors=False)

        thread = threading.Thread(target=start, daemon=True)
        thread.start()


class IndirectSUTFactory(DynamicSUTFactory):

    def get_secrets(self) -> list[InjectSecret]:
        return []

    def make_sut(self, sut_definition: SUTDefinition) -> IndirectSUT:
        sut = IndirectSUT(sut_definition.uid)
        sut.run()

        return sut
