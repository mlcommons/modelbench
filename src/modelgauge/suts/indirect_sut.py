import threading
from typing import List, Optional
from queue import Queue

import fastapi
import uvicorn
from pydantic import BaseModel

from modelgauge.dynamic_sut_factory import DynamicSUTFactory
from modelgauge.prompt import TextPrompt
from modelgauge.ready import ReadyResponse
from modelgauge.secret_values import InjectSecret
from modelgauge.sut import PromptResponseSUT, SUTResponse, SUTOptions
from modelgauge.suts.openai_client import _USER_ROLE as USER_ROLE
from modelgauge.sut_capabilities import AcceptsTextPrompt
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


class ChatMessage(BaseModel):
    content: str
    role: str


class IndirectSUTRequest(BaseModel):
    """Compatible with the OpenAI standard."""

    request_id: int  # This field is unique to us; can we still call the request OpenAI-compatible?
    model: str  # Required by OpenAI standard
    messages: List[ChatMessage]
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = None


class IndirectSUTResponse(BaseModel):
    request_id: int
    response: str


@modelgauge_sut(capabilities=[AcceptsTextPrompt])
class IndirectSUT(PromptResponseSUT):

    def __init__(self, uid: str, model_name: str):
        super().__init__(uid)
        self._model_name = model_name
        self._server = IndirectSUTServer(7777)
        self._id_generator = ThreadsafeIdGenerator()

    def is_ready(self) -> ReadyResponse:
        # TODO: should we actually do a readiness check?
        return ReadyResponse(True)

    def translate_text_prompt(self, prompt: TextPrompt, options: SUTOptions) -> IndirectSUTRequest:
        messages = [ChatMessage(content=prompt.text, role=USER_ROLE)]
        return IndirectSUTRequest(
            request_id=self._id_generator.next(),
            model=self._model_name,
            messages=messages,
            max_completion_tokens=options.max_tokens,
            temperature=options.temperature,
        )

    def evaluate(self, request: IndirectSUTRequest) -> IndirectSUTResponse:
        return self._server.get_response(request)

    def translate_response(self, request: IndirectSUTRequest, response: IndirectSUTResponse) -> SUTResponse:
        return SUTResponse(text=response.response)

    def run(self):
        self._server.run()


class IndirectSUTServer:
    def __init__(self, port: int):
        self.port = port
        self.prompts = []
        self.queues: dict[int, Queue] = {}
        self.outstanding_requests: dict[int, IndirectSUTRequest] = {}
        self.app = self.make_app()

    def make_app(self):
        # TODO: Auth?
        # Question: Is it weird to use FastAPI here and connexion for BaaS?
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
            # TODO: What happens if there's no queue for a request ID?
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
        # TODO: Auth for fastapi server
        return []

    def make_sut(self, sut_definition: SUTDefinition) -> IndirectSUT:
        sut = IndirectSUT(sut_definition.uid, model_name=sut_definition.external_model_name())
        sut.run()

        return sut
