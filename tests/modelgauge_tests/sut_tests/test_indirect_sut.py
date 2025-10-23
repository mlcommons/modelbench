import pytest
from fastapi.testclient import TestClient
from queue import Queue
from unittest.mock import patch, MagicMock


from modelgauge.prompt import TextPrompt
from modelgauge.sut import SUTOptions
from modelgauge.suts.indirect_sut import (
    ChatMessage,
    IndirectSUT,
    IndirectSUTRequest,
    IndirectSUTResponse,
    IndirectSUTServer,
    USER_ROLE,
)


@pytest.fixture
def sut_request():
    messages = [ChatMessage(content="text", role=USER_ROLE)]
    return IndirectSUTRequest(request_id=1, messages=messages, model="model-name")


class TestSUT:
    @pytest.fixture
    def sut(self):
        fake_server = MagicMock()
        fake_server.get_response.return_value = IndirectSUTResponse(request_id=1, response="ok")

        with patch("modelgauge.suts.indirect_sut.IndirectSUTServer", return_value=fake_server):
            return IndirectSUT(uid="fake-sut", model_name="model-name")

    def test_translate_text_prompt(self, sut):
        prompt = TextPrompt(text="text")
        options = SUTOptions(max_tokens=20, temperature=0.3)

        request = sut.translate_text_prompt(prompt, options)

        assert request == IndirectSUTRequest(
            request_id=1,
            model="model-name",
            messages=[ChatMessage(content="text", role=USER_ROLE)],
            max_completion_tokens=20,
            temperature=0.3,
        )

        request = sut.translate_text_prompt(prompt, options)
        assert request.request_id == 2

    def test_evaluate(self, sut, sut_request):
        response = sut.evaluate(sut_request)

        assert isinstance(response, IndirectSUTResponse)
        assert response.request_id == 1
        assert response.response == "ok"

    def test_translate_response(self, sut, sut_request):
        response = IndirectSUTResponse(request_id=1, response="ok")

        sut_response = sut.translate_response(sut_request, response)

        assert sut_response.text == "ok"


class TestServer:
    @pytest.fixture
    def server(self):
        return IndirectSUTServer(8000)

    @pytest.fixture
    def client(self, server):
        return TestClient(server.app)

    def test_requests(self, client, server, sut_request):
        client_response = client.get("/requests")
        assert client_response.status_code == 200
        assert client_response.json() == []

        server.outstanding_requests[sut_request.request_id] = sut_request
        for _ in range(2):
            # Make sure we keep the request in the outstanding requests
            client_response = client.get("/requests")
            assert client_response.status_code == 200
            assert client_response.json() == [
                {
                    "request_id": 1,
                    "model": "model-name",
                    "messages": [{"content": "text", "role": "user"}],
                    "max_completion_tokens": None,
                    "temperature": None,
                }
            ]

    def test_responses(self, client, server, sut_request):
        response_data = [{"request_id": 1, "response": "ok"}]

        server.queues[1] = Queue()

        client_response = client.post("/responses", json=response_data)
        assert client_response.status_code == 200
        assert server.queues[1].get() == IndirectSUTResponse(request_id=1, response="ok")

    def test_get_response(self, client, server, sut_request):
        # TODO: Add more edge cases?
        fake_response = IndirectSUTResponse(request_id=1, response="ok")
        fake_queue = MagicMock()
        fake_queue.get.return_value = fake_response

        with patch("modelgauge.suts.indirect_sut.Queue", return_value=fake_queue):
            response = server.get_response(sut_request)

        # Make sure it was removed from the outstanding requests
        assert len(server.outstanding_requests) == 0

        assert isinstance(response, IndirectSUTResponse)
        assert response.request_id == 1
        assert response.response == "ok"
