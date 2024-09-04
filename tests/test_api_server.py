import os
from unittest.mock import patch

from starlette.testclient import TestClient  # type: ignore


class TestApiApp:
    def setup_method(self):
        real_getenv = os.getenv
        self.secret_key = "whatever"
        with patch(
            "os.getenv",
            lambda *args: (
                self.secret_key if args[0] == "SECRET_KEY" else real_getenv(*args)
            ),
        ):
            with patch(
                "modelgauge.config.load_secrets_from_config",
                lambda: {"together": {"api_key": "ignored"}},
            ):
                import modelgauge.api_server

                self.client = TestClient(modelgauge.api_server.app)

    def test_get_main(self):
        response = self.client.get("/")
        assert response.status_code == 200

        j = response.json()
        assert "llama_guard_1" in j["annotators"]
        assert "llama-2-13b-chat" in j["suts"]

    def test_post_main_key_required(self):
        response = self.client.post("/")
        assert response.status_code == 403

    def test_post_main_key_must_be_correct(self):
        response = self.client.post(
            "/", json=self.a_request(), headers={"X-key": "wrong key"}
        )
        assert response.status_code == 401

    def a_request(self, prompt=None, sut=None):
        request = {"prompts": [], "suts": [], "annotators": []}
        if prompt:
            request["prompts"].append({"text": prompt})
        if sut:
            request["suts"].append(sut)
        return request

    def test_post_main_empty(self):
        response = self.client.post(
            "/", json=self.a_request(), headers={"X-key": self.secret_key}
        )
        assert response.status_code == 200

    def test_post_main_with_item_and_sut(self):
        with patch("modelgauge.api_server.process_sut_item"):
            response = self.client.post(
                "/",
                json=self.a_request(prompt="hello", sut="llama-2-13b-chat"),
                headers={"X-key": self.secret_key},
            )
            assert response.status_code == 200

    def test_post_main_with_unknown_sut(self):
        with patch("modelgauge.api_server.process_sut_item"):
            response = self.client.post(
                "/",
                json=self.a_request(prompt="hello", sut="doesnotexist"),
                headers={"X-key": self.secret_key},
            )
            assert response.status_code == 422
