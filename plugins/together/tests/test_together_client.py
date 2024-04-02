import pytest
from newhelm.suts.together_client import _retrying_post
from requests import HTTPError
from unittest import mock


class MockResponse:
    """Bare bones mock of requests.Response"""

    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if 400 <= self.status_code < 500:
            raise HTTPError(f"Status {self.status_code}")


@mock.patch("requests.Session")
def test_handle_together_400(mock_session):
    """Found in the wild on 2024 Feb 07 when calling particular models with an n of 25"""
    session = mock.MagicMock()
    session.post.return_value = MockResponse(
        400,
        '{"error": {"message": "Input validation error: best_of must be > 0 and <= 2. Given: 25","type": "invalid_request_error","param": null,"code": null}}',
    )

    mock_session.return_value = session
    with pytest.raises(Exception) as e:
        _retrying_post("http://example.org/together", {}, "{}")
    assert "Input validation error" in str(e.value)
