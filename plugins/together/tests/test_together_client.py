from unittest import mock

import pytest

from newhelm.suts.together_client import _retrying_post


@mock.patch("requests.Session")
def test_handle_together_400(mock_session):
    """Found in the wild on 2024 Feb 07 when calling particular models with an n of 25"""
    session = mock.MagicMock()
    session.post.return_value.status_code = 400
    session.post.return_value.text = '{"error": {"message": "Input validation error: best_of must be > 0 and <= 2. Given: 25","type": "invalid_request_error","param": null,"code": null}}'
    mock_session.return_value = session
    with pytest.raises(Exception) as e:
        _retrying_post("http://example.org/together", {}, "{}")
    assert "Input validation error" in str(e.value)
